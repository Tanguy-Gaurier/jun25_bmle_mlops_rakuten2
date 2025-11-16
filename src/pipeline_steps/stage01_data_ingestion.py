"""
Étape 1 : Ingestion des données (Data Ingestion) - Version PostgreSQL corrigée.
==============================================================================

Cette étape gère le chargement des données depuis CSV OU PostgreSQL.
S'adapter à la structure project.items 

Responsabilités :
- Charger depuis CSV (mode legacy/dev) ou PostgreSQL (mode production)
- Créer un row_index technique après chargement pour compatibilité
- Valider la cohérence des données
- Afficher des statistiques de base

Utilisation:
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    
    # Mode CSV (par défaut)
    pipeline = DataIngestionPipeline(config, source="csv")
    X_train, y_train, X_test = pipeline.run()
    
    # Mode PostgreSQL
    pipeline = DataIngestionPipeline(config, source="postgres")
    X_train, y_train, X_test = pipeline.run()
"""
from __future__ import annotations

import logging
from typing import Tuple, Literal, Optional

import pandas as pd

from src.data.load_data import (
    load_train_data,
    load_test_data,
    validate_dataframes,
    check_missing_values
)
from src.utils.profiling import Timer

logger = logging.getLogger(__name__)

# Import conditionnel pour PostgreSQL
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.debug("psycopg2 non disponible - mode PostgreSQL désactivé")


class DataIngestionPipeline:
    """
    Pipeline d'ingestion des données hybride (CSV ou PostgreSQL).
    
    Charge les données d'entraînement et de test depuis CSV ou PostgreSQL.
    VERSION CORRIGÉE : Adapté à la structure project.items (sans row_index).
    
    Attributes:
        config: Configuration complète du projet
        source: Source des données ('csv' ou 'postgres')
        
    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> 
        >>> # Mode CSV
        >>> pipeline = DataIngestionPipeline(config, source="csv")
        >>> X_train, y_train, X_test = pipeline.run()
        >>> 
        >>> # Mode PostgreSQL
        >>> pipeline = DataIngestionPipeline(config, source="postgres")
        >>> X_train, y_train, X_test = pipeline.run()
    """
    
    def __init__(
        self, 
        config,
        source: Literal["csv", "postgres"] = "csv",
        db_config: Optional[dict] = None
    ):
        """
        Initialise le pipeline d'ingestion.
        
        Args:
            config: Objet Config contenant tous les paramètres
            source: Source des données ('csv' ou 'postgres')
            db_config: Configuration PostgreSQL (optionnel)
        """
        self.config = config
        self.source = source
        self.db_config = db_config or self._get_default_db_config()
        
        logger.info("=" * 70)
        logger.info("ÉTAPE 1 : INGESTION DES DONNÉES")
        logger.info(f"Source: {self.source.upper()}")
        logger.info("=" * 70)
        
        # Vérifier que PostgreSQL est disponible si nécessaire
        if self.source == "postgres" and not POSTGRES_AVAILABLE:
            raise ImportError(
                "psycopg2 n'est pas installé. "
                "Installez-le avec: pip install psycopg2-binary"
            )
    
    def _get_default_db_config(self) -> dict:
        """Retourne la configuration DB par défaut."""
        return {
            "host": self.config.get("database.host", "localhost"),
            "port": self.config.get("database.port", 5433),
            "database": self.config.get("database.name", "rakuten"),
            "user": self.config.get("database.user", "mlops"),
            "password": self.config.get("database.password", "mlops")
        }
    
    def _load_from_csv(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Charge les données depuis CSV (mode legacy)."""
        logger.info("\n--- Chargement depuis CSV ---")
        
        # Charger les données d'entraînement
        X_train, y_train = load_train_data(
            x_train_path=self.config.paths["x_train_csv"],
            y_train_path=self.config.paths["y_train_csv"]
        )
        
        # Charger les données de test
        X_test = load_test_data(
            x_test_path=self.config.paths["x_test_csv"]
        )
        
        return X_train, y_train, X_test
    
    def _get_db_connection(self):
        """Crée une connexion PostgreSQL."""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            raise ConnectionError(f"Échec de connexion à PostgreSQL: {e}")
    
    def _load_from_postgres(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Charge les données depuis PostgreSQL (mode production).
        
        CORRECTION IMPORTANTE :
        - project.items n'a PAS de row_index
        - On crée un row_index technique après chargement pour compatibilité
        - Structure réelle: (productid, imageid, designation, description, prdtypecode)
        """
        logger.info("\n--- Chargement depuis PostgreSQL ---")
        
        conn = self._get_db_connection()
        
        try:
            # Récupérer le hash du dernier dataset
            cursor = conn.cursor()
            cursor.execute("""
                SELECT data_hash 
                FROM project.datasets 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            if result:
                dataset_hash = result[0]
                logger.info(f"Dataset hash: {dataset_hash[:12]}...")
            else:
                logger.warning("Aucun dataset trouvé dans project.datasets")
            
            cursor.close()
            
            # ================================================================
            # CHARGER TRAIN (avec labels) - SANS row_index
            # ================================================================
            logger.info("\nChargement des données d'entraînement...")
            query_train = """
                SELECT 
                    productid,
                    imageid,
                    designation,
                    description,
                    prdtypecode
                FROM project.items
                WHERE prdtypecode IS NOT NULL
                ORDER BY productid
            """
            df_train = pd.read_sql_query(query_train, conn)
            
            # Séparer features et labels
            X_train = df_train.drop(columns=['prdtypecode'])
            y_train = df_train['prdtypecode']
            
            #  CRÉER un row_index technique pour compatibilité avec le reste du code
            X_train = X_train.reset_index().rename(columns={'index': 'row_index'})
            
            logger.info(f" X_train chargé: {X_train.shape}")
            logger.info(f" y_train chargé: {y_train.shape}")
            logger.info(f"  Colonnes: {list(X_train.columns)}")
            
            # ================================================================
            # CHARGER TEST (sans labels) - SANS row_index
            # ================================================================
            logger.info("\nChargement des données de test...")
            query_test = """
                SELECT 
                    productid,
                    imageid,
                    designation,
                    description
                FROM project.items
                WHERE prdtypecode IS NULL
                ORDER BY productid
            """
            X_test = pd.read_sql_query(query_test, conn)
            
            # Si pas de données de test dans project.items, fallback sur CSV
            if len(X_test) == 0:
                logger.warning(" Aucune donnée de test dans PostgreSQL")
                logger.info("  Tentative de chargement depuis CSV...")
                X_test = load_test_data(
                    x_test_path=self.config.paths["x_test_csv"]
                )
            
            #  CRÉER un row_index technique pour compatibilité
            X_test = X_test.reset_index().rename(columns={'index': 'row_index'})
            
            logger.info(f" X_test chargé: {X_test.shape}")
            logger.info(f"  Colonnes: {list(X_test.columns)}")
            
            return X_train, y_train, X_test
            
        finally:
            conn.close()
    
    def run(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Exécute le pipeline d'ingestion complet.
        
        Returns:
            Tuple (X_train, y_train, X_test)
            
        Raises:
            FileNotFoundError: Si les fichiers CSV n'existent pas (mode csv)
            ConnectionError: Si la connexion PostgreSQL échoue (mode postgres)
            ValueError: Si les données sont incohérentes
        """
        with Timer("Ingestion des données"):
            # ========================================
            # 1. Charger les données selon la source
            # ========================================
            if self.source == "csv":
                X_train, y_train, X_test = self._load_from_csv()
            elif self.source == "postgres":
                X_train, y_train, X_test = self._load_from_postgres()
            else:
                raise ValueError(
                    f"Source invalide: {self.source}. "
                    "Utilisez 'csv' ou 'postgres'."
                )
            
            # ========================================
            # 2. Valider la compatibilité
            # ========================================
            logger.info("\n--- Validation de la compatibilité ---")
            validate_dataframes(X_train, X_test)
            
            # ========================================
            # 3. Analyser les valeurs manquantes
            # ========================================
            logger.info("\n--- Analyse des valeurs manquantes ---")
            check_missing_values(X_train, "X_train")
            check_missing_values(X_test, "X_test")
            
            # ========================================
            # 4. Résumé final
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info("RÉSUMÉ DE L'INGESTION")
            logger.info("=" * 70)
            logger.info(f"Source: {self.source.upper()}")
            logger.info(f" X_train : {X_train.shape}")
            logger.info(f" y_train : {y_train.shape}")
            logger.info(f" X_test  : {X_test.shape}")
            logger.info(f"  Colonnes : {list(X_train.columns)}")
            logger.info(f"  Classes : {y_train.nunique()}")
            
            if self.source == "postgres":
                logger.info(f"  Database: {self.db_config['database']}@{self.db_config['host']}")
            
            logger.info("=" * 70 + "\n")
            
            return X_train, y_train, X_test


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.config import load_config
    
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Test de DataIngestionPipeline (Version Corrigée)")
    print("="*70 + "\n")
    
    try:
        # Charger la configuration
        config = load_config()
        
        # Test mode CSV
        print(">>> Test 1: Mode CSV")
        pipeline_csv = DataIngestionPipeline(config, source="csv")
        X_train_csv, y_train_csv, X_test_csv = pipeline_csv.run()
        print(f" CSV: X_train={X_train_csv.shape}, y_train={y_train_csv.shape}")
        
        # Test mode PostgreSQL (si disponible)
        if POSTGRES_AVAILABLE:
            print("\n>>> Test 2: Mode PostgreSQL")
            try:
                pipeline_pg = DataIngestionPipeline(config, source="postgres")
                X_train_pg, y_train_pg, X_test_pg = pipeline_pg.run()
                print(f" PostgreSQL: X_train={X_train_pg.shape}, y_train={y_train_pg.shape}")
            except ConnectionError as e:
                print(f" PostgreSQL non disponible: {e}")
        else:
            print("\n psycopg2 non installé - mode PostgreSQL ignoré")
        
        print("\n Tests terminés avec succès!")
        
    except FileNotFoundError as e:
        print(f"\n Erreur: {e}")
        print("S'assurer que les fichiers CSV existent dans data/raw/")
    except Exception as e:
        print(f"\n Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
























