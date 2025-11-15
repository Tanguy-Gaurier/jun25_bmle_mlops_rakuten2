#!/usr/bin/env python3
"""
Script de prédiction Rakuten - Intégré avec l'architecture modulaire.
=========================================================================

Ce script charge un modèle entraîné et génère des prédictions sur de nouvelles données.
Il s'intègre avec un pipeline existant (stages 1-5) et supporte CSV ou PostgreSQL.

Fonctionnalités :
- Charge modèle depuis fichier ou PostgreSQL registry
- Supporte CSV ou PostgreSQL en entrée
- Utilise un pipeline de transformation (stage03)
- Génère prédictions + probabilités + top-3 classes
- Sauvegarde dans PostgreSQL (optionnel)

Utilisation:
    # Prédiction sur CSV
    python scripts/predict.py --model-path models/rakuten_xgb_final.pkl
    
    # Prédiction depuis PostgreSQL
    python scripts/predict.py --model-id 1 --source postgres
    
    # Avec sauvegarde en base
    python scripts/predict.py --model-id 1 --save-to-db
    python scripts/predict.py --model-path models/model_full_pipeline.joblib --source postgres --output results/predictions/predictions_postgres.csv --verbose
"""
import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.config import load_config
from src.utils.profiling import Timer

# Import conditionnel pour PostgreSQL
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logging.warning("psycopg2 non disponible - mode PostgreSQL désactivé")

logger = logging.getLogger(__name__)


# =============================================================================
# Utilitaires PostgreSQL
# =============================================================================

class PostgresModelLoader:
    """Charge les modèles depuis PostgreSQL."""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
    
    def get_connection(self):
        """Crée une connexion PostgreSQL."""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            raise ConnectionError(f"Échec de connexion à PostgreSQL: {e}")
    
    def load_model_metadata(self, model_id: Optional[int] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Charge les métadonnées d'un modèle depuis project.models.
        
        Args:
            model_id: ID du modèle (prioritaire)
            model_name: Nom du modèle (si model_id non fourni)
        
        Returns:
            Dictionnaire avec métadonnées du modèle
        """
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if model_id:
                query = "SELECT * FROM project.models WHERE model_id = %s"
                cursor.execute(query, (model_id,))
            elif model_name:
                query = """
                    SELECT * FROM project.models 
                    WHERE model_name = %s 
                    ORDER BY trained_at DESC 
                    LIMIT 1
                """
                cursor.execute(query, (model_name,))
            else:
                # Charger le dernier modèle
                query = """
                    SELECT * FROM project.models 
                    ORDER BY trained_at DESC 
                    LIMIT 1
                """
                cursor.execute(query)
            
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                raise ValueError("Aucun modèle trouvé dans project.models")
            
            return dict(result)
            
        finally:
            conn.close()
    
    def save_predictions(
        self, 
        model_id: int, 
        predictions: pd.Series,
        confidences: pd.Series,
        product_ids: Optional[pd.Series] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """Sauvegarde les prédictions dans project.predictions."""
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            for i in range(len(predictions)):
                product_id = int(product_ids.iloc[i]) if product_ids is not None else None
                
                cursor.execute("""
                    INSERT INTO project.predictions 
                    (model_id, product_id, predicted_class, confidence, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    model_id,
                    product_id,
                    int(predictions.iloc[i]),
                    float(confidences.iloc[i]),
                    json.dumps(metadata or {})
                ))
            
            conn.commit()
            logger.info(f" {len(predictions)} prédictions sauvegardées dans project.predictions")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Échec de sauvegarde des prédictions: {e}")
            raise
        finally:
            cursor.close()
            conn.close()


def load_data_from_postgres(db_config, dataset_type="test"):
    """
    Charge les données depuis PostgreSQL.
    dataset_type = "test" -> items sans label (prdtypecode IS NULL)
    """
    logger.info("Chargement des données depuis PostgreSQL (%s)...", dataset_type)

    conn = psycopg2.connect(
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
    )

    try:
        if dataset_type == "test":
            query = """
                SELECT
                    designation,
                    description,
                    productid,
                    imageid
                FROM project.items
                WHERE prdtypecode IS NULL
                ORDER BY productid
            """
        else:
            raise ValueError(f"dataset_type inconnu: {dataset_type}")

        df = pd.read_sql_query(query, conn)

        # On recrée un row_index côté Pandas comme dans training.py
        df = df.reset_index().rename(columns={"index": "row_index"})

        logger.info(" Données chargées depuis PostgreSQL")
        logger.info("  Shape: %s", df.shape)
        logger.debug("  Colonnes: %s", list(df.columns))

        return df

    finally:
        conn.close()


# =============================================================================
# Chargement du modèle
# =============================================================================

def load_model_and_pipeline(
    model_path: Optional[str] = None,
    model_id: Optional[int] = None,
    model_name: Optional[str] = None,
    db_config: Optional[dict] = None
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Charge le modèle et le pipeline de features.
    
    Args:
        model_path: Chemin vers le modèle .pkl
        model_id: ID du modèle dans PostgreSQL
        model_name: Nom du modèle dans PostgreSQL
        db_config: Configuration PostgreSQL
    
    Returns:
        Tuple (model, feature_pipeline, metadata)
    """
    metadata = {}
    
    # Option 1 : Charger depuis PostgreSQL
    if (model_id or model_name) and db_config and POSTGRES_AVAILABLE:
        logger.info("\n--- Chargement du modèle depuis PostgreSQL ---")
        
        loader = PostgresModelLoader(db_config)
        model_metadata = loader.load_model_metadata(model_id=model_id, model_name=model_name)
        
        model_path_db = model_metadata['model_path']
        logger.info(f"Modèle: {model_metadata['model_name']}")
        logger.info(f"Type: {model_metadata['model_type']}")
        logger.info(f"Entraîné le: {model_metadata['trained_at']}")
        logger.info(f"Métriques: {model_metadata['metrics']}")
        
        model_path = model_path_db
        metadata = model_metadata
    
    # Option 2 : Charger depuis fichier local
    elif model_path:
        logger.info(f"\n--- Chargement du modèle depuis fichier ---")
        logger.info(f"Chemin: {model_path}")
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    else:
        raise ValueError(
            "Vous devez fournir soit --model-path, soit --model-id/--model-name avec --source postgres"
        )
    
    # Charger le modèle
    model_path = Path(model_path)
    logger.info(f"Chargement du modèle: {model_path}")
    model = joblib.load(model_path)
    logger.info(" Modèle chargé")
    
    # Charger le pipeline (convention: même nom avec suffix _pipeline)
    pipeline_path = model_path.parent / f"{model_path.stem}_pipeline.pkl"
    
    if pipeline_path.exists():
        logger.info(f"Chargement du pipeline: {pipeline_path}")
        feature_pipeline = joblib.load(pipeline_path)
        logger.info(" Pipeline chargé")
    else:
        logger.warning("Pipeline non trouvé: %s", pipeline_path)
        # Nouveau : gérer le cas où le fichier contient un dict (full pipeline)
        feature_pipeline = None

        if isinstance(model, dict):
            logger.info("Objet modèle détecté comme dictionnaire, extraction du pipeline et du modèle...")
            feature_pipeline = model.get("feature_pipeline")
            inner_model = model.get("model")

            if inner_model is not None:
                model = inner_model
                logger.info("  Modèle interne récupéré depuis le dictionnaire.")
            else:
                logger.warning("  Aucun modèle interne trouvé dans le dictionnaire, on garde l'objet tel quel.")

            if feature_pipeline is not None:
                logger.info("  Pipeline de features embarqué récupéré depuis le dictionnaire.")
            else:
                logger.warning("  Aucun pipeline de features embarqué trouvé, on utilisera les features brutes.")
        else:
            logger.warning("Les features devront être fournies pré-transformées")
    
    return model, feature_pipeline, metadata


# =============================================================================
# Prédictions
# =============================================================================

def make_predictions(
    model: Any,
    X: np.ndarray,
    include_proba: bool = True,
    top_n: int = 3
) -> Dict[str, Any]:
    """
    Génère les prédictions.
    
    Args:
        model: Modèle entraîné
        X: Features transformées
        include_proba: Inclure les probabilités
        top_n: Nombre de top classes à retourner
    
    Returns:
        Dictionnaire avec prédictions, confidences, top classes
    """
    logger.info("\n--- Génération des prédictions ---")
    
    with Timer("Prédictions"):
        # Prédictions
        y_pred = model.predict(X)
        logger.info(f" {len(y_pred)} prédictions générées")
        
        results = {
            'predictions': y_pred,
            'n_samples': len(y_pred)
        }
        
        # Probabilités
        if include_proba and hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            confidences = y_proba.max(axis=1)
            results['confidences'] = confidences
            results['mean_confidence'] = float(confidences.mean())
            
            logger.info(f" Confidence moyenne: {confidences.mean():.4f}")
            
            # Top N classes
            if hasattr(model, 'classes_'):
                top_n_indices = np.argsort(y_proba, axis=1)[:, -top_n:][:, ::-1]
                
                top_classes = []
                for i in range(len(y_proba)):
                    sample_top = [
                        {
                            'class': int(model.classes_[idx]),
                            'probability': float(y_proba[i, idx])
                        }
                        for idx in top_n_indices[i]
                    ]
                    top_classes.append(sample_top)
                
                results['top_classes'] = top_classes
        
        # Distribution des prédictions
        unique, counts = np.unique(y_pred, return_counts=True)
        results['class_distribution'] = dict(zip(map(int, unique), map(int, counts)))
        
        logger.info(f" {len(unique)} classes différentes prédites")
    
    return results


# =============================================================================
# Sauvegarde des résultats
# =============================================================================

def save_predictions_to_csv(
    predictions: np.ndarray,
    confidences: Optional[np.ndarray],
    top_classes: Optional[list],
    original_data: pd.DataFrame,
    output_path: str
) -> None:
    """Sauvegarde les prédictions dans un CSV."""
    
    results_df = pd.DataFrame({
        'row_index': original_data.get('row_index', range(len(predictions))),
        'productid': original_data.get('productid', None),
        'prediction': predictions
    })
    
    if confidences is not None:
        results_df['confidence'] = confidences
    
    if top_classes is not None:
        # Ajouter top 3 classes
        for i in range(min(3, len(top_classes[0]) if top_classes else 0)):
            results_df[f'top{i+1}_class'] = [tc[i]['class'] for tc in top_classes]
            results_df[f'top{i+1}_proba'] = [tc[i]['probability'] for tc in top_classes]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    logger.info(f" Prédictions sauvegardées: {output_path}")


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Prédictions Rakuten avec intégration PostgreSQL"
    )
    
    # Source des données
    parser.add_argument(
        "--source",
        type=str,
        choices=["csv", "postgres"],
        default="csv",
        help="Source des données (csv ou postgres)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Chemin vers le CSV d'entrée (si source=csv)"
    )
    
    # Modèle
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-path",
        type=str,
        help="Chemin vers le modèle .pkl"
    )
    model_group.add_argument(
        "--model-id",
        type=int,
        help="ID du modèle dans PostgreSQL"
    )
    model_group.add_argument(
        "--model-name",
        type=str,
        help="Nom du modèle dans PostgreSQL"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.toml",
        help="Chemin vers le fichier de configuration"
    )
    
    # Options
    parser.add_argument(
        "--output",
        type=str,
        default="results/predictions/predictions.csv",
        help="Chemin de sortie pour les prédictions"
    )
    
    parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Sauvegarder les prédictions dans PostgreSQL"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode verbeux (DEBUG)"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Nombre de top classes à retourner (défaut: 3)"
    )
    
    return parser.parse_args()


# =============================================================================
# Pipeline principal
# =============================================================================

def main():
    """Fonction principale."""
    args = parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logger.info("\n" + "=" * 70)
    logger.info("SCRIPT DE PRÉDICTION RAKUTEN")
    logger.info("=" * 70)
    logger.info(f"Source: {args.source.upper()}")
    
    # Charger la configuration
    try:
        config = load_config(args.config)
        logger.info(f" Configuration chargée: {args.config}")
        
        # Configuration DB
        db_config = None
        if args.source == "postgres" or args.save_to_db or args.model_id or args.model_name:
            if not POSTGRES_AVAILABLE:
                logger.error("psycopg2 n'est pas installé!")
                logger.error("Installez-le avec: pip install psycopg2-binary")
                return 1
            
            db_config = {
                "host": config.get("database.host", "localhost"),
                "port": config.get("database.port", 5433),
                "database": config.get("database.name", "rakuten"),
                "user": config.get("database.user", "mlops"),
                "password": config.get("database.password", "mlops")
            }
            logger.info(f"Database: {db_config['database']}@{db_config['host']}")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        return 1
    
    try:
        # ================================================
        # ÉTAPE 1 : Charger le modèle
        # ================================================
        model, feature_pipeline, model_metadata = load_model_and_pipeline(
            model_path=args.model_path,
            model_id=args.model_id,
            model_name=args.model_name,
            db_config=db_config
        )
        
        # ================================================
        # ÉTAPE 2 : Charger les données
        # ================================================
        logger.info("\n" + "=" * 70)
        logger.info("CHARGEMENT DES DONNÉES")
        logger.info("=" * 70)
        
        if args.source == "csv":
            input_path = args.input or config.paths.get("x_test_csv")
            if not input_path:
                logger.error("Vous devez fournir --input ou configurer x_test_csv")
                return 1
            
            logger.info(f"Chargement depuis CSV: {input_path}")
            X_raw = pd.read_csv(input_path)
            logger.info(f" {len(X_raw)} échantillons chargés")
            
        elif args.source == "postgres":
            logger.info("Chargement depuis PostgreSQL...")
            X_raw = load_data_from_postgres(db_config, dataset_type="test")
        
        # ================================================
        # ÉTAPE 3 : Transformation des features
        # ================================================
        logger.info("\n" + "=" * 70)
        logger.info("TRANSFORMATION DES FEATURES")
        logger.info("=" * 70)

        if feature_pipeline is None:
            # Cas 1 : le fichier chargé est déjà un pipeline complet (préprocessing + modèle)
            logger.warning("Aucun pipeline de features séparé détecté.")
            logger.warning("On suppose que le modèle chargé est un pipeline complet (features + modèle).")
            X_transformed = X_raw
        else:
            # Cas 2 : modèle simple + pipeline de features séparé
            with Timer("Transformation des features"):
                X_transformed = feature_pipeline.transform(X_raw)
                logger.info(f" Features transformées: {X_transformed.shape}")
        
        # ================================================
        # ÉTAPE 4 : Prédictions
        # ================================================
        logger.info("\n" + "=" * 70)
        logger.info("GÉNÉRATION DES PRÉDICTIONS")
        logger.info("=" * 70)
        
        results = make_predictions(
            model=model,
            X=X_transformed,
            include_proba=True,
            top_n=args.top_n
        )
        
        # ================================================
        # ÉTAPE 5 : Sauvegarde CSV
        # ================================================
        logger.info("\n" + "=" * 70)
        logger.info("SAUVEGARDE DES RÉSULTATS")
        logger.info("=" * 70)
        
        save_predictions_to_csv(
            predictions=results['predictions'],
            confidences=results.get('confidences'),
            top_classes=results.get('top_classes'),
            original_data=X_raw,
            output_path=args.output
        )
        
        # ================================================
        # ÉTAPE 6 : Sauvegarde PostgreSQL (optionnel)
        # ================================================
        if args.save_to_db and db_config:
            logger.info("\n--- Sauvegarde dans PostgreSQL ---")
            
            model_id = model_metadata.get('model_id') or args.model_id
            if not model_id:
                logger.warning("Impossible de sauvegarder: model_id inconnu")
            else:
                loader = PostgresModelLoader(db_config)
                loader.save_predictions(
                    model_id=model_id,
                    predictions=pd.Series(results['predictions']),
                    confidences=pd.Series(results.get('confidences', [0]*len(results['predictions']))),
                    product_ids=X_raw.get('productid'),
                    metadata={
                        'timestamp': datetime.now().isoformat(),
                        'source': args.source,
                        'n_samples': results['n_samples']
                    }
                )
        
        # ================================================
        # RÉSUMÉ FINAL
        # ================================================
        logger.info("\n" + "=" * 70)
        logger.info("PRÉDICTIONS TERMINÉES")
        logger.info("=" * 70)
        logger.info(f"Échantillons: {results['n_samples']}")
        logger.info(f"Classes prédites: {len(results['class_distribution'])}")
        
        if 'mean_confidence' in results:
            logger.info(f"Confidence moyenne: {results['mean_confidence']:.4f}")
        
        logger.info(f"\nFichier de sortie: {args.output}")
        
        if args.save_to_db:
            logger.info(f"Sauvegardé en base: project.predictions")
        
        logger.info("\nDistribution des prédictions:")
        for class_id, count in sorted(results['class_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  Classe {class_id}: {count} échantillons ({count/results['n_samples']*100:.1f}%)")
        
        logger.info("=" * 70 + "\n")
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"\n Fichier non trouvé: {e}")
        return 1
    
    except ConnectionError as e:
        logger.error(f"\n Erreur de connexion PostgreSQL: {e}")
        return 1
    
    except Exception as e:
        logger.error(f"\n Erreur durant les prédictions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())










