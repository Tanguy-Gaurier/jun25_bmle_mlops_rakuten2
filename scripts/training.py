#!/usr/bin/env python3
"""
Pipeline d'entraînement PostgreSQL - Intégration production.
===========================================================

Ce script orchestre le pipeline complet avec PostgreSQL :
1. Data Ingestion (PostgreSQL)
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation
6. Registry en PostgreSQL

Différences avec train_pipeline.py :
- Charge depuis PostgreSQL au lieu de CSV
- Enregistre les splits dans project.splits
- Enregistre le modèle dans project.models
- Enregistre les prédictions dans project.predictions

Utilisation:
    # Pipeline complet avec PostgreSQL
    python scripts/training.py
    
    # Avec validation croisée
    python scripts/training.py --cv
    
    # Mode verbeux
    python scripts/training.py --verbose
    
    # Utiliser un dataset spécifique
    python scripts/training.py --dataset-hash 8297115e

    # 1. Vérifier que PostgreSQL tourne
    docker ps | grep rakuten_db

    # 2. Entraîner avec PostgreSQL
    python scripts/training.py \
  --cv \
  --evaluate-on-train \
  --model-name rakuten_prod_$(date +%Y%m%d)

"""
import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.config import load_config
from src.utils.profiling import Timer

# Import des stages existants
from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline
from src.pipeline_steps.stage04_model_training import ModelTrainingPipeline
from src.pipeline_steps.stage05_model_evaluation import ModelEvaluationPipeline

logger = logging.getLogger(__name__)

# Import PostgreSQL
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import joblib
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 non disponible - certaines fonctionnalités seront désactivées")


# =============================================================================
# Utilitaires PostgreSQL
# =============================================================================

class PostgresRegistry:
    """Gère l'enregistrement dans PostgreSQL."""
    
    def __init__(self, db_config: dict):
        """
        Initialise le registry.
        
        Args:
            db_config: Configuration de la base de données
        """
        self.db_config = db_config
    
    def get_connection(self):
        """Crée une connexion PostgreSQL."""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            raise ConnectionError(f"Échec de connexion à PostgreSQL: {e}")
    
    def get_latest_dataset_hash(self) -> Optional[str]:
        """Récupère le hash du dernier dataset."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT data_hash 
                FROM project.datasets 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None
        finally:
            conn.close()
    
    def register_split(
        self,
        dataset_hash: str,
        split_name: str,
        indices: np.ndarray,
        note: Optional[str] = None
    ) -> None:
        """
        Enregistre un split dans project.splits.
        
        Args:
            dataset_hash: Hash du dataset
            split_name: Nom du split (e.g., 'train_20241113_143000')
            indices: Array des indices (row_index)
            note: Note optionnelle
        """
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Convertir indices en array PostgreSQL
            indices_array = "{" + ",".join(map(str, indices.tolist())) + "}"
            
            cursor.execute("""
                INSERT INTO project.splits (dataset_hash, split_name, row_indices, note)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (dataset_hash, split_name) 
                DO UPDATE SET 
                    row_indices = EXCLUDED.row_indices,
                    note = EXCLUDED.note,
                    created_at = CURRENT_TIMESTAMP
            """, (dataset_hash, split_name, indices_array, note))
            
            conn.commit()
            logger.info(f" Split '{split_name}' enregistré ({len(indices)} samples)")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Échec d'enregistrement du split: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        dataset_hash: str,
        metrics: dict,
        model_path: str,
        config: dict,
        note: Optional[str] = None
    ) -> int:
        """
        Enregistre un modèle dans project.models.
        
        Returns:
            model_id: ID du modèle enregistré
        """
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO project.models 
                (model_name, model_type, dataset_hash, metrics, model_path, config, note)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING model_id
            """, (
                model_name,
                model_type,
                dataset_hash,
                json.dumps(metrics),
                model_path,
                json.dumps(config),
                note
            ))
            
            model_id = cursor.fetchone()[0]
            conn.commit()
            
            logger.info(f" Modèle '{model_name}' enregistré (ID: {model_id})")
            return model_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Échec d'enregistrement du modèle: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def register_predictions(
        self,
        model_id: int,
        predictions: pd.Series,
        confidences: Optional[pd.Series] = None,
        product_ids: Optional[pd.Series] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Enregistre des prédictions dans project.predictions.
        
        Args:
            model_id: ID du modèle
            predictions: Classes prédites
            confidences: Confidences (optionnel)
            product_ids: IDs des produits (optionnel)
            metadata: Métadonnées (optionnel)
        """
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            for i, pred_class in enumerate(predictions):
                confidence = confidences[i] if confidences is not None else None
                product_id = product_ids[i] if product_ids is not None else None
                
                cursor.execute("""
                    INSERT INTO project.predictions 
                    (model_id, product_id, predicted_class, confidence, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    model_id,
                    int(product_id) if product_id is not None else None,
                    int(pred_class),
                    float(confidence) if confidence is not None else None,
                    json.dumps(metadata or {})
                ))
            
            conn.commit()
            logger.info(f" {len(predictions)} prédictions enregistrées")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Échec d'enregistrement des prédictions: {e}")
            raise
        finally:
            cursor.close()
            conn.close()


# =============================================================================
# Stage 1 customisé pour PostgreSQL
# =============================================================================

def load_from_postgres(db_config: dict, dataset_hash: Optional[str] = None):
    """
    Charge les données depuis PostgreSQL.
    
    Args:
        db_config: Configuration de la base
        dataset_hash: Hash spécifique (optionnel)
    
    Returns:
        Tuple (X_train, y_train, X_test)
    """
    logger.info("=" * 70)
    logger.info("ÉTAPE 1 : INGESTION DES DONNÉES (PostgreSQL)")
    logger.info("=" * 70)
    
    conn = psycopg2.connect(**db_config)
    
    try:
        # Récupérer le hash du dataset si non fourni
        if dataset_hash is None:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT data_hash 
                FROM project.datasets 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            dataset_hash = result[0] if result else None
            cursor.close()
            
            if dataset_hash:
                logger.info(f"Dataset hash: {dataset_hash[:12]}...")
        
        # Charger train (avec labels)
        logger.info("\nChargement des données d'entraînement...")
        query_train = """
            SELECT 
                designation,
                description,
                productid,
                imageid,
                prdtypecode
            FROM project.items
            WHERE prdtypecode IS NOT NULL
        """
        df_train = pd.read_sql_query(query_train, conn)
        
        # On recrée un identifiant de ligne pour rester compatible avec le reste du code
        df_train = df_train.reset_index().rename(columns={"index": "row_index"})

        X_train = df_train.drop(columns=['prdtypecode'])
        y_train = df_train['prdtypecode']
        
        logger.info(f" X_train: {X_train.shape}")
        logger.info(f" y_train: {y_train.shape}")
        
        # Charger test (sans labels)
        logger.info("\nChargement des données de test...")
        query_test = """
            SELECT 
                designation,
                description,
                productid,
                imageid
            FROM project.items
            WHERE prdtypecode IS NULL
        """
        X_test = pd.read_sql_query(query_test, conn)
        
        # On fabrique un row_index technique
        X_test = X_test.reset_index().rename(columns={"index": "row_index"})

        # Si pas de test dans project.items, on peut charger depuis CSV
        if len(X_test) == 0:
            logger.warning("Aucune donnée de test dans PostgreSQL")
            # Fallback: charger depuis CSV si configuré
            from src.data.load_data import load_test_data
            from src.utils.config import load_config
            config = load_config()
            X_test = load_test_data(config.paths["x_test_csv"])

            # Harmoniser le schéma avec X_train : créer row_index technique
            X_test = X_test.reset_index().rename(columns={"index": "row_index"})
        
        logger.info(f" X_test: {X_test.shape}")
        
        logger.info("\n" + "=" * 70)
        logger.info(f" Chargement terminé: {len(X_train)} train + {len(X_test)} test")
        logger.info("=" * 70 + "\n")
        
        return X_train, y_train, X_test, dataset_hash
        
    finally:
        conn.close()


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Pipeline d'entraînement PostgreSQL Rakuten"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.toml",
        help="Chemin vers le fichier de configuration"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Active les logs détaillés (niveau DEBUG)"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Passe l'étape de validation"
    )
    
    parser.add_argument(
        "--evaluate-on-train",
        action="store_true",
        help="Évalue également sur le jeu d'entraînement"
    )
    
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Active la validation croisée"
    )
    
    parser.add_argument(
        "--dataset-hash",
        type=str,
        default=None,
        help="Hash spécifique du dataset à utiliser"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Nom personnalisé du modèle"
    )
    
    return parser.parse_args()


# =============================================================================
# Pipeline principal
# =============================================================================

def main():
    """Fonction principale du pipeline PostgreSQL."""
    args = parse_args()
    
    # Vérifier PostgreSQL
    if not POSTGRES_AVAILABLE:
        logger.error("psycopg2 n'est pas installé!")
        logger.error("Installez-le avec: pip install psycopg2-binary")
        return 1
    
    # Configuration du logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE D'ENTRAÎNEMENT RAKUTEN - MODE POSTGRESQL")
    logger.info("=" * 70)
    logger.info("Architecture: 5 stages + Registry PostgreSQL")
    logger.info("=" * 70 + "\n")
    
    # Chargement de la configuration
    try:
        config = load_config(args.config)
        logger.info(f" Configuration chargée: {args.config}")
        logger.info(f"  Modèle: {config.model['name'].upper()}")
        logger.info(f"  Random seed: {config.random_seed}")
        
        # Configuration DB
        db_config = {
            "host": config.get("database.host", "localhost"),
            "port": config.get("database.port", 5433),
            "database": config.get("database.name", "rakuten"),
            "user": config.get("database.user", "mlops"),
            "password": config.get("database.password", "mlops")
        }
        logger.info(f"  Database: {db_config['database']}@{db_config['host']}:{db_config['port']}")
        
        # Override CV si demandé
        if args.cv:
            config.cv['enabled'] = True
            logger.info(f"  CV: ACTIVÉE (--cv, {config.cv.get('splits', 3)} folds)")
        
        # Initialiser le registry
        registry = PostgresRegistry(db_config)
        
    except FileNotFoundError as e:
        logger.error(f" Fichier de configuration non trouvé: {e}")
        return 1
    except Exception as e:
        logger.error(f" Erreur lors du chargement: {e}")
        return 1
    
    try:
        with Timer("Pipeline complet"):
            
            # ================================================
            # ÉTAPE 1 : Data Ingestion (PostgreSQL)
            # ================================================
            logger.info("\n" + "★" * 35)
            logger.info("★ ÉTAPE 1/5 : INGESTION (PostgreSQL)")
            logger.info("★" * 35)
            
            X_train, y_train, X_test, dataset_hash = load_from_postgres(
                db_config=db_config,
                dataset_hash=args.dataset_hash
            )
            
            # ================================================
            # ÉTAPE 2 : Data Validation
            # ================================================
            if not args.skip_validation:
                logger.info("\n" + "★" * 35)
                logger.info("★ ÉTAPE 2/5 : VALIDATION")
                logger.info("★" * 35)
                
                stage2 = DataValidationPipeline(config)
                validation_ok = stage2.run(X_train, y_train, X_test)
                
                if not validation_ok:
                    logger.error("\n Validation échouée - Arrêt du pipeline")
                    return 1
            else:
                logger.warning("\n⚠ Validation ignorée (--skip-validation)")
            
            # ================================================
            # ÉTAPE 3 : Data Transformation
            # ================================================
            logger.info("\n" + "★" * 35)
            logger.info("★ ÉTAPE 3/5 : TRANSFORMATION")
            logger.info("★" * 35)
            
            stage3 = DataTransformationPipeline(config)
            X_train_t, y_train_t, X_test_t, feature_pipeline, feature_mapping = stage3.run(
                X_train, y_train, X_test
            )
            
            # Enregistrer les splits dans PostgreSQL
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            try:
                registry.register_split(
                    dataset_hash=dataset_hash,
                    split_name=f"train_{timestamp}",
                    indices=X_train['row_index'].values,
                    note=f"Train split - {len(X_train)} samples"
                )
            except Exception as e:
                logger.warning(f"Impossible d'enregistrer le split: {e}")
            
            # ================================================
            # ÉTAPE 4 : Model Training
            # ================================================
            logger.info("\n" + "★" * 35)
            logger.info("★ ÉTAPE 4/5 : TRAINING")
            logger.info("★" * 35)
            
            stage4 = ModelTrainingPipeline(config)
            model = stage4.run(X_train_t, y_train_t, feature_pipeline)
            
            # ================================================
            # ÉTAPE 5 : Model Evaluation
            # ================================================
            logger.info("\n" + "★" * 35)
            logger.info("★ ÉTAPE 5/5 : EVALUATION")
            logger.info("★" * 35)
            
            stage5 = ModelEvaluationPipeline(config)
            
            # Évaluation sur train (optionnel)
            train_results = None
            if args.evaluate_on_train:
                logger.info(f"\nÉvaluation sur train ({len(X_train_t)} samples)...")
                train_results = stage5.run(
                    model, X_train_t, y_train_t,
                    dataset_name="train",
                    trainer=stage4.trainer,
                    feature_mapping=feature_mapping,
                    feature_pipeline=feature_pipeline
                )
            
            # Prédictions sur test
            logger.info(f"\nGénération des prédictions test ({len(X_test_t)} samples)...")
            test_results = stage5.run(
                model, X_test_t, y_true=None,
                dataset_name="test",
                trainer=stage4.trainer,
                feature_mapping=feature_mapping,
                feature_pipeline=feature_pipeline
            )
            
            # Sauvegarder les prédictions test
            pred_output = config.paths.get("pred_out", "results/predictions/test_predictions.csv")
            pred_output = pred_output.replace("{kind}", config.model["name"])
            pred_output = pred_output.replace("{phase}", "final")
            
            predictions_df = pd.DataFrame({
                "prediction": test_results["predictions"]
            }, index=X_test.index)
            
            pred_output_path = Path(pred_output)
            pred_output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(pred_output_path)
            
            logger.info(f" Prédictions test sauvegardées: {pred_output_path}")
            
            # ================================================
            # Enregistrement PostgreSQL
            # ================================================
            logger.info("\n" + "★" * 35)
            logger.info("★ ENREGISTREMENT POSTGRESQL")
            logger.info("★" * 35)
            
            # Préparer les métriques
            metrics = {}
            if train_results:
                metrics = {
                    "accuracy": float(train_results["accuracy"]),
                    "f1_weighted": float(train_results["f1_weighted"]),
                    "f1_macro": float(train_results["f1_macro"])
                }
            
            # Nom du modèle
            model_name = args.model_name or f"rakuten_{config.model['name']}_{timestamp}"

            # Chemin du modèle : on utilise directement ceux de la pipeline d'entraînement
            if hasattr(stage4, "model_path") and stage4.model_path is not None:
                model_path = stage4.model_path
            else:
                # fallback très simple si jamais
                from pathlib import Path
                model_dir = Path(config.paths["model_dir"])
                model_path = model_dir / "model.joblib"
            
            # Enregistrer le modèle
            try:
                model_id = registry.register_model(
                    model_name=model_name,
                    model_type=config.model['name'].upper(),
                    dataset_hash=dataset_hash,
                    metrics=metrics,
                    model_path=str(model_path),
                    config=dict(config.model),
                    note=f"Trained with {len(X_train_t)} samples"
                )
                
                logger.info(f" Modèle enregistré (ID: {model_id})")
                
                # Enregistrer les prédictions test
                if 'productid' in X_test.columns:
                    registry.register_predictions(
                        model_id=model_id,
                        predictions=pd.Series(test_results["predictions"]),
                        product_ids=X_test['productid'],
                        metadata={"dataset": "test", "timestamp": timestamp}
                    )
                
            except Exception as e:
                logger.error(f" Échec d'enregistrement: {e}")
        
        # ================================================
        # Résumé final
        # ================================================
        logger.info("\n" + "=" * 70)
        logger.info(" PIPELINE TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 70)
        logger.info(f"\nRésultats:")
        logger.info(f"  • Modèle: {model_name}")
        logger.info(f"  • Type: {config.model['name'].upper()}")
        logger.info(f"  • Dataset hash: {dataset_hash[:12]}...")
        logger.info(f"  • Données train: {X_train_t.shape}")
        logger.info(f"  • Données test: {X_test_t.shape}")
        
        if train_results:
            logger.info(f"\nPerformance sur train:")
            logger.info(f"  • Accuracy: {train_results['accuracy']:.4f}")
            logger.info(f"  • F1 weighted: {train_results['f1_weighted']:.4f}")
            logger.info(f"  • F1 macro: {train_results['f1_macro']:.4f}")
        
        logger.info(f"\nFichiers générés:")
        logger.info(f"  • Modèle: {model_path}")
        logger.info(f"  • Prédictions: {pred_output_path}")
        
        logger.info(f"\nPostgreSQL:")
        logger.info(f"  • Model ID: {model_id}")
        logger.info(f"  • Database: {db_config['database']}")
        
        logger.info("\n" + "=" * 70 + "\n")
        
        return 0
    
    except ConnectionError as e:
        logger.error(f"\n Erreur de connexion PostgreSQL: {e}")
        logger.error("Vérifiez que PostgreSQL est démarré et accessible")
        return 1
    
    except FileNotFoundError as e:
        logger.error(f"\n Fichier non trouvé: {e}")
        return 1
    
    except Exception as e:
        logger.error(f"\n Erreur durant le pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
