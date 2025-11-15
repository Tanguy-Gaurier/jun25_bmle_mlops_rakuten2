"""
Etape 5 : Evaluation du modele (Model Evaluation).
===================================================

Cette etape evalue les performances avec :
- Matrices de confusion en 5 formats (nombres, pourcentages, hybride)
- Rapports de classification detailles
- SHAP decompose par composante (optionnel)
- Metriques sauvegardees en JSON

Utilisation:
    from src.pipeline_steps.stage05_model_evaluation import ModelEvaluationPipeline
    
    pipeline = ModelEvaluationPipeline(config)
    results = pipeline.run(
        model, X_test, y_test,
        dataset_name="test",
        trainer=trainer,
        feature_mapping=feature_mapping  # Pour SHAP decompose
    )

"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from scipy import sparse

from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


# ========================================
# Fonctions helper
# ========================================

def _load_labels_map(path: str | Path) -> Dict[int, str]:
    """Charge le mapping ID -> Nom de classe depuis JSON."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"labels_map.json introuvable a {path} - noms de classes indisponibles.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Clefs peuvent etre str dans le JSON -> cast en int
    return {int(k): v for k, v in raw.items()}


def _decode_to_original_ids(y: np.ndarray, trainer: Any) -> np.ndarray:
    """
    Convertit les labels vers les IDs originaux.
    
    Detecte automatiquement si y est encode (0..n-1) ou deja original (10, 40, 1140...).
    
    Args:
        y: Labels (encodes ou originaux)
        trainer: Trainer avec label_encoder
        
    Returns:
        Labels avec IDs originaux
    """
    if trainer is None or not hasattr(trainer, "label_encoder"):
        # Pas de trainer : on suppose que y est deja original
        return y
    
    # Heuristique : si max(y) < nombre de classes, alors y est encode
    n_classes = len(trainer.label_encoder.classes_)
    
    if y.max() < n_classes:
        # y est encode (0..n-1) -> decoder
        return trainer.label_encoder.inverse_transform(y)
    else:
        # y contient deja les IDs originaux
        return y


def _build_label_names(ids: List[int], labels_map: Dict[int, str]) -> List[str]:
    """
    Construit une liste de noms lisibles: 'id - nom' (ou 'id' si nom absent).
    """
    out = []
    for cid in ids:
        name = labels_map.get(cid)
        out.append(f"{cid} - {name}" if name else str(cid))
    return out


def _get_feature_names_from_trainer(trainer: Any, n_features: int) -> List[str]:
    """
    Recupere des noms de features si possible.
    """
    try:
        if hasattr(trainer, "feature_pipeline") and hasattr(
            trainer.feature_pipeline, "get_feature_names_out"
        ):
            names = list(trainer.feature_pipeline.get_feature_names_out())
            if len(names) == n_features:
                return names
    except Exception as e:
        logger.debug(f"Impossible de recuperer les noms de features depuis le pipeline: {e}")

    return [f"f{i}" for i in range(n_features)]


# ========================================
# Pipeline principal
# ========================================

class ModelEvaluationPipeline:
    """
    Evalue les performances, genere rapports et matrices de confusion.
    Gere un export SHAP optionnel avec decomposition par composante.
    """

    def __init__(self, config):
        self.config = config
        self.results: Dict[str, Any] = {}

        logger.info("=" * 70)
        logger.info("ETAPE 5 : EVALUATION DU MODELE")
        logger.info("=" * 70)

        # Chemins et options
        self.results_dir = Path("results/metrics")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = self.results_dir
        self.model_name = config.model.get("name", "model")

        # Charger le labels_map
        labels_map_path = Path("config/labels_map.json")
        self.labels_map = _load_labels_map(labels_map_path)

        # Config SHAP
        self.shap_enabled = config.get("explainability.enable_shap", False)
        self.shap_decomposed_enabled = config.get("explainability.enable_shap_decomposed", False)
        self.shap_samples = config.get("explainability.shap_samples", 10000)
        self.shap_decomposed_samples = config.get("explainability.shap_decomposed_samples", 1000)

    def predict(
        self,
        model: Any,
        X: np.ndarray,
        trainer: Any
    ) -> np.ndarray:
        """
        Effectue les predictions et decode vers IDs originaux.
        
        Args:
            model: Modele entraine
            X: Features
            trainer: Trainer avec label_encoder
            
        Returns:
            Predictions avec IDs originaux
        """
        n_samples = X.shape[0]
        
        logger.info(f"\n--- Predictions sur {n_samples} echantillons ---")
        
        with Timer(f"Prediction ({n_samples} echantillons)"):
            y_pred_encoded = model.predict(X)
        
        # Decoder vers IDs originaux
        if trainer is not None and hasattr(trainer, "label_encoder"):
            y_pred_original = trainer.label_encoder.inverse_transform(y_pred_encoded)
            logger.info("Predictions decodees vers classes originales")
        else:
            y_pred_original = y_pred_encoded
            logger.warning("Pas de decoder disponible - predictions brutes")
        
        return y_pred_original

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str
    ) -> Dict[str, float]:
        """
        Calcule les metriques de classification.
        
        Args:
            y_true: Vrais labels (IDs originaux)
            y_pred: Predictions (IDs originaux)
            dataset_name: Nom du dataset
            
        Returns:
            Dict avec metriques
        """
        logger.info(f"\n--- Calcul des metriques ({dataset_name}) ---")
        
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average='weighted')),
            "f1_macro": float(f1_score(y_true, y_pred, average='macro')),
            "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
        
        return metrics

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[str, List[int], List[str]]:
        """
        Genere un rapport de classification detaille.
        
        Args:
            y_true: Vrais labels (IDs originaux)
            y_pred: Predictions (IDs originaux)
            
        Returns:
            Tuple (rapport_texte, ordered_ids, target_names)
        """
        logger.info("\n--- Rapport de classification ---")
        
        # IDs uniques tries
        ordered_ids = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        # Noms correspondants
        target_names = _build_label_names(ordered_ids, self.labels_map)
        
        # Rapport sklearn
        report_text = classification_report(
            y_true,
            y_pred,
            labels=ordered_ids,
            target_names=target_names,
            digits=2,
            zero_division=0
        )
        
        # Afficher apercu
        logger.info("Apercu du rapport:")
        for line in report_text.split('\n')[:11]:  # Premieres 11 lignes
            logger.info(line)
        
        return report_text, ordered_ids, target_names

    def save_confusion_matrices(
        self,
        cm: np.ndarray,
        ordered_ids: List[int],
        target_names: List[str],
        dataset_name: str
    ) -> None:
        """
        Sauvegarde les matrices de confusion en 5 formats.
        
        1. Nombres avec IDs
        2. Nombres avec noms
        3. Pourcentages avec IDs
        4. Pourcentages avec noms
        5. Hybride : "nombre (pourcentage%)"
        
        Args:
            cm: Matrice de confusion (numpy array)
            ordered_ids: Liste des IDs de classes
            target_names: Liste des noms de classes
            dataset_name: Nom du dataset
        """
        logger.info("\n--- Matrice de confusion ---")
        logger.info(f"Dimensions: {cm.shape}")
        logger.info(f"Diagonal (correctes): {cm.diagonal().sum()}")
        logger.info(f"Total: {cm.sum()}")
        logger.info(f"Accuracy (depuis matrice): {cm.diagonal().sum() / cm.sum():.4f}")
        
        logger.info("\n--- Sauvegarde des resultats ---")
        
        # ========================================
        # 1 & 2. Matrices avec nombres 
        # ========================================
        df_ids = pd.DataFrame(cm, index=ordered_ids, columns=ordered_ids)
        path_ids = self.output_dir / f"{self.model_name}_confusion_matrix_ids.csv"
        df_ids.to_csv(path_ids)
        logger.info(f"[OK] Matrice (IDs): {path_ids}")
        
        df_named = pd.DataFrame(cm, index=target_names, columns=target_names)
        path_named = self.output_dir / f"{self.model_name}_confusion_matrix_named.csv"
        df_named.to_csv(path_named)
        logger.info(f"[OK] Matrice (nominative): {path_named}")
        
        # ========================================
        # 3 & 4. Matrices avec pourcentages 
        # ========================================
        # Normaliser par ligne (% de chaque classe predite)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_percent = np.nan_to_num(cm_percent)  # Remplacer NaN par 0
        
        # Avec IDs
        df_percent_ids = pd.DataFrame(
            cm_percent,
            index=ordered_ids,
            columns=ordered_ids
        )
        path_percent_ids = self.output_dir / f"{self.model_name}_confusion_matrix_percent_ids.csv"
        df_percent_ids.to_csv(path_percent_ids, float_format='%.2f')
        logger.info(f"[OK] Matrice (pourcentages IDs): {path_percent_ids}")
        
        # Avec noms
        df_percent_named = pd.DataFrame(
            cm_percent,
            index=target_names,
            columns=target_names
        )
        path_percent_named = self.output_dir / f"{self.model_name}_confusion_matrix_percent_named.csv"
        df_percent_named.to_csv(path_percent_named, float_format='%.2f')
        logger.info(f"[OK] Matrice (pourcentages noms): {path_percent_named}")
        
        # ========================================
        # 5. Matrice hybride 
        # ========================================
        cm_hybrid = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percent = cm_percent[i, j]
                
                # Format : "nombre (pourcentage%)"
                if count == 0:
                    cm_hybrid[i, j] = "0 (0.0%)"
                else:
                    cm_hybrid[i, j] = f"{count} ({percent:.1f}%)"
        
        df_hybrid = pd.DataFrame(
            cm_hybrid,
            index=target_names,
            columns=target_names
        )
        path_hybrid = self.output_dir / f"{self.model_name}_confusion_matrix_hybrid.csv"
        df_hybrid.to_csv(path_hybrid)
        logger.info(f"[OK] Matrice (hybride): {path_hybrid}")

    def run_shap_decomposed(
        self,
        model: Any,
        X: np.ndarray,
        feature_mapping: Dict[str, Tuple[int, int]],
        trainer: Any
    ) -> Dict[str, Any]:
        """
        Analyse SHAP decomposee par composante de features.
        
        Necessite :
        - pip install shap
        - feature_mapping depuis stage03
        
        Args:
            model: Modele entraine
            X: Features (matrice complete)
            feature_mapping: Mapping des ranges {'text': (0, 15000), ...}
            trainer: Trainer pour decoder les classes
            
        Returns:
            Dict avec SHAP values par composante
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP n'est pas installe. Installez avec: pip install shap")
            return {}
        
        logger.info("\n" + "=" * 70)
        logger.info("ANALYSE SHAP DECOMPOSEE PAR COMPOSANTE")
        logger.info("=" * 70)
        
        max_samples = min(self.shap_decomposed_samples, X.shape[0])
        logger.info(f"Echantillon SHAP: {max_samples} lignes")
        logger.info(f"Composantes detectees: {list(feature_mapping.keys())}")
        
        # Echantillonner si necessaire
        if X.shape[0] > max_samples:
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Creer l'explainer
        logger.info("\nCreation de l'explainer SHAP...")
        explainer = shap.TreeExplainer(model)
        
        logger.info("Calcul des SHAP values...")
        with Timer("SHAP values"):
            shap_values = explainer.shap_values(X_sample)
        
        # Si multi-classe, shap_values est une liste
        if isinstance(shap_values, list):
            n_classes = len(shap_values)
        else:
            n_classes = 1
            shap_values = [shap_values]
        
        logger.info(f"SHAP values calculees pour {n_classes} classes")
        
        # Decomposer par composante
        results = {
            'components': {},
            'global_importance': {},
            'per_class_importance': {}
        }
        
        logger.info("\n--- Decomposition par composante ---")
        
        for component_name, (start, end) in feature_mapping.items():
            logger.info(f"\n Analyse de {component_name} (colonnes {start}-{end})")
            
            # Extraire les SHAP values pour cette composante
            component_shap_values = []
            
            for class_idx in range(n_classes):
                # SHAP values pour cette classe et cette composante
                shap_class = shap_values[class_idx][:, start:end]
                component_shap_values.append(shap_class)
                
                # Importance moyenne pour cette classe
                importance = np.abs(shap_class).mean()
                
                # Decoder l'ID de classe
                class_id = int(trainer.label_encoder.classes_[class_idx])
                
                if class_id not in results['per_class_importance']:
                    results['per_class_importance'][class_id] = {}
                
                results['per_class_importance'][class_id][component_name] = float(importance)
            
            # Importance globale (moyenne sur toutes les classes)
            global_importance = np.mean([
                np.abs(shap_class).mean() 
                for shap_class in component_shap_values
            ])
            
            results['global_importance'][component_name] = float(global_importance)
            results['components'][component_name] = {
                'start': int(start),
                'end': int(end),
                'n_features': int(end - start)
            }
            
            logger.info(f"  Importance globale: {global_importance:.6f}")
        
        # Sauvegarder les resultats JSON
        output_json = self.output_dir / "shap_decomposed.json"
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n[OK] Resultats SHAP decompose: {output_json}")
        
        # Sauvegarder un resume texte
        output_txt = self.output_dir / "shap_summary.txt"
        with open(output_txt, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("IMPORTANCE GLOBALE PAR COMPOSANTE\n")
            f.write("=" * 70 + "\n\n")
            
            for comp, imp in sorted(
                results['global_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                f.write(f"{comp:25s} : {imp:.6f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("TOP 3 COMPOSANTES PAR CLASSE\n")
            f.write("=" * 70 + "\n\n")
            
            for class_id in sorted(results['per_class_importance'].keys()):
                class_name = self.labels_map.get(class_id, str(class_id))
                f.write(f"\nClasse {class_id} - {class_name}:\n")
                
                # Trier par importance
                class_importances = results['per_class_importance'][class_id]
                top3 = sorted(class_importances.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for comp, val in top3:
                    f.write(f"  {comp:25s}: {val:.6f}\n")
        
        logger.info(f"[OK] Resume texte: {output_txt}")
        
        # Afficher un resume dans les logs
        logger.info("\n" + "=" * 70)
        logger.info("IMPORTANCE GLOBALE PAR COMPOSANTE")
        logger.info("=" * 70)
        for comp, imp in sorted(
            results['global_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            logger.info(f"  {comp:25s} : {imp:.6f}")
        logger.info("=" * 70 + "\n")
        
        return results

    def maybe_run_shap(
        self,
        model: Any,
        X: np.ndarray,
        trainer: Any = None
    ) -> None:
        """
        Lance SHAP standard si active dans la config.
        
        Args:
            model: Modele entraine
            X: Features
            trainer: Trainer (optionnel)
        """
        if not self.shap_enabled:
            logger.info("SHAP desactive dans la configuration.")
            return
        
        try:
            import shap
        except ImportError:
            logger.error("SHAP n'est pas installe. Installez avec: pip install shap")
            return
        
        logger.info("\n--- Analyse SHAP standard ---")
        
        max_samples = min(self.shap_samples, X.shape[0])
        logger.info(f"Echantillon SHAP: {max_samples} lignes")
        
        # Echantillonner
        if X.shape[0] > max_samples:
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Creer explainer
        explainer = shap.TreeExplainer(model)
        
        with Timer("SHAP values (standard)"):
            shap_values = explainer.shap_values(X_sample)
        
        logger.info("[OK] SHAP values calcules")
        logger.info("Pour visualiser, utilisez shap.summary_plot() dans un notebook")

    def run(
        self,
        model: Any,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        dataset_name: str = "dataset",
        trainer: Any = None,
        feature_mapping: Optional[Dict[str, Tuple[int, int]]] = None,
        feature_pipeline: Any = None, 
    ) -> Dict[str, Any]:
        """
        Execute le pipeline d'evaluation complet.
        
        Args:
            model: Modele entraine
            X: Features
            y_true: Vrais labels (optionnel pour test)
            dataset_name: Nom du dataset
            trainer: Trainer avec label_encoder
            feature_mapping: Mapping des features pour SHAP decompose
            
        Returns:
            Dict avec resultats
        """
        with Timer("Evaluation du modele"):
            
            # ========================================
            # 1. Predictions
            # ========================================
            y_pred_any = self.predict(model, X, trainer)
            
            self.results["predictions"] = y_pred_any
            self.results["dataset_name"] = dataset_name
            
            # ========================================
            # 2. Metriques (si y_true fourni)
            # ========================================
            if y_true is None:
                logger.info("\nPas de labels fournis - evaluation limitee aux predictions")
                
                # SHAP eventuellement
                if self.shap_enabled:
                    self.maybe_run_shap(model, X, trainer)
                
                # SHAP decompose
                if self.shap_decomposed_enabled and feature_mapping is not None:
                    self.run_shap_decomposed(model, X, feature_mapping, trainer)
                
                return self.results
            
            # Decoder y_true si necessaire
            if trainer is None:
                raise ValueError(
                    "Trainer requis pour decoder y_true/y_pred vers IDs originaux."
                )
            
            y_true_orig = _decode_to_original_ids(y_true, trainer)
            y_pred_orig = y_pred_any  # Deja decode dans predict()
            
            # ========================================
            # 3. Calcul des metriques
            # ========================================
            metrics = self.calculate_metrics(y_true_orig, y_pred_orig, dataset_name)
            
            # ========================================
            # 4. Rapport de classification
            # ========================================
            report_text, ordered_ids, target_names = self.generate_classification_report(
                y_true_orig, y_pred_orig
            )
            
            # ========================================
            # 5. Matrice de confusion
            # ========================================
            cm = confusion_matrix(y_true_orig, y_pred_orig, labels=ordered_ids)
            self.save_confusion_matrices(cm, ordered_ids, target_names, dataset_name)
            
            # ========================================
            # 6. Sauvegarder JSON et rapport
            # ========================================
            # Metriques JSON
            metrics_path = self.output_dir / f"{self.model_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"[OK] Metriques: {metrics_path}")
            
            # Rapport texte
            report_path = self.output_dir / f"{self.model_name}_classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(report_text)
            logger.info(f"[OK] Rapport: {report_path}")
            
            # ========================================
            # 7. SHAP standard
            # ========================================
            if self.shap_enabled:
                self.maybe_run_shap(model, X, trainer)
            
            # ========================================
            # 8. SHAP decompose (NOUVEAU)
            # ========================================
            if self.shap_decomposed_enabled and feature_mapping is not None:
                logger.info("\n[INFO] Lancement du SHAP decompose...")
                shap_results = self.run_shap_decomposed(model, X, feature_mapping, trainer)
                self.results['shap_decomposed'] = shap_results
            elif self.shap_decomposed_enabled and feature_mapping is None:
                logger.warning("[WARNING] SHAP decompose active mais feature_mapping non fourni")
            
            # ========================================
            # 9. Resume final
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info("RESUME DE L'EVALUATION")
            logger.info("=" * 70)
            logger.info(f"[OK] Dataset: {dataset_name}")
            logger.info(f"[OK] Predictions: {len(y_pred_orig)}")
            logger.info(f"[OK] Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"[OK] F1 (weighted): {metrics['f1_weighted']:.4f}")
            logger.info(f"[OK] F1 (macro): {metrics['f1_macro']:.4f}")
            logger.info(f"[OK] Resultats sauvegardes dans: {self.output_dir}")
            logger.info("=" * 70 + "\n")
            
            # Stocker les resultats
            self.results.update(metrics)
            self.results['confusion_matrix'] = cm.tolist()
            self.results['classification_report'] = report_text
            
            return self.results


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.config import load_config
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
    from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline
    from src.pipeline_steps.stage04_model_training import ModelTrainingPipeline
    
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Test de ModelEvaluationPipeline")
    print("="*70 + "\n")
    
    try:
        # Charger la configuration
        config = load_config()
        
        # Stages 1-4
        stage1 = DataIngestionPipeline(config)
        X_train, y_train, X_test = stage1.run()
        
        stage2 = DataValidationPipeline(config)
        validation_ok = stage2.run(X_train, y_train, X_test)
        
        if not validation_ok:
            print("\n[ERROR] Validation echouee")
        else:
            stage3 = DataTransformationPipeline(config)
            X_train_t, y_train_t, X_test_t, pipeline, feature_mapping = stage3.run(
                X_train, y_train, X_test
            )
            
            stage4 = ModelTrainingPipeline(config)
            model = stage4.run(X_train_t, y_train_t, pipeline)
            
            # Stage 5: Evaluation avec feature_mapping
            stage5 = ModelEvaluationPipeline(config)
            
            results = stage5.run(
                model, X_train_t, y_train_t,
                dataset_name="train",
                trainer=stage4.trainer,
                feature_mapping=feature_mapping
            )
            
            print("\n[OK] Evaluation terminee avec succes!")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  F1 (weighted): {results['f1_weighted']:.4f}")
            print(f"  Resultats dans: results/metrics/")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Erreur: {e}")
    except Exception as e:
        print(f"\n[ERROR] Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()




























