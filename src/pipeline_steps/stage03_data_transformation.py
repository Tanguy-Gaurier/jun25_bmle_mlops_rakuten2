"""
Etape 3 : Transformation des donnees (Data Transformation).
==========================================================

Cette etape applique le reechantillonnage et construit les features.

Responsabilites :
- Reechantillonnage (under/over sampling)
- Construction des pipelines de features (texte + image + CNN)
- Fusion ponderee des branches
- Transformation des donnees train et test
- Creation du mapping des features pour SHAP decompose

Utilisation:
    from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline
    
    pipeline = DataTransformationPipeline(config)
    X_train_t, y_train_t, X_test_t, feature_pipeline, feature_mapping = pipeline.run(
        X_train, y_train, X_test
    )

"""
from __future__ import annotations

import logging
from typing import Tuple, Any, Dict

import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion

from src.data.sampling import apply_sampling
from src.pipelines.text_pipeline import create_text_pipeline_from_cfg
from src.pipelines.image_pipeline import create_image_pipeline_from_cfg
from src.features.cnn_features import CNNFeaturizer
from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


class DataTransformationPipeline:
    """
    Pipeline de transformation des donnees.
    
    Applique le reechantillonnage et construit les features texte/image/CNN.
    
    Attributes:
        config: Configuration complete du projet
        feature_pipeline: Pipeline sklearn de features (apres fit)
        feature_mapping: Mapping des ranges de colonnes par composante
        
    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> pipeline = DataTransformationPipeline(config)
        >>> X_train_t, y_train_t, X_test_t, pipe, mapping = pipeline.run(
        ...     X_train, y_train, X_test
        ... )
    """
    
    def __init__(self, config):
        """
        Initialise le pipeline de transformation.
        
        Args:
            config: Objet Config contenant tous les parametres
        """
        self.config = config
        self.feature_pipeline = None
        self.feature_mapping = {}
        self._original_X_train = None  # Pour creer le mapping
        
        # ========================================
        # Système de cache
        # ========================================
        from pathlib import Path
        self.cache_dir = Path("artifacts/cache/features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = config.get("compute.use_cache", True)
        
        logger.info("=" * 70)
        logger.info("ETAPE 3 : TRANSFORMATION DES DONNEES")
        logger.info("=" * 70)
        if self.use_cache:
            logger.info(f" Cache activé: {self.cache_dir}")

    
    def apply_resampling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applique le reechantillonnage pour equilibrer les classes.
        
        Args:
            X_train: Features d'entrainement
            y_train: Labels d'entrainement
            
        Returns:
            Tuple (X_train_resampled, y_train_resampled)
        """
        logger.info("\n--- Reechantillonnage ---")
        
        # Parametres depuis la config
        sampling_config = self.config.sampling
        
        major_class = sampling_config.get("major_class", 2583)
        major_cap = sampling_config.get("major_cap", 2500)
        tail_min = sampling_config.get("tail_min", 1500)
        random_state = self.config.random_seed
        
        logger.info(f"Parametres:")
        logger.info(f"  - Classe majoritaire: {major_class}")
        logger.info(f"  - Cap majorite: {major_cap}")
        logger.info(f"  - Min minorite: {tail_min}")
        
        # Appliquer le sampling
        X_resampled, y_resampled = apply_sampling(
            X=X_train,
            y=y_train,
            major_class=major_class,
            major_cap=major_cap,
            tail_min=tail_min,
            random_state=random_state
        )
        
        logger.info(f"[OK] Reechantillonnage termine: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def build_feature_pipeline(self) -> FeatureUnion:
        """
        Construit le pipeline de features (texte + image + CNN).
        
        Returns:
            FeatureUnion sklearn combinant toutes les branches activees
        """
        logger.info("\n--- Construction du pipeline de features ---")
        
        transformers = []
        weights = {}
        
        # ========================================
        # Branche Texte
        # ========================================
        logger.info("Construction de la branche texte...")
        
        # Passer la config complète avec SVD
        # Au lieu de self.config.text, on passe self.config["features"]["text"]
        # ou on construit un dict avec toutes les sections nécessaires
        text_config = {}
        
        # Récupérer la config text complète
        if hasattr(self.config, 'text'):
            text_config = dict(self.config.text) if hasattr(self.config.text, '__iter__') else {}
        else:
            # Fallback: construire depuis features.text
            text_config = self.config.get("features.text", {})
        
        # CRITIQUE: S'assurer que la section SVD est incluse
        if "svd" not in text_config:
            svd_config = self.config.get("features.text.svd", {})
            if svd_config:
                text_config["svd"] = svd_config
                logger.info(f"   Section SVD ajoutée manuellement: {svd_config}")
        
        logger.info(f"   Config texte complète: {list(text_config.keys())}")
        if "svd" in text_config:
            logger.info(f"   Config SVD présente: {text_config['svd']}")
        
        text_pipeline = create_text_pipeline_from_cfg(text_config)
        transformers.append(("text", text_pipeline))
        
        # Poids de la branche texte
        text_weight = self.config.get("fusion.weights.text", 1.0)
        weights["text"] = float(text_weight)
        logger.info(f"  Poids texte: {text_weight}")
        
        # ========================================
        # Branche Image (pixels, stats, etc.)
        # ========================================
        # Verifier si les images traditionnelles sont activees
        image_pixels_enabled = self.config.get("features.image.pixels.enabled", False)
        image_stats_enabled = self.config.get("features.image.stats.enabled", False)
        
        if image_pixels_enabled or image_stats_enabled:
            logger.info("Construction de la branche image (pixels/stats)...")
            image_pipeline = create_image_pipeline_from_cfg(
                self.config.images,
                use_test_dir=False  # Train d'abord
            )
            transformers.append(("image_pixels", image_pipeline))
            
            # Poids
            image_pixels_weight = self.config.get("fusion.weights.image_pixels", 0.5)
            weights["image_pixels"] = float(image_pixels_weight)
            logger.info(f"  Poids image_pixels: {image_pixels_weight}")
        
        # ========================================
        # Branche CNN ResNet/ViT
        # ========================================
        cnn_enabled = self.config.get("features.image.cnn.enabled", False)
        
        if cnn_enabled:
            logger.info("Construction de la branche CNN...")
            
            # Importer CNNFeaturizer
            try:
                from src.features.cnn_features import CNNFeaturizer
            except ImportError as e:
                logger.error(f"Impossible d'importer CNNFeaturizer: {e}")
                logger.warning("CNN desactive - continuons sans CNN")
                cnn_enabled = False
            
            if cnn_enabled:
                # Parametres CNN depuis config
                cnn_config = self.config.get("features.image.cnn", {})
                
                # Determiner le chemin des images
                image_dir = self.config.get("paths.image_train_dir", 
                                           self.config.get("images.train_dir", 
                                                          "data/images/images/image_train"))
                
                logger.info(f"  Architecture: {cnn_config.get('arch', 'resnet50')}")
                logger.info(f"  Batch size: {cnn_config.get('batch_size', 16)}")
                logger.info(f"  Device: {cnn_config.get('device', 'auto')}")
                logger.info(f"  Image dir: {image_dir}")
                
                # Creer le CNNFeaturizer (SANS les paramètres SVD)
                cnn_transformer = CNNFeaturizer(
                    image_dir=image_dir,
                    arch=cnn_config.get("arch", "resnet50"),
                    batch_size=cnn_config.get("batch_size", 16),
                    device=cnn_config.get("device", "auto"),
                    use_imagenet_norm=cnn_config.get("use_imagenet_norm", True),
                    fallback_zero=cnn_config.get("fallback_zero", True),
                    dtype=cnn_config.get("dtype", "float32"),
                    num_workers=cnn_config.get("num_workers", 0),
                    
                    # Fine-tuning
                    finetune_epochs=cnn_config.get("finetune_epochs", 0),
                    finetune_lr=cnn_config.get("finetune_lr", 0.0003),
                    finetune_weight_decay=cnn_config.get("finetune_weight_decay", 0.01),
                    finetune_max_n=cnn_config.get("finetune_max_n", 8000),
                    trainable_last_n=cnn_config.get("trainable_last_n", 0),
                    ft_patience=cnn_config.get("ft_patience", 3),
                    label_smoothing=cnn_config.get("label_smoothing", 0.0),
                    
                    # Augmentation
                    aug_hflip_p=cnn_config.get("aug_hflip_p", 0.2),
                    aug_color_jitter=cnn_config.get("aug_color_jitter", 0.0),
                    random_resized_crop_scale=cnn_config.get("random_resized_crop_scale", [0.9, 1.0]),
                    random_resized_crop_ratio=cnn_config.get("random_resized_crop_ratio", [0.95, 1.05]),
                    mixup_alpha=cnn_config.get("mixup_alpha", 0.0),
                    cutmix_alpha=cnn_config.get("cutmix_alpha", 0.0),
                    
                    # Grad-CAM
                    save_head_path=cnn_config.get("save_head_path", "artifacts/head_ft.pt"),
                    save_head_normalize=cnn_config.get("save_head_normalize", True),
                    
                    # Optimisation
                    foreach=cnn_config.get("foreach", True),
                )
                
                # ========================================
                # WRAPPER SVD : Appliquer SVD APRÈS le CNN si activé
                # ========================================
                svd_config = cnn_config.get("svd", {})
                svd_enabled = svd_config.get("enabled", False)
                
                logger.info(f"   CNN config SVD: {svd_config}")
                
                if svd_enabled:
                    from sklearn.pipeline import Pipeline as SkPipeline
                    from sklearn.decomposition import TruncatedSVD
                    from sklearn.preprocessing import Normalizer
                    
                    n_components = int(svd_config.get("n_components", 500))
                    random_state = int(svd_config.get("random_state", 42))
                    
                    logger.info(f"   SVD CNN activée: n_components={n_components}")
                    logger.info(f"   Création du wrapper Pipeline: CNN → SVD → L2 norm")
                    
                    # Wrapper avec SVD
                    cnn_pipeline = SkPipeline([
                        ("cnn", cnn_transformer),
                        ("svd", TruncatedSVD(n_components=n_components, random_state=random_state)),
                        ("l2norm", Normalizer(copy=False))
                    ])
                    
                    transformers.append(("image_cnn", cnn_pipeline))
                    logger.info(f"   SVD post-CNN: {n_components} composantes")
                else:
                    # Pas de SVD, utiliser le CNN directement
                    transformers.append(("image_cnn", cnn_transformer))
                    logger.info(f"   SVD post-CNN: désactivée")
                
                # Poids
                cnn_weight = self.config.get("fusion.weights.image_cnn", 1.0)
                weights["image_cnn"] = float(cnn_weight)
                logger.info(f"  Poids image_cnn: {cnn_weight}")
        
        # ========================================
        # Branche ViT (optionnel)
        # ========================================
        vit_enabled = self.config.get("features.image.vit.enabled", False)
        
        if vit_enabled:
            logger.info("Construction de la branche ViT...")
            logger.warning("ViT non encore implemente dans ce fichier")
            # TODO: Ajouter support ViT si necessaire
        
        # ========================================
        # Creer le FeatureUnion
        # ========================================
        if not transformers:
            raise ValueError("Aucune branche de features activee ! "
                           "Verifier la configuration.")
        
        feature_pipeline = FeatureUnion(
            transformers,
            transformer_weights=weights if weights else None
        )
        
        logger.info(f"[OK] Pipeline cree avec {len(transformers)} branche(s)")
        logger.info(f"  Branches: {[name for name, _ in transformers]}")
        
        return feature_pipeline
    
    def create_feature_mapping(
        self,
        X_transformed: np.ndarray
    ) -> Dict[str, Tuple[int, int]]:
        """
        Cree un mapping des ranges de colonnes pour chaque transformateur.
        
        Gère correctement les Pipelines et FeatureUnions imbriqués.
        Ceci est essentiel pour l'analyse SHAP decomposee par composante.
        
        Args:
            X_transformed: Matrice de features transformees
            
        Returns:
            Dict avec ranges : {'text_tfidf': (0, 100000), 'text_has_desc': (100000, 100001), ...}
        """
        logger.info("\n--- Creation du mapping des features ---")
        
        mapping = {}
        start_idx = 0
        
        # Parcourir les transformateurs du FeatureUnion
        for name, transformer in self.feature_pipeline.transformer_list:
            try:
                # Cas 1: Pipeline sklearn (ex: Pipeline([TextCleaner, TfidfVectorizer]))
                if hasattr(transformer, 'steps'):
                    logger.info(f"  Traitement du Pipeline '{name}'...")
                    # Prendre le dernier step qui génère les features
                    last_step_name, last_step = transformer.steps[-1]
                    
                    if hasattr(last_step, 'get_feature_names_out'):
                        feature_names = last_step.get_feature_names_out()
                        n_features = len(feature_names)
                    else:
                        # Fallback: transformer un échantillon
                        sample = self._original_X_train.iloc[:1]
                        transformed = transformer.transform(sample)
                        n_features = transformed.shape[1]
                    
                    end_idx = start_idx + n_features
                    mapping[name] = (start_idx, end_idx)
                    logger.info(f"    {name:20s}: colonnes {start_idx:6d} - {end_idx:6d}  ({n_features:6d} features)")
                    start_idx = end_idx
                
                # Cas 2: FeatureUnion imbriqué (ex: word_branch qui contient tfidf, has_desc, etc.)
                elif hasattr(transformer, 'transformer_list'):
                    logger.info(f"  Traitement du FeatureUnion imbriqué '{name}'...")
                    sub_start = start_idx
                    
                    for sub_name, sub_trans in transformer.transformer_list:
                        # Gérer les sous-pipelines
                        if hasattr(sub_trans, 'steps'):
                            last_step_name, last_step = sub_trans.steps[-1]
                            if hasattr(last_step, 'get_feature_names_out'):
                                sub_names = last_step.get_feature_names_out()
                                sub_n = len(sub_names)
                            else:
                                sample = self._original_X_train.iloc[:1]
                                transformed = sub_trans.transform(sample)
                                sub_n = transformed.shape[1]
                        # Gérer les transformers simples
                        elif hasattr(sub_trans, 'get_feature_names_out'):
                            sub_names = sub_trans.get_feature_names_out()
                            sub_n = len(sub_names)
                        else:
                            sample = self._original_X_train.iloc[:1]
                            transformed = sub_trans.transform(sample)
                            sub_n = transformed.shape[1]
                        
                        # Créer un nom composite
                        composite_name = f"{name}_{sub_name}"
                        mapping[composite_name] = (sub_start, sub_start + sub_n)
                        logger.info(f"    {composite_name:20s}: colonnes {sub_start:6d} - {sub_start + sub_n:6d}  ({sub_n:6d} features)")
                        sub_start += sub_n
                    
                    n_features = sub_start - start_idx
                    start_idx = sub_start
                
                # Cas 3: Transformer simple (ex: CNNFeaturizer)
                elif hasattr(transformer, 'get_feature_names_out'):
                    feature_names = transformer.get_feature_names_out()
                    n_features = len(feature_names)
                    
                    end_idx = start_idx + n_features
                    mapping[name] = (start_idx, end_idx)
                    logger.info(f"  {name:20s}: colonnes {start_idx:6d} - {end_idx:6d}  ({n_features:6d} features)")
                    start_idx = end_idx
                
                # Cas 4: Fallback - transformer un échantillon
                else:
                    sample = self._original_X_train.iloc[:1]
                    transformed = transformer.transform(sample)
                    
                    if hasattr(transformed, 'shape'):
                        n_features = transformed.shape[1]
                    else:
                        n_features = 1
                    
                    end_idx = start_idx + n_features
                    mapping[name] = (start_idx, end_idx)
                    logger.info(f"  {name:20s}: colonnes {start_idx:6d} - {end_idx:6d}  ({n_features:6d} features)")
                    start_idx = end_idx
                
            except Exception as e:
                logger.error(f"  Erreur lors du mapping de {name}: {e}")
                # Utiliser la shape totale comme fallback
                if start_idx < X_transformed.shape[1]:
                    end_idx = X_transformed.shape[1]
                    mapping[name] = (start_idx, end_idx)
                    n_features = end_idx - start_idx
                    logger.warning(f"  Fallback pour {name}: ({start_idx}, {end_idx}) - {n_features} features")
                    start_idx = end_idx
        
        # Verification finale
        total_mapped = sum(end - start for start, end in mapping.values())
        if total_mapped != X_transformed.shape[1]:
            logger.warning(
                f"  ATTENTION: Mapping incomplet ! "
                f"Total mappe: {total_mapped}, attendu: {X_transformed.shape[1]}"
            )
        else:
            logger.info(f"[OK] Mapping complet: {total_mapped} features")
        
        return mapping
    
    def transform_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        feature_pipeline: FeatureUnion
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforme les donnees avec le pipeline de features.
        
        Args:
            X_train: Features d'entrainement (brutes)
            y_train: Labels d'entrainement  
            X_test: Features de test (brutes)
            feature_pipeline: Pipeline sklearn a appliquer
            
        Returns:
            Tuple (X_train_transformed, X_test_transformed)
        """
        logger.info("\n--- Transformation des donnees ---")
        logger.info(f"[INFO] Train: {len(X_train)} echantillons a transformer")
        logger.info(f"[INFO] Test: {len(X_test)} echantillons a transformer")
        
        # Sauvegarder X_train pour le mapping
        self._original_X_train = X_train.copy()
        
        # ========================================
        # Fit + Transform sur train
        # ========================================
        logger.info("\n[1/2] Fit + Transform sur TRAIN...")
        with Timer("Fit + Transform sur train"):
            X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
        
        logger.info(f"[OK] Train transforme: {X_train_transformed.shape}")
        logger.info(f"  Type: {type(X_train_transformed)}")
        logger.info(f"  Progression: {len(X_train)}/{len(X_train)} échantillons traités ✓")
        
        # Afficher les stats si c'est sparse
        if hasattr(X_train_transformed, 'nnz'):
            density = X_train_transformed.nnz / np.prod(X_train_transformed.shape)
            logger.info(f"  Densite: {density:.4f} ({density*100:.2f}%)")
        
        # ========================================
        # Transform sur test
        # ========================================
        logger.info("\n[2/2] Transform sur TEST...")
        with Timer("Transform sur test"):
            X_test_transformed = feature_pipeline.transform(X_test)
        
        logger.info(f"[OK] Test transforme: {X_test_transformed.shape}")
        
        return X_train_transformed, X_test_transformed
    
    def _get_cache_key(self, X_train: pd.DataFrame, y_train: pd.Series) -> str:
        """
        Génère une clé de cache basée sur les données et la config.
        
        Args:
            X_train: DataFrame d'entraînement
            y_train: Labels d'entraînement
            
        Returns:
            Nom du fichier de cache
        """
        import hashlib
        import json
        
        # Hash des index (représente les données)
        data_hash = hashlib.md5(
            str(sorted(X_train.index)).encode()
        ).hexdigest()[:8]
        
        # Hash de la config importante
        config_dict = {
            "text": self.config.get("text", {}),
            "sampling": self.config.sampling,
            "random_seed": self.config.random_seed
        }
        config_hash = hashlib.md5(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"features_d{data_hash}_c{config_hash}.joblib"
    
    def _load_from_cache(self, cache_file) -> Tuple:
        """Charge les features depuis le cache."""
        import joblib
        
        logger.info(f" Chargement des features depuis le cache...")
        logger.info(f"   Fichier: {cache_file.name}")
        
        with Timer("Chargement du cache"):
            cached = joblib.load(cache_file)
        
        logger.info(f" Cache chargé avec succès!")
        logger.info(f"   X_train: {cached['X_train_t'].shape}")
        logger.info(f"   X_test: {cached['X_test_t'].shape}")
        
        return (
            cached["X_train_t"],
            cached["y_train_t"],
            cached["X_test_t"],
            cached["feature_pipeline"],
            cached["feature_mapping"]
        )
    
    def _save_to_cache(
        self, 
        cache_file,
        X_train_t,
        y_train_t,
        X_test_t,
        feature_pipeline,
        feature_mapping
    ):
        """Sauvegarde les features dans le cache."""
        import joblib
        
        logger.info(f" Sauvegarde dans le cache...")
        logger.info(f"   Fichier: {cache_file.name}")
        
        with Timer("Sauvegarde du cache"):
            joblib.dump({
                "X_train_t": X_train_t,
                "y_train_t": y_train_t,
                "X_test_t": X_test_t,
                "feature_pipeline": feature_pipeline,
                "feature_mapping": feature_mapping
            }, cache_file)
        
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        logger.info(f" Cache sauvegardé ({size_mb:.1f} MB)")
    
    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.Series, np.ndarray, FeatureUnion, Dict[str, Tuple[int, int]]]:
        """
        Execute le pipeline de transformation complet avec cache.
        
        Args:
            X_train: Features d'entrainement (brutes)
            y_train: Labels d'entrainement
            X_test: Features de test (brutes)
            
        Returns:
            Tuple (X_train_transformed, y_train_resampled, 
                   X_test_transformed, feature_pipeline, feature_mapping)
        """
        # ========================================
        # ESSAYER DE CHARGER LE CACHE
        # ========================================
        if self.use_cache:
            cache_file = self.cache_dir / self._get_cache_key(X_train, y_train)
            
            if cache_file.exists():
                try:
                    return self._load_from_cache(cache_file)
                except Exception as e:
                    logger.warning(f"  Erreur lors du chargement du cache: {e}")
                    logger.warning("   Recalcul des features...")
        
        # ========================================
        # CALCUL NORMAL (si pas de cache)
        # ========================================
        logger.info("  Calcul des features (pas de cache valide)...")
        
        with Timer("Transformation des donnees"):
            
            # ========================================
            # 1. Reechantillonnage
            # ========================================
            X_train_resampled, y_train_resampled = self.apply_resampling(
                X_train, y_train
            )
            
            # ========================================
            # 2. Construction du pipeline
            # ========================================
            self.feature_pipeline = self.build_feature_pipeline()
            
            # ========================================
            # 3. Transformation
            # ========================================
            X_train_transformed, X_test_transformed = self.transform_data(
                X_train_resampled,
                y_train_resampled,
                X_test,
                self.feature_pipeline
            )
            
            # ========================================
            # 4. Creation du mapping des features
            # ========================================
            self.feature_mapping = self.create_feature_mapping(X_train_transformed)
            
            # ========================================
            # 5. Resume final
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info("RESUME DE LA TRANSFORMATION")
            logger.info("=" * 70)
            logger.info(f"[INFO] X_train : {X_train.shape} -> {X_train_transformed.shape}")
            logger.info(f"[INFO] y_train : {y_train.shape} -> {y_train_resampled.shape}")
            logger.info(f"[INFO] X_test  : {X_test.shape} -> {X_test_transformed.shape}")
            logger.info(f"[INFO] Pipeline sauvegarde : {self.feature_pipeline is not None}")
            logger.info(f"[INFO] Feature mapping cree : {len(self.feature_mapping)} composantes")
            logger.info("=" * 70 + "\n")
            
            # ========================================
            # 6. Sauvegarder dans le cache
            # ========================================
            if self.use_cache:
                cache_file = self.cache_dir / self._get_cache_key(X_train, y_train)
                try:
                    self._save_to_cache(
                        cache_file,
                        X_train_transformed,
                        y_train_resampled,
                        X_test_transformed,
                        self.feature_pipeline,
                        self.feature_mapping
                    )
                except Exception as e:
                    logger.warning(f"  Erreur lors de la sauvegarde du cache: {e}")
            
            return (
                X_train_transformed,
                y_train_resampled,
                X_test_transformed,
                self.feature_pipeline,
                self.feature_mapping
            )


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.config import load_config
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
    
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Test de DataTransformationPipeline")
    print("="*70 + "\n")
    
    try:
        # Charger la configuration
        config = load_config()
        
        # Stage 1: Ingestion
        stage1 = DataIngestionPipeline(config)
        X_train, y_train, X_test = stage1.run()
        
        # Stage 2: Validation
        stage2 = DataValidationPipeline(config)
        validation_ok = stage2.run(X_train, y_train, X_test)
        
        if not validation_ok:
            print("\n[ERROR] Validation echouee - arret du pipeline")
        else:
            # Stage 3: Transformation
            stage3 = DataTransformationPipeline(config)
            X_train_t, y_train_t, X_test_t, pipeline, mapping = stage3.run(
                X_train, y_train, X_test
            )
            
            print("\n[OK] Transformation terminee avec succes!")
            print(f"  X_train transforme: {X_train_t.shape}")
            print(f"  y_train reechantillonne: {y_train_t.shape}")
            print(f"  X_test transforme: {X_test_t.shape}")
            print(f"  Feature mapping: {len(mapping)} composantes")
            for comp, (start, end) in mapping.items():
                print(f"    {comp}: colonnes {start}-{end}")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Erreur: {e}")
        print("Verification que les fichiers CSV existent dans data/raw/")
    except Exception as e:
        print(f"\n[ERROR] Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()





























