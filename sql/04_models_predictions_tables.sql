-- Tables supplémentaires pour le suivi des modèles et prédictions
-- À exécuter après 01_tables.sql

SET search_path TO project, public;

-- Table de suivi des modèles entraînés
CREATE TABLE IF NOT EXISTS project.models (
    model_id    SERIAL PRIMARY KEY,
    model_name  TEXT NOT NULL,
    model_type  TEXT,
    dataset_hash TEXT,
    metrics     JSONB,
    model_path  TEXT,
    config      JSONB,
    note        TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);
COMMENT ON TABLE project.models IS 'Historique des modèles entraînés avec leurs métriques.';
COMMENT ON COLUMN project.models.model_id IS 'Identifiant unique du modèle.';
COMMENT ON COLUMN project.models.model_name IS 'Nom du modèle (ex: rakuten_xgb_20241115).';
COMMENT ON COLUMN project.models.model_type IS 'Type de modèle (XGB, LGBM, etc.).';
COMMENT ON COLUMN project.models.dataset_hash IS 'Hash du dataset utilisé pour l''entraînement.';
COMMENT ON COLUMN project.models.metrics IS 'Métriques de performance (JSON).';
COMMENT ON COLUMN project.models.model_path IS 'Chemin vers le fichier du modèle sérialisé.';
COMMENT ON COLUMN project.models.config IS 'Configuration du modèle (hyperparamètres, etc.).';
COMMENT ON COLUMN project.models.note IS 'Notes sur l''entraînement.';
COMMENT ON COLUMN project.models.created_at IS 'Date de création du modèle.';

-- Table des prédictions
CREATE TABLE IF NOT EXISTS project.predictions (
    prediction_id  SERIAL PRIMARY KEY,
    model_id       INT REFERENCES project.models(model_id),
    product_id     BIGINT,
    predicted_class INT,
    confidence     FLOAT,
    metadata       JSONB,
    created_at     TIMESTAMP DEFAULT NOW()
);
COMMENT ON TABLE project.predictions IS 'Prédictions générées par les modèles.';
COMMENT ON COLUMN project.predictions.prediction_id IS 'Identifiant unique de la prédiction.';
COMMENT ON COLUMN project.predictions.model_id IS 'Référence au modèle ayant fait la prédiction.';
COMMENT ON COLUMN project.predictions.product_id IS 'Produit concerné par la prédiction.';
COMMENT ON COLUMN project.predictions.predicted_class IS 'Classe prédite.';
COMMENT ON COLUMN project.predictions.confidence IS 'Score de confiance.';
COMMENT ON COLUMN project.predictions.metadata IS 'Métadonnées supplémentaires (JSON).';
COMMENT ON COLUMN project.predictions.created_at IS 'Date de la prédiction.';

-- Index pour améliorer les performances
CREATE INDEX IF NOT EXISTS idx_models_dataset_hash ON project.models (dataset_hash);
CREATE INDEX IF NOT EXISTS idx_models_created_at ON project.models (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON project.predictions (model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_product_id ON project.predictions (product_id);
