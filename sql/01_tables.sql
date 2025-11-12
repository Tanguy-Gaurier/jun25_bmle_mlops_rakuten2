-- Définition des tables permanentes utilisées dans le pipeline.
-- Inclut les zones de staging, les tables métier et le suivi des jeux de données.

SET search_path TO project, public;

-- Table de staging pour les features du train (X). Permet de conserver les valeurs brutes nettoyées.
CREATE TABLE IF NOT EXISTS project.stg_x_train (
    row_index   BIGINT,
    designation TEXT,
    description TEXT,
    productid   BIGINT,
    imageid     BIGINT
);
COMMENT ON TABLE project.stg_x_train IS 'Zone de staging pour les features X_train après nettoyage minimal.';
COMMENT ON COLUMN project.stg_x_train.row_index IS 'Index original fourni dans le CSV, utilisé comme clé technique.';
COMMENT ON COLUMN project.stg_x_train.designation IS 'Titre produit nettoyé.';
COMMENT ON COLUMN project.stg_x_train.description IS 'Description longue avec NULL pour les valeurs vides.';
COMMENT ON COLUMN project.stg_x_train.productid IS 'Identifiant produit fourni par Rakuten.';
COMMENT ON COLUMN project.stg_x_train.imageid IS 'Identifiant image associé au produit.';

-- Table de staging pour les labels du train (Y). On conserve uniquement l’index et le code produit.
CREATE TABLE IF NOT EXISTS project.stg_y_train (
    row_index   BIGINT,
    prdtypecode INT
);
COMMENT ON TABLE project.stg_y_train IS 'Zone de staging pour les labels Y_train, alignés sur row_index.';
COMMENT ON COLUMN project.stg_y_train.row_index IS 'Index d’alignement partagé avec X_train.';
COMMENT ON COLUMN project.stg_y_train.prdtypecode IS 'Code de type produit utilisé comme label de classification.';

-- Table de staging pour les features du test (X_test) en vue d’analyses futures.
CREATE TABLE IF NOT EXISTS project.stg_x_test (
    row_index   BIGINT,
    designation TEXT,
    description TEXT,
    productid   BIGINT,
    imageid     BIGINT
);
COMMENT ON TABLE project.stg_x_test IS 'Zone de staging pour les features X_test.';
COMMENT ON COLUMN project.stg_x_test.row_index IS 'Index fourni dans X_test.';
COMMENT ON COLUMN project.stg_x_test.designation IS 'Titre produit pour les données de test.';
COMMENT ON COLUMN project.stg_x_test.description IS 'Description produit pour les données de test.';
COMMENT ON COLUMN project.stg_x_test.productid IS 'Identifiant produit dans X_test.';
COMMENT ON COLUMN project.stg_x_test.imageid IS 'Identifiant image dans X_test.';

-- Table métier principale contenant les items prêts pour la consommation MLOps.
CREATE TABLE IF NOT EXISTS project.items (
    productid    BIGINT PRIMARY KEY,
    imageid      BIGINT,
    designation  TEXT,
    description  TEXT NULL,
    prdtypecode  INT NOT NULL
);
COMMENT ON TABLE project.items IS 'Table finale contenant les items enrichis, prête pour les pipelines d''entraînement.';
COMMENT ON COLUMN project.items.productid IS 'Identifiant unique d''un produit Rakuten.';
COMMENT ON COLUMN project.items.imageid IS 'Identifiant image le plus récent connu pour ce produit.';
COMMENT ON COLUMN project.items.designation IS 'Titre du produit prêt pour la recherche textuelle.';
COMMENT ON COLUMN project.items.description IS 'Description alignée sur la dernière ingestion; NULL si non fournie.';
COMMENT ON COLUMN project.items.prdtypecode IS 'Label cible utilisé dans les modèles de classification.';

-- Table de référence pour stocker les métadonnées des assets visuels.
CREATE TABLE IF NOT EXISTS project.assets (
    imageid    BIGINT PRIMARY KEY,
    image_uri  TEXT,
    checksum   TEXT,
    size_bytes INT,
    w          INT,
    h          INT
);
COMMENT ON TABLE project.assets IS 'Référentiel optionnel des images produits (à compléter selon les besoins).';
COMMENT ON COLUMN project.assets.image_uri IS 'Emplacement ou URL de l''image.';
COMMENT ON COLUMN project.assets.checksum IS 'Empreinte pour contrôler l''intégrité du fichier image.';
COMMENT ON COLUMN project.assets.size_bytes IS 'Poids du fichier image en octets.';
COMMENT ON COLUMN project.assets.w IS 'Largeur en pixels.';
COMMENT ON COLUMN project.assets.h IS 'Hauteur en pixels.';

-- Historique des jeux de données chargés pour assurer une traçabilité simple.
CREATE TABLE IF NOT EXISTS project.datasets (
    dataset_id SERIAL PRIMARY KEY,
    source     TEXT,
    data_hash  TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    note       TEXT
);
COMMENT ON TABLE project.datasets IS 'Historique des snapshots importés avec hash de référence.';
COMMENT ON COLUMN project.datasets.source IS 'Origine déclarée (local, blob, etc.).';
COMMENT ON COLUMN project.datasets.data_hash IS 'Hash SHA256 combiné des fichiers importés.';
COMMENT ON COLUMN project.datasets.created_at IS 'Horodatage de l''insertion.';
COMMENT ON COLUMN project.datasets.note IS 'Notes libres pour contextualiser l''import.';

-- Table des splits rattachés à un dataset donné (train/val/test).
CREATE TABLE IF NOT EXISTS project.splits (
    dataset_id INT REFERENCES project.datasets(dataset_id),
    productid  BIGINT,
    split      TEXT CHECK (split IN ('train', 'val', 'test')),
    PRIMARY KEY (dataset_id, productid)
);
COMMENT ON TABLE project.splits IS 'Association entre un dataset versionné et les produits qui le composent par split.';
COMMENT ON COLUMN project.splits.split IS 'Split logique : train, validation ou test.';
