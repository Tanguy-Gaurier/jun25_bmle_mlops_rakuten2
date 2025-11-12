-- Indexation ciblée pour accélérer les requêtes métiers.
-- À exécuter après la création des tables.

SET search_path TO project, public;

-- Active l'extension trigram pour supporter les recherches textuelles floues.
CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;

-- Optimise les filtres fréquents sur les labels.
CREATE INDEX IF NOT EXISTS idx_items_prdtypecode
    ON project.items (prdtypecode);
COMMENT ON INDEX idx_items_prdtypecode IS 'Index BTREE pour accélérer les agrégations et filtres par prdtypecode.';

-- Permet les recherches textuelles rapides sur designation/description.
CREATE INDEX IF NOT EXISTS idx_items_textsearch
    ON project.items
    USING GIN ((coalesce(designation, '') || ' ' || coalesce(description, '')) gin_trgm_ops);
COMMENT ON INDEX idx_items_textsearch IS 'Index GIN + trigrammes pour la similarité textuelle sur les contenus produits.';
