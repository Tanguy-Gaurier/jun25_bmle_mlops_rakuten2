-- Vues métiers pour faciliter l'analyse rapide depuis psql ou des notebooks.

SET search_path TO project, public;

CREATE OR REPLACE VIEW project.vw_items_by_class AS
SELECT
    prdtypecode,
    COUNT(*) AS nb_items
FROM project.items
GROUP BY prdtypecode
ORDER BY nb_items DESC;
COMMENT ON VIEW project.vw_items_by_class IS 'Vue de contrôle donnant le volume d''items par prdtypecode.';

CREATE OR REPLACE VIEW project.vw_dataset_inventory AS
SELECT
    d.dataset_id,
    d.source,
    d.data_hash,
    d.created_at,
    COUNT(s.productid) AS nb_items
FROM project.datasets d
LEFT JOIN project.splits s ON s.dataset_id = d.dataset_id
GROUP BY d.dataset_id, d.source, d.data_hash, d.created_at
ORDER BY d.created_at DESC;
COMMENT ON VIEW project.vw_dataset_inventory IS 'Inventaire des datasets et du nombre de lignes déclarées par split.';

CREATE OR REPLACE VIEW project.vw_latest_models AS
SELECT
    d.dataset_id,
    d.created_at AS dataset_loaded_at,
    d.note AS training_note,
    ('model_rakuten_' || d.dataset_id)::TEXT AS model_stub,
    d.data_hash
FROM project.datasets d
ORDER BY d.created_at DESC
LIMIT 20;
COMMENT ON VIEW project.vw_latest_models IS 'Stub indiquant les derniers datasets chargés, base pour suivre les entraînements.';
