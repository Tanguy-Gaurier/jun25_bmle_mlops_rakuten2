-- Enregistrement du snapshot courant dans project.datasets.
-- Le script init.sh fournit la variable psql :data_hash.

\echo 'Insertion d\'un nouveau snapshot avec le hash fourni...'

INSERT INTO project.datasets (source, data_hash, note)
VALUES ('import_local', :'data_hash', 'Import initial des CSV Rakuten')
RETURNING dataset_id;

-- Rappel de l\'inventaire pour contrôle.
SELECT dataset_id, source, data_hash, created_at
FROM project.datasets
ORDER BY dataset_id DESC
LIMIT 5;
