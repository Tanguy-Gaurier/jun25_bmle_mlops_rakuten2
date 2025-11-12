-- Contrôles de cohérence simples pour valider l'import.

SET search_path TO project, public;

-- Vérifie les doublons potentiels.
WITH dup AS (
    SELECT productid, COUNT(*) AS c
    FROM project.items
    GROUP BY productid
    HAVING COUNT(*) > 1
)
SELECT 'doublons_items' AS test, COALESCE(SUM(c), 0) AS valeur
FROM dup;

-- Vérifie l'alignement entre X_train et Y_train.
SELECT 'rows_jointure_train' AS test,
       (SELECT COUNT(*) FROM project.stg_x_train) AS nb_x_train,
       (SELECT COUNT(*) FROM project.stg_y_train) AS nb_y_train,
       (SELECT COUNT(*) FROM project.stg_x_train INNER JOIN project.stg_y_train USING (row_index)) AS nb_match
;

-- Vérifie que tous les items portent un label.
SELECT 'items_sans_label' AS test, COUNT(*) AS valeur
FROM project.items
WHERE prdtypecode IS NULL;

-- Distribution par split (si alimentée).
SELECT dataset_id, split, COUNT(*) AS nb_items
FROM project.splits
GROUP BY dataset_id, split
ORDER BY dataset_id, split;
