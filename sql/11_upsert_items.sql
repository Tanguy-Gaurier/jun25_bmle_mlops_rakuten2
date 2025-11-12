-- Upsert entre les tables de staging et la table finale items.

SET search_path TO project, public;

INSERT INTO project.items (productid, imageid, designation, description, prdtypecode)
SELECT
    x.productid,
    x.imageid,
    x.designation,
    x.description,
    y.prdtypecode
FROM project.stg_x_train AS x
INNER JOIN project.stg_y_train AS y USING (row_index)
WHERE x.productid IS NOT NULL
ON CONFLICT (productid) DO UPDATE
SET
    imageid = EXCLUDED.imageid,
    designation = EXCLUDED.designation,
    description = EXCLUDED.description,
    prdtypecode = EXCLUDED.prdtypecode;

ANALYZE project.items;

-- Vérification rapide du remplissage.
SELECT COUNT(*) AS nb_items FROM project.items;
SELECT prdtypecode, COUNT(*) AS nb_par_label
FROM project.items
GROUP BY prdtypecode
ORDER BY nb_par_label DESC
LIMIT 10;
