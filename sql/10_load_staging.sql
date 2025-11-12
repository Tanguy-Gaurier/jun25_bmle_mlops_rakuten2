-- Nettoyage minimal et transfert des données brutes vers les tables de staging.

SET search_path TO project, public;

TRUNCATE TABLE project.stg_x_train;
INSERT INTO project.stg_x_train (row_index, designation, description, productid, imageid)
SELECT
    col1::BIGINT AS row_index,
    NULLIF(BTRIM(col2), '') AS designation,
    NULLIF(BTRIM(col3), '') AS description,
    NULLIF(BTRIM(col4), '')::BIGINT AS productid,
    NULLIF(BTRIM(col5), '')::BIGINT AS imageid
FROM project._raw_x_train;

TRUNCATE TABLE project.stg_y_train;
INSERT INTO project.stg_y_train (row_index, prdtypecode)
SELECT
    col1::BIGINT AS row_index,
    NULLIF(BTRIM(col2), '')::INT AS prdtypecode
FROM project._raw_y_train;

TRUNCATE TABLE project.stg_x_test;
INSERT INTO project.stg_x_test (row_index, designation, description, productid, imageid)
SELECT
    col1::BIGINT AS row_index,
    NULLIF(BTRIM(col2), '') AS designation,
    NULLIF(BTRIM(col3), '') AS description,
    NULLIF(BTRIM(col4), '')::BIGINT AS productid,
    NULLIF(BTRIM(col5), '')::BIGINT AS imageid
FROM project._raw_x_test;
