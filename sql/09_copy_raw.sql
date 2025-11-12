-- Copie brute des CSV dans des tables temporaires _raw_*.
-- Ces tables servent uniquement d'étape intermédiaire avant le nettoyage.

SET search_path TO project, public;

DROP TABLE IF EXISTS project._raw_x_train;
CREATE UNLOGGED TABLE project._raw_x_train (
    col1 TEXT,
    col2 TEXT,
    col3 TEXT,
    col4 TEXT,
    col5 TEXT
);
COMMENT ON TABLE project._raw_x_train IS 'Copie brute de X_train_update.csv (toutes les colonnes en TEXT).';

DROP TABLE IF EXISTS project._raw_y_train;
CREATE UNLOGGED TABLE project._raw_y_train (
    col1 TEXT,
    col2 TEXT
);
COMMENT ON TABLE project._raw_y_train IS 'Copie brute de Y_train_CVw08PX.csv.';

DROP TABLE IF EXISTS project._raw_x_test;
CREATE UNLOGGED TABLE project._raw_x_test (
    col1 TEXT,
    col2 TEXT,
    col3 TEXT,
    col4 TEXT,
    col5 TEXT
);
COMMENT ON TABLE project._raw_x_test IS 'Copie brute de X_test_update.csv.';

-- Import physique des fichiers déposés dans ./import (montés dans /import).
COPY project._raw_x_train
FROM '/import/X_train_update.csv'
WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

COPY project._raw_y_train
FROM '/import/Y_train_CVw08PX.csv'
WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

COPY project._raw_x_test
FROM '/import/X_test_update.csv'
WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
