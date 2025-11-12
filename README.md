# Pipeline Postgres Rakuten MLOps

Ce module fournit une base PostgreSQL 16 prête à l'emploi pour charger les CSV du challenge Rakuten (classification de produits e-commerce) et alimenter une table `project.items` prête pour les pipelines de features/MLflow. Les scripts fournis créent le schéma, assurent l'ingestion complète des fichiers `X_train_update.csv`, `Y_train_CVw08PX.csv`, `X_test_update.csv` et tracent chaque import via un hash combiné.

## 1. Pré-requis
- Docker + Docker Compose (plugin v2 ou `docker-compose`).
- Python 3.9+ disponible sur la machine hôte pour exécuter `etl/manifest_and_hash.py`.
- Les trois CSV Rakuten déposés dans `./import/` (les noms doivent correspondre exactement).

## 2. Installation et lancement
```bash
docker compose up -d  # démarre le conteneur Postgres nommé rakuten_db
bash init.sh          # crée le schéma, charge les CSV, calcule le hash et enregistre le snapshot
```
Le script `init.sh` rejouera toutes les étapes si vous ajoutez ou remplacez des CSV (le hash changera automatiquement).

## 3. Requêtes de vérification rapides
Connectez-vous avec `psql postgres://mlops:mlops@localhost:5433/rakuten` puis lancez :
```sql
SELECT COUNT(*) FROM project.items;
SELECT prdtypecode, COUNT(*) FROM project.items GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
```
Des vues pratiques (`project.vw_items_by_class`, `project.vw_dataset_inventory`, `project.vw_latest_models`) sont également disponibles.

## 4. Dépannage
- **Encodage CSV** : les fichiers doivent être UTF-8. Si vous venez d'une autre plateforme, convertissez-les (`iconv -f ISO-8859-1 -t UTF-8 fichier.csv > fichier_utf8.csv`).
- **Permissions Docker** : assurez-vous d'avoir les droits de lecture sur `./import`. Le volume est monté en lecture seule dans le conteneur.
- **pg_trgm manquant** : l'extension est créée automatiquement, mais si elle échoue, vérifiez que l'image Postgres est bien en version 16 officielle.
- **Hash non généré** : le script Python doit voir au moins un CSV dans `./import`; vérifiez le chemin avant de rejouer `init.sh`.

## 5. Structure SQL et flux
1. `docker-compose.yml` démarre un Postgres 16 (`mlops/mlops/rakuten`) et monte `./import` + `./sql`.
2. Les scripts `sql/00_schema.sql` → `sql/11_upsert_items.sql` créent le schéma, les tables, importent les CSV via `COPY`, nettoient les données et alimentent `project.items`.
3. `etl/manifest_and_hash.py` calcule un SHA256 combiné sur les CSV et écrit `snapshot.json` (+ manifeste optionnel des images si `data/images` existe).
4. `sql/12_register_snapshot.sql` enregistre le hash dans `project.datasets` pour la traçabilité; `sql/20_checks.sql` effectue des contrôles de cohérence.

### Tables principales
- `project.items` : produits prêts pour l'entraînement, obtenus par jointure `stg_x_train` ↔ `stg_y_train`.
- `project.datasets` : historique des imports avec hash, note et horodatage.
- `project.splits` : emplacements réservés pour stocker les splits futurs rattachés à un dataset versionné.

Rejouer `init.sh` après la mise à jour des CSV suffit pour rafraîchir la base tout en conservant l'historique des hashes.
