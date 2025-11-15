#!/usr/bin/env bash
export MSYS_NO_PATHCONV=1
# demande bash sous Windows pour éviter la conversion des chemins par MSYS2
# Initialise la base Rakuten : schéma, ingestion et enregistrement du hash.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
SQL_DIR="$ROOT_DIR/sql"
PY_SCRIPT="$ROOT_DIR/etl/manifest_and_hash.py"

# Sous Git Bash (Windows), convertir en chemin Windows pour que python.exe le comprenne
if command -v cygpath >/dev/null 2>&1; then
  PY_SCRIPT_WIN="$(cygpath -w "$PY_SCRIPT")"
else
  PY_SCRIPT_WIN="$PY_SCRIPT"
fi

if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  COMPOSE_CMD="docker compose"
fi

# Démarre (ou redémarre) le service Postgres si nécessaire.
$COMPOSE_CMD up -d postgres >/dev/null 2>&1 || true

echo ">>> Attente de la disponibilité de Postgres..."
until $COMPOSE_CMD exec -T postgres pg_isready -U mlops -d rakuten >/dev/null 2>&1; do
  sleep 2
  echo "    ... toujours en attente"
done

echo "Postgres est prêt. Calcul du hash des CSV..."
# Détection automatique Windows vs Linux
if command -v cygpath >/dev/null 2>&1; then
  # Windows (Git Bash)
  echo "[INFO] Windows détecté - utilisation du chemin Windows"
  DATA_HASH=$(python "$PY_SCRIPT_WIN")
else
  # Linux / macOS
  echo "[INFO] Linux/Mac détecté - utilisation du chemin UNIX"
  DATA_HASH=$(python "$PY_SCRIPT")
fi

if [[ -z "$DATA_HASH" ]]; then
  echo "Impossible de récupérer le hash des données."
  exit 1
fi

echo "Hash combiné : $DATA_HASH"

SQL_FILES=(
  00_schema.sql
  01_tables.sql
  02_indexes.sql
  03_views.sql
  09_copy_raw.sql
  10_load_staging.sql
  11_upsert_items.sql
)

for sql_file in "${SQL_FILES[@]}"; do
  echo ">>> Exécution de $sql_file"
  $COMPOSE_CMD exec -T postgres psql -U mlops -d rakuten -v ON_ERROR_STOP=1 -f "/sql/$sql_file"
done

echo ">>> Enregistrement du snapshot dans project.datasets"
$COMPOSE_CMD exec -T postgres \
  psql -U mlops -d rakuten -v ON_ERROR_STOP=1 -v data_hash="$DATA_HASH" -f "/sql/12_register_snapshot.sql"

echo ">>> Exécution des contrôles finaux"
$COMPOSE_CMD exec -T postgres psql -U mlops -d rakuten -v ON_ERROR_STOP=1 -f "/sql/20_checks.sql"

echo ">>> Aperçu final"
$COMPOSE_CMD exec -T postgres psql -U mlops -d rakuten -c "SELECT COUNT(*) AS nb_items FROM project.items;"
$COMPOSE_CMD exec -T postgres psql -U mlops -d rakuten -c "SELECT prdtypecode, COUNT(*) FROM project.items GROUP BY 1 ORDER BY 2 DESC LIMIT 10;"

echo "Fin de l'initialisation."
