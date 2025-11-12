#!/usr/bin/env python3
"""Calcule un hash SHA256 combiné des CSV présents dans ./import et génère un snapshot."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_DIR = REPO_ROOT / "import"
SNAPSHOT_PATH = REPO_ROOT / "snapshot.json"
IMAGES_DIR = REPO_ROOT / "data" / "images"
IMAGES_MANIFEST = REPO_ROOT / "images_manifest.json"


def list_csv_files(directory: Path) -> List[Path]:
    """Retourne les CSV à prendre en compte, triés pour garantir la reproductibilité."""
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Aucun fichier CSV trouvé dans ./import : impossible de calculer le hash.")
    return files


def sha256_files(files: Iterable[Path]) -> str:
    """Calcule une empreinte SHA256 en concaténant nom + contenu de chaque fichier."""
    hasher = hashlib.sha256()
    for csv_file in files:
        logging.info("Lecture de %s", csv_file)
        hasher.update(csv_file.name.encode("utf-8"))
        with csv_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def write_snapshot(data_hash: str, files: List[Path]) -> None:
    payload = {
        "data_hash": data_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [f.name for f in files],
    }
    SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logging.info("snapshot.json mis à jour (%s)", SNAPSHOT_PATH)


def maybe_write_images_manifest() -> None:
    if not IMAGES_DIR.exists():
        logging.info("Aucun répertoire data/images détecté : manifeste d'images ignoré.")
        return

    manifest = []
    for image_path in sorted(IMAGES_DIR.rglob("*")):
        if image_path.is_file():
            stat = image_path.stat()
            manifest.append(
                {
                    "relative_path": str(image_path.relative_to(REPO_ROOT)),
                    "size_bytes": stat.st_size,
                }
            )
    IMAGES_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logging.info("images_manifest.json généré (%d fichiers)", len(manifest))


def main() -> None:
    csv_files = list_csv_files(IMPORT_DIR)
    data_hash = sha256_files(csv_files)
    write_snapshot(data_hash, csv_files)
    maybe_write_images_manifest()
    print(data_hash)


if __name__ == "__main__":
    main()
