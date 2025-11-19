# Rakuten Multimodal MLOps Platform

Plateforme complète pour la classification produits du challenge Rakuten, combinant pipelines texte et image, orchestration MLOps, suivi des versions de données et diffusion via Streamlit.

---

## 1. Présentation & TL;DR

- **Objectif** : entraîner, évaluer et servir un modèle multimodal (texte + image) pour classer les produits Rakuten (codes `prdtypecode`).
- **Dataset** : fichiers CSV officiels (`X_train_update.csv`, `Y_train_CVw08PX.csv`, `X_test_update.csv`) et images associées, versionnés via `snapshot.json`.
- **Technos clés** : PostgreSQL 16, pipelines scikit-learn/ResNet, PyTorch, XGBoost/LGBM, Streamlit, SHAP, Docker.
- **Scripts principaux** : `init.sh`, `scripts/train_pipeline.py`, `streamlit_app/config.py`, outils sous `tools/`.

---

## 2. Vue d’ensemble de l’architecture

```mermaid
flowchart LR
    subgraph Data Layer
        CSV[X/Y CSV<br/>import/] -->|COPY| PG[(Postgres 16)]
        Images[data/images] --> Pipelines
        snapshot[snapshot.json]
    end

    subgraph Pipelines (src/pipeline_steps)
        DI(Data Ingestion) --> DV(Data Validation) --> DT(Data Transformation)
        DT --> MT(Model Training) --> ME(Model Evaluation)
    end

    PG --> DI
    DT --> Artifacts[(artifacts/, models/, results/)]
    Artifacts --> Streamlit
    Streamlit --> Users((Utilisateurs))
    tools[tools/* scripts] --> Pipelines
```

- **Base de données** : `docker-compose.yml` expose PostgreSQL (`rakuten_db`) avec tout le schéma `project.*`.
- **Pipeline ML** : `scripts/train_pipeline.py` orchestre les 5 étapes sous `src/pipeline_steps`.
- **Serving** : `streamlit_app/` charge les artefacts Joblib/NPZ pour proposer une expérience interactive, complétée par une inference lite (`streamlit_app/inference_lite.py`).

---

## 3. Actifs data & gouvernance

- **Ingestion CSV** : placer les trois fichiers fournis par Rakuten dans `import/` (noms strictement identiques).  
  - `init.sh` déclenche `etl/manifest_and_hash.py` qui calcule un SHA256 combiné et génère `snapshot.json`.
  - Les scripts SQL (`sql/00_schema.sql` → `sql/11_upsert_items.sql`) créent le schéma `project`, importent les données en staging, joignent `prdtypecode` et exposent la table centrale `project.items`.
- **Traçabilité** : `sql/12_register_snapshot.sql` stocke le hash dans `project.datasets` ; `sql/20_checks.sql` valide volumes et intégrité.
- **Mapping classes** : `config/labels_map.json` contient les libellés métiers utilisés par Streamlit et les rapports.
- **Manifeste image** : si `data/images` est présent, `images_manifest.json` est généré automatiquement (taille, chemin relatif).

---

## 4. Mise en route & prérequis

### 4.1 Base Postgres (Docker)
```bash
docker compose up -d postgres            # démarre rakuten_db
bash init.sh                             # charge les CSV, crée les tables et enregistre le snapshot
```
- Accès local : `postgres://mlops:mlops@localhost:5433/rakuten`.
- Vérification rapide :
```sql
SELECT COUNT(*) FROM project.items;
SELECT prdtypecode, COUNT(*) FROM project.items GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
```

### 4.2 Environnement ML
- **Python** : 3.10+ recommandé, installer `pip install -r requirements.txt`.
- **GPU** : facultatif (accélère `CNNFeaturizer` via PyTorch). Les paramètres `features.image.cnn.device` et `batch_size` sont configurables.
- **Configuration** : `config/config.toml` centralise chemins, seeds, options de features, résampling et fusion multimodale.
- **Images** : `config.images.train_dir` et `test_dir` pointent vers `data/images/images/...`. Garantir la présence des couples `image_{imageid}_product_{productid}.jpg`.

---

## 5. Flux MLOps bout en bout

`python scripts/train_pipeline.py` orchestre les 5 étapes principales :

| Étape | Module | Entrées | Sorties |
| --- | --- | --- | --- |
| 1. Data Ingestion | `src/pipeline_steps/stage01_data_ingestion.py` | CSV Postgres / fichiers | `X_train`, `y_train`, `X_test` alignés |
| 2. Data Validation | `stage02_data_validation.py` | Données brutes | Rapports schema/qualité, sanity checks |
| 3. Data Transformation | `stage03_data_transformation.py` | Données validées + config features | Matrices sparse/denses, pipeline sklearn fit, mapping des features |
| 4. Model Training | `stage04_model_training.py` | Matrices transformées | Modèle entraîné, scoring CV optionnel |
| 5. Model Evaluation | `stage05_model_evaluation.py` | Modèle + jeux transformés | Rapports metrics (`results/metrics/`), matrices de confusion, SHAP |

```mermaid
sequenceDiagram
    participant CSV as CSV / Images
    participant Pipeline as scripts/train_pipeline.py
    participant Artifacts as artifacts/ & results/
    CSV->>Pipeline: Chargement + validation
    Pipeline->>Pipeline: Transformation multimodale
    Pipeline->>Artifacts: Modèles, pipelines, diagnostics
    Artifacts->>Streamlit: Chargement des artefacts
```

- Exécution partielle : `python scripts/train_pipeline.py --skip-validation`, `--cv`, `--evaluate-on-train`.
- Mode profiling rapide : `python tools/test_pipeline_sample.py --sample-size 2000 --profile-features`.

---

## 6. Feature engineering multimodale

### Texte (`src/pipelines/text_pipeline.py`)
- Nettoyage avancé (`TextCleaner`) : normalisation Unicode, dictionnaire de traduction (`config/translate_map_starter_from_cleaned.json`), stemming, gestion emojis.
- TF-IDF word (n-grams 1-2, `max_features` configurable) + options char.
- Features additionnelles : indicateur de description, longueur de titre, statistiques lexicales (`TextStatistics`, `TextStatisticsPro`), détection de langue, lexique chi².
- Réduction dimensionnelle optionnelle (SVD) et normalisation L2.

### Images
- **Pixels bruts** : `src/pipelines/image_pipeline.py` + `ImageLoader` pour aplatir et normaliser, avec PCA/SVD optionnel.
- **Statistiques** : `ImageStatsCombinedFeaturizer` calcule occupancy, entropy, gradients, colorimétrie et métriques PRO.
- **CNN embeddings** : `src/features/cnn_features.py` extrait des embeddings ResNet18/50/101 ou ViT (HuggingFace), supporte fine-tuning léger, MixUp/CutMix et Grad-CAM.

### Fusion
- `FeatureUnion` agrège les branches texte / pixels / stats / CNN.  
- Les pondérations sont définies dans `[fusion.weights]` du TOML.  
- `tools/check_pipeline_branches.py` permet de vérifier quelles branches sont actives.

---

## 7. Modélisation & entraînement

- **ModelTrainer** (`src/models/model_trainer.py`) supporte Logistic Regression, Linear SVC, XGBoost et LightGBM. Les hyperparamètres sont lus dans `[model]` du TOML.
- **Rééchantillonnage** : `src/data/sampling.py` applique un under/over-sampling ciblé (classe majoritaire plafonnée, classes minoritaires gonflées).
- **Validation croisée** : `stage04_model_training.py` propose StratifiedKFold (paramétrable `cv.*`) avec reporting F1 pondéré.
- **Sauvegardes** :
  - Modèle final : `artifacts/model_{kind}_{phase}.joblib`
  - Pipeline complet : `artifacts/{kind}_full_pipeline.joblib` (si activé)
  - Métriques comparatives : `models/compare_cv_results.csv`

---

## 8. Évaluation, monitoring & explainability

- **Rapports** (`stage05_model_evaluation.py`) :
  - Accuracy, F1 (macro/weighted), precision/recall.
  - Matrices de confusion multi-formats (valeurs, pourcentages, top erreurs).
  - Export JSON/CSV pour intégration dans dashboards (`results/metrics/`).
- **SHAP** :
  - Calcul global et décomposé par bloc (texte, CNN, stats) selon `explainability.*`.
  - `tools/shap_block_aggregation.py` agrège les contributions et peut produire un graphique `results/shap/shap_blocks.png`.
- **Monitoring continu** :
  - Rapports journaliers pour les lots `b2` et `b4` stockés dans `streamlit_app/rapport/`.
  - Scripts dédiés (ex: `streamlit_app/rapport/*`) comparent matrices actuelles vs baseline pour détecter du drift.

---

## 9. Serving & expérience utilisateur

- **Inference lite** (`streamlit_app/inference_lite.py`) : charge `Text` TF-IDF, embeddings images numpy (`data/demo_image_features.npz`), index JSON et modèle final pour répondre à des requêtes API ou tests unitaires.
- **Application Streamlit** (`streamlit_app/config.py`) :
  - Upload CSV/images, visualisation des prédictions et Top-K classes.
  - Overview dataset (distribution classes, tokens fréquents, wordcloud si `wordcloud` installé).
  - Démo images (`streamlit_app/demo_images/`) et assets logos (`streamlit_app/assets/`).
  - Lancement : `python -m streamlit run streamlit_app/config.py`.

![Flux utilisateur](streamlit_app/mermaid-flow.svg)

- **Assets additionnels** : `streamlit_app/style.css` pour la charte graphique, `member.py` pour afficher les profils équipe.

---

## 10. Structure du dépôt & artefacts

| Répertoire / fichier | Rôle principal |
| --- | --- |
| `config/` | Fichier TOML, mappings labels/thèmes/traductions |
| `docker-compose.yml` | Service Postgres autonome pour ingestion CSV |
| `etl/manifest_and_hash.py` | Calcul du hash combiné et du manifeste images |
| `import/` | CSV officiels Rakuten |
| `sql/` | Scripts de création de schéma, staging, vues, contrôles |
| `scripts/` | Entrées CLI (`train.py`, `train_pipeline.py`) |
| `src/` | Code source (data, features, pipelines, modèles, utils) |
| `streamlit_app/` | Application Streamlit, assets et rapports |
| `tools/` | Scripts auxiliaires (diagnostic pipeline, profiling, cache) |
| `reports/`, `models/`, `artifacts/`, `results/` | Sorties générées (figures, métriques, artefacts ML) |
| `snapshot.json`, `images_manifest.json` | Traçabilité des données |

---

## 11. Configuration & secrets

- **`config/config.toml`** (sections principales) :
  - `[paths]` : chemins CSV et images.
  - `[images]` + `[images.dim_reduction]` : tailles, PCA/SVD.
  - `[features.text|image]` : hyperparamètres de chaque branche.
  - `[fusion.weights]` : pondération finale dans le FeatureUnion.
  - `[model]`, `[cv]`, `[sampling]`, `[explainability]`, `[compute]`, `[debug]`.
- **Variables sensibles** :
  - `.env` (non versionné) : identifiants Kaggle, Gmail (mot de passe applicatif) utilisés par certains scripts Docker historiques.
  - API keys Streamlit (si déploiement cloud).
  - Chemins images : s’assurer qu’aucune donnée propriétaire n’est commitée.

---

## 12. Outils de diagnostic & productivité

| Commande | Description |
| --- | --- |
| `python tools/check_pipeline_branches.py --config config/config.toml` | Liste les branches actives et leurs poids dans la fusion |
| `python tools/test_pipeline_sample.py --sample-size 1500 --profile-features` | Exécute un run réduit avec profiling détaillé |
| `python tools/clear_cache.py --all` | Purge le cache Joblib des features et les modèles intermédiaires |
| `python tools/shap_block_aggregation.py --shap_values ...` | Agrège les valeurs SHAP par bloc et génère un graphique |
| `python scripts/train.py --compare-all` | Lance un entraînement expérimental (structure réduit) |

Astuce : activer les logs DEBUG (`--verbose`) pour suivre les timers (`src/utils/profiling.py`).

---

## 13. Bonnes pratiques, contribution & roadmap

- **Qualité** :
  - Toujours rejouer `init.sh` après mise à jour des CSV pour garder la cohérence du hash.
  - Utiliser `tools/clear_cache.py` lorsque les features changent profondément.
  - Surveiller attentive­ment la densité des matrices (loggée dans `ModelTrainingPipeline`) pour ajuster `StandardScaler`/PCA.
- **Contribution** :
  - Fork + PR, description claire des changements, formatage `black`/`ruff` au besoin.
  - Ajouter des tests unitaires (ex : composants `src/features/*`) avant ouverture de PR.
  - Mentionner les scripts impactés (pipeline vs Streamlit vs SQL).
- **Roadmap suggérée** :
  1. Ajout d’une API FastAPI pour exposer les prédictions via REST.
  2. Intégration de tests e2e sur un échantillon (CI) avec données synthétiques.
  3. Renforcement du monitoring temps réel (alerting drift automatique).
  4. Packaging Streamlit + inference pour déploiement Docker unique.

---

## 14. Annexes & ressources

- **Challenge Rakuten** : https://challengedata.ens.fr/participants/challenges/42/
- **Documentations** :
  - PostgreSQL COPY / indexes : `sql/` fichiers commentés.
  - scikit-learn pipelines : `src/pipelines/`.
  - PyTorch vision modèles : `src/features/cnn_features.py`.
  - Streamlit best practices : `streamlit_app/config.py`.
- **Glossaire rapide** :
  - `prdtypecode` : code catégorie cible.
  - `FeatureUnion` : combinaison pondérée de transformateurs.
  - `SHAP` : SHapley Additive exPlanations, pour l’explicabilité granulaire.

---

### Licence

Projet distribué sous licence MIT (`LICENSE`). Toute contribution implique l’acceptation des mêmes termes.
