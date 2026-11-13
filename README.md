# Telco Retention Optimize

Projet exploratoire de churn télécom à 30 jours, organisé en **quatre notebooks**.
Les fonctions restent dans les notebooks. MLflow local conserve les modèles et
leur traçabilité ; aucune CI/CD, API ou mise en production n'est ajoutée.

## Installation

Python 3.12 et Poetry, depuis la racine :

```bash
poetry install --with notebook --without dev
poetry run python -m ipykernel install --user --name telco-retention --display-name "Python 3.12 (Telco)"
```

Choisir ce noyau dans l'éditeur. Poetry gère uniquement les dépendances avec
`package-mode = false`. MLflow est fixé à 3.18.0. Les dépendances présentes sont
conservées pour les analyses à venir.

## Parcours : 00 → 01 → 02 → 03

| Notebook | Responsabilité | Sortie |
| --- | --- | --- |
| [00 — Exploration et profils clients](notebooks/00_churn_exploration_et_profils_clients.ipynb) | Composition, qualité, panel client × mois, relations, KPI et aperçu des tranches tarifaires | Analyses et graphiques uniquement |
| [01 — Feature engineering](notebooks/01_churn_feature_engineering.ipynb) | Lecture normalisée, transformations, colonnes retenues, partitions et sauvegarde ; aucun modèle prédictif | CSV enrichis, contrat des features et rapports |
| [02 — Modélisation](notebooks/02_churn_modelisation.ipynb) | Chargement des données préparées, entraînement principal, choix du seuil sur validation, diagnostics et suivi MLflow ; aucune nouvelle feature | Modèles et runs MLflow |
| [03 — Évaluation et explicabilité](notebooks/03_churn_evaluation_et_explicabilite.ipynb) | Chargement du dernier principal MLflow, évaluation et SHAP ; aucun entraînement | Métriques, figures, bootstrap et run d'évaluation |

Exécuter les cellules dans l'ordre, depuis la racine ou `notebooks/`. Chaque
notebook fonctionne dans un noyau neuf avec les sauvegardes de l'étape précédente.
Le 00 est descriptif et n'est pas une dépendance d'exécution du 01.

## Configuration et dossiers

`settings.py` expose la configuration Pydantic :

```python
from settings import settings

settings.DATA_PATH
settings.PROCESSED_DIR
settings.OUTPUT_DIR
settings.MLFLOW_TRACKING_URI
```

`CHURN_DATA_PATH` peut remplacer `data/raw/churn_clients.csv` avant l'import.
Un chemin relatif part de la racine du projet. Redémarrer le noyau après un
changement de variable d'environnement.

- `data/raw/` : source churn et pilote 2023, inchangés.
- `data/processed/` : uniquement `train.csv`, `validation.csv` et `test.csv`,
  enrichis et sans index.
- `data/interim/features/feature_contract.json` : colonnes, catégories, bornes,
  chemins et SHA-256 des données produites par le 01.
- `data/interim/features/reports/` : contrôles de préparation.
- `data/interim/features/source/` : copie du code de feature engineering sans sorties.
- `data/interim/features/diagnostics/` : entrées préparées pour les diagnostics
  par client et les folds temporels, sans aucun modèle.
- `data/output/` : réservé au futur fichier de prédictions final.
- `.mlflow/` : base SQLite et artefacts locaux, ignorés par Git.
- `note/` : énoncé et références ; `logs/` : réservé au logger à intégrer.

Relancer le 01 remplace les fichiers de préparation. Le 02 vérifie les SHA-256
avant d'entraîner. Chaque run MLflow archive ensuite ses propres copies : une
préparation ultérieure ne change pas les données d'un ancien modèle.

## Feature engineering : un point d'entrée unique

Les petites classes sont internes à `ChurnFeatureEngineer`, défini dans le 01 :

```python
feature_engineer = ChurnFeatureEngineer()
feature_engineer.fit(feature_train)
featured = feature_engineer.transform(labeled)
```

`fit_transform(train_frame)` combine l'ajustement et la transformation du train.
Les 22 nouvelles variables incluent `monthly_fee_band`, dont les tertiles sont
ajustés sur les mensualités valides du train janvier–juillet uniquement. Les
mêmes bornes s'appliquent ensuite à validation et test ; les montants absents ou
invalides ont une catégorie « Non renseigné ». Le montant numérique est conservé.
Les copies exactes sont supprimées ; les clés client × snapshot ambiguës sont
exclues intégralement, sans arbitrage.

Pour les diagnostics, le 01 prépare aussi des partitions avec leur propre
instance de la classe, ajustée seulement sur leur train. Le 02 lit ces résultats
et les métadonnées ; il ne recopie pas les classes et ne recalcule pas les bornes.
`make_matrix` n'y fait que des conversions techniques en chaînes/flottants.

## Entraînement et MLflow

Dans le 02, les imports et paramètres sont au début, les helpers de suivi dans
une section dédiée, puis un bloc `with mlflow.start_run(...)` encadre réellement
`fit`, le choix du seuil et la sauvegarde. Chaque diagnostic a son propre run.
Une erreur dans le bloc marque le run `FAILED`.

MLflow conserve les paramètres, la durée de fit, les données référencées via
`log_input`, les instantanés CSV, les SHA-256, les courbes par itération, le seuil,
le modèle CatBoost natif avec signature et exemple, et les configurations.
`source/notebook_02.ipynb` et `.py` contiennent le code de modélisation ;
`source/training_cell.py` conserve la cellule exécutée dans Jupyter, ou celle du
fichier lors d'une exécution hors Jupyter. `source/features/` conserve le code et
le contrat du 01 associés à ces données. Sauvegarder le notebook avant l'exécution
permet de figer toutes les définitions amont.

Les métriques sont publiées pendant fit puis complétées sans doublons pour toutes
les itérations. Elles restent consultables dans `learning/history.csv`,
`learning/history.json` et `learning/curves.html`. Le 02 conserve la baseline,
le diagnostic clients jamais vus, le walk-forward et les ablations de facture.

Le 03 sélectionne le **dernier run principal terminé avec succès**, par date de
fin, avec `stage=training`, `role=principal`, `contract_version=1`. Il ne choisit
ni le meilleur score ni le dernier diagnostic. Le run et le seuil sont figés
pour la session ; les CSV proviennent de ses artefacts vérifiés. Aucun fit de
secours ni recalcul du seuil n'est effectué.

L'évaluation conserve onze figures : calibration/distribution, seuil/volumes,
gains/lift, importance native, quatre vues SHAP, confusion/ROC, précision-rappel
et bootstrap par client. Les tables, scores et prédictions test sont loggués dans
un run d'évaluation lié au principal.

Interface locale facultative, depuis la racine :

```bash
poetry run mlflow ui --backend-store-uri "sqlite:///$PWD/.mlflow/mlflow.db" --host 127.0.0.1 --port 5000
```

Ouvrir http://127.0.0.1:5000. Aucun serveur permanent n'est requis par les
notebooks. La télémétrie MLflow est désactivée.

## Limites d'interprétation

Le split principal est rétrospectif : janvier–juillet 2024 pour train,
août–septembre pour validation, octobre–novembre pour test. Toute l'année est
explorée dans le 00 ; ce test n'est pas aveugle. La validation sert à l'early
stopping et au seuil ; le test est uniquement archivé dans le 02, puis évalué
dans le 03 sans ajustement. Les diagnostics utilisent une validation temporelle,
pas des prédictions OOF.

Un client peut contribuer plusieurs snapshots, qui ne sont pas indépendants.
L'âge et les identifiants sont exclus des prédicteurs. La facture et ses dérivées
peuvent révéler la cible ; leur temporalité doit être confirmée. L'ablation sans
facture reste disponible. SHAP décrit les contributions au score en log-odds,
sans établir de causalité ; son additivité est vérifiée.

Les figures Matplotlib des 00 et 03 utilisent un affichage SVG à largeur
adaptable ; les tailles restent ajustables. Le ciblage à 300 contacts, le gain
économique, le pilote et le scoring final restent à réaliser. Les données et
travaux restent locaux. Les diagnostics de types des bibliothèques ne sont pas
encore tous résolus ; Pyright CLI nécessite l'extraction des cellules en Python.

## Variation mensuelle du tarif

Le 01 crée `monthly_fee_change_pct = 100 × (tarif courant − tarif précédent) /
tarif précédent` via une sous-transformation interne à `ChurnFeatureEngineer`.
Le calcul compare le même client sur deux mois consécutifs. Sans historique,
avec un mois absent, un montant invalide ou un tarif précédent nul, la valeur
reste manquante. Un prix futur ne modifie jamais une observation antérieure.
Les prix historiques de mois exclus du train pour purge des labels peuvent
servir de contexte, sans utiliser leurs labels. Le 02 lit la colonne déjà créée.
`invoice_fee_ratio` reste dans les CSV pour audit mais est exclue des prédicteurs ;
`invoice_fee_gap` est conservée.
