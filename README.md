# Telecom Retention Optimize — refonte v2

Étude locale du churn à 30 jours, organisée en quatre notebooks indépendants.
Les calculs métier restent dans les notebooks ; `settings.py` centralise les
chemins. Aucun service externe, CI/CD, API ou déploiement n'est nécessaire.

## Installation

Python 3.12 et Poetry :

```bash
poetry install --with notebook --without dev
poetry run python -m ipykernel install --user --name telco-retention --display-name "Python 3.12 (Telco)"
```

Choisir ce noyau. Poetry gère les dépendances avec `package-mode = false` ;
MLflow reste fixé à 3.18.0. Les dépendances déjà présentes sont conservées.

## Parcours

| Notebook | Responsabilité |
| --- | --- |
| [00 — Exploration](notebooks/00_churn_exploration_et_profils_clients.ipynb) | Structure, qualité, clients et cohortes, relations, signaux suspects et dix KPI descriptifs |
| [01 — Feature engineering](notebooks/01_churn_feature_engineering.ipynb) | Normalisation, neuf features candidates, partitions et préparation versionnée ; aucun modèle prédictif |
| [02 — Modélisation](notebooks/02_churn_modelisation.ipynb) | Chargement des préparations, entraînement principal et diagnostics, seuil sur validation, sauvegarde MLflow ; aucune nouvelle feature |
| [03 — Évaluation et explicabilité](notebooks/03_churn_evaluation_et_explicabilite.ipynb) | Chargement du dernier principal v2 complet, métriques, SHAP, bootstrap et figures ; aucun fit |

Exécuter chaque notebook dans l'ordre de ses cellules, depuis la racine ou
`notebooks/`. Le 02 requiert les sauvegardes du 01 ; le 03 requiert un entraînement
principal du 02. Un noyau neuf convient à chaque étape. Le 00 reste descriptif
et n'est pas une dépendance d'exécution.

## Choix de la refonte

La référence conserve les variables sources connues au snapshot : contrat,
région, département, canal, mensualité, consommation, support, retards et NPS.
L'âge, les identifiants, la facture et l'intervention de rétention ne sont pas
prédicteurs. Les anciennes dérivées de facture, produits arbitraires, croisements
systématiques et seuils non justifiés ne sont pas repris dans cette référence.

Les neuf candidates sont :

- `tenure_months` : ancienneté à la date du snapshot ;
- `usage_change_vs_recent_pct` : consommation relative à la médiane des trois
  mois précédents, avec historique complet et référence positive ;
- `zero_usage_streak` : mois consécutifs d'usage nul ;
- `support_pressure_change` : variation nette du compteur glissant de support ;
- `support_presence_3_snapshots` : persistance du support, sans sommer des
  fenêtres de trois mois qui se chevauchent ;
- `last_observed_nps` et `months_since_observed_nps` : dernière satisfaction
  renseignée et fraîcheur de cette présence dans le fichier ;
- `monthly_fee_change_pct` : variation du prix entre mois consécutifs ;
- `monthly_fee_band` : tranches relatives au train, avec catégorie de manquants.

Une hypothèse métier ne démontre pas un apport prédictif. La période et les
unités de consommation, les variations presque systématiques de tarif et la
date réelle des notes de satisfaction restent à clarifier.

## Une seule classe pour les transformations

```python
feature_engineer = ChurnFeatureEngineer()
feature_engineer.fit(train_source)
featured = feature_engineer.transform(context)
```

La classe principale orchestre les petites classes. Seules les bornes de tarif
sont apprises ; son historique ne mémorise pas les labels. Les calculs temporels
sont par client, triés chronologiquement, et ne prennent aucune observation
future. Les deltas et fenêtres exigent les mois consécutifs appropriés. Les
premières lignes et historiques incomplets restent manquants. Le NPS peut être
reporté depuis un snapshot antérieur avec son ancienneté, sans remplissage depuis
le futur. Les tarifs précédents nuls ne servent pas de dénominateur.

Le 01 ajuste les bornes sur janvier–juillet, puis transforme le contexte connu.
Chaque diagnostic a ses propres bornes ajustées sur son train ; des covariables
connues de mois purgés peuvent servir d'historique sans utiliser leurs labels.
Les conversions de `make_matrix` dans le 02 sont uniquement techniques.

## Configuration et fichiers

`settings.py` expose un objet Pydantic immuable. `CHURN_DATA_PATH` peut remplacer
le CSV par défaut ; un chemin relatif part de la racine du projet. Redémarrer
le noyau après un changement d'environnement.

- `data/raw/` : sources conservées et jamais modifiées par les notebooks ;
- `data/processed/` : trois CSV principaux enrichis, sans index ;
- `data/interim/features/feature_contract.json` : préparation v2, colonnes,
  métadonnées, fichiers et SHA-256 ;
- `data/interim/features/diagnostics/` : entrées des diagnostics par client et
  des folds temporels ;
- `data/interim/features/reports/` : qualité des features et résumé des partitions ;
- `data/interim/features/source/` : copie du notebook de feature engineering ;
- `.mlflow/` : SQLite et artefacts locaux, ignorés par Git ;
- `data/output/` : réservé au futur fichier de scoring final ;
- `note/` : énoncé et références.

Relancer le 01 remplace les préparations. Leur contrat est la référence de lecture ;
le 02 vérifie les empreintes avant utilisation. Les runs MLflow archivent ensuite
leurs propres données et ne dépendent plus des CSV courants.

## MLflow et traçabilité

Les imports, paramètres, données et helpers sont regroupés par section. Les blocs
`with mlflow.start_run(...)` encadrent réellement `fit`, le choix du seuil et les
sauvegardes. Un échec empêche le run d'être chargé automatiquement par le 03.

Chaque modèle conserve paramètres demandés et effectifs, données référencées par
`log_input`, instantanés CSV, code de features, code de modélisation, cellule
exécutée, configurations, environnement, historique complet par itération et
modèle CatBoost natif avec signature et exemple. Le principal conserve aussi son
seuil choisi sur validation et le test en simple référence archivée. Le notebook
sur disque est copié sans ses sorties ; le sauvegarder avant fit permet de figer
toutes les définitions amont.

Les modèles de diagnostic sont séparés du principal. Le 03 filtre les runs
`FINISHED`, `stage=training`, `role=principal`, `contract_version=2`,
`artifacts_complete=true`, par date de fin décroissante. Les anciens runs v1
restent conservés mais ne sont pas sélectionnés par cette refonte. Le run chargé
et son seuil sont figés pour toute la session d'évaluation.

Le 03 loggue ses scores, tables, prédictions test, code et onze figures dans un
run d'évaluation lié à l'entraînement. Les tracés couvrent calibration et
probabilités, compromis de seuil, gains/lift, importance native, quatre vues SHAP,
confusion/ROC, précision-rappel et bootstrap par client. L'additivité SHAP est
vérifiée contre le score brut ; les contributions s'expriment en log-odds.

Interface locale facultative, depuis la racine :

```bash
poetry run mlflow ui --backend-store-uri "sqlite:///$PWD/.mlflow/mlflow.db" --host 127.0.0.1 --port 5000
```

Ouvrir http://127.0.0.1:5000. Les notebooks utilisent directement les fichiers
locaux ; l'interface n'est pas nécessaire à leur exécution. La télémétrie MLflow
est désactivée.

## Validation et limites

Train : janvier–juillet 2024 ; validation : août–septembre ; test : octobre–novembre.
Toute l'année est explorée dans le 00 : le test reste rétrospectif. La validation
sert au seuil et à l'early stopping ; ses scores de sélection sont optimistes.
Les diagnostics temporels conservateurs entraînent jusqu'à M−2 pour le mois M.
Les répétitions d'un client sont dépendantes ; le bootstrap rééchantillonne les
clients, sans réentraînement.

Les copies normalisées sont dédupliquées ; toutes les lignes des clés ambiguës
sont exclues. Les montants et compteurs invalides sont neutralisés dans la copie
de travail. Les exclusions et domaines sont visibles, sans modification des sources.

Le ciblage mensuel à 300 contacts, la rentabilité, le pilote et le scoring final
restent des étapes distinctes à réaliser. Ni l'importance native ni SHAP ne
prouvent un effet causal.

## Vérifications locales

Les outils de vérification sont dans le groupe Poetry `dev` ; aucune CI n'est
ajoutée. Les annotations pandas permettent à Pyright d'analyser son API publique.
Le script extrait les cellules dans leur ordre et vérifie leur syntaxe et leurs
types sans exécuter les notebooks :

```bash
poetry install --with notebook,dev
poetry run pytest -q
poetry run python scripts/check_notebooks.py
```

Les tests de régression couvrent les fenêtres métier, les mois absents, les
zéros, la fraîcheur du NPS, l'absence d'information future, l'isolation des
clients, la préservation des données et la gestion des clés ambiguës.
Les compatibilités d'annotations spécifiques à MLflow et scikit-learn sont
signalées ponctuellement dans le code, sans désactiver les contrôles globaux.
