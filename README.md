# Telecom Retention Optimize

Deux notebooks indépendants :

- [Sélection de variables et validation groupée](notebooks/telco_churn_feature_selection_grouped_cv.ipynb) : version simplifiée, sans création de features.
- [Version temporelle](notebooks/telco_churn_modeling.ipynb) : version historique conservée.

La version avec feature engineering reste dans [notebooks/old/](notebooks/old/telco_churn_modeling_with_feature_engineering.ipynb).

## Installation

Python 3.12 et Poetry :

```bash
poetry install --with notebook,dev
```

Choisir le noyau du projet et ouvrir le notebook depuis la racine ou `notebooks/`.
`settings.py` centralise les chemins. `CHURN_DATA_PATH` permet de remplacer le CSV
`data/raw/churn_clients.csv` ; les chemins relatifs partent de la racine.

## Utiliser le notebook groupé

La première partie regroupe les imports et les fonctions. La seconde charge les
données, présente les analyses, entraîne et affiche les résultats.

1. Examiner les variables catégorielles et numériques sur le train du premier fold.
2. Remplacer `SELECTED_FEATURES = None` par la liste des variables à conserver.
3. Exécuter la comparaison sur les mêmes folds, puis le bilan sur le test final.

Avec `None`, seule la référence est entraînée : le notebook ne bloque pas et
n’invente pas de sélection. Une liste ajoute un second modèle utilisant exactement
ces colonnes. Aucune permutation, ablation ou réintégration automatique n’est faite.

Les huit prédicteurs disponibles sont `contract`, `channel`, `monthly_fee`,
`data_gb`, `support_calls_3m`, `late_payments`, `nps`, `last_invoice_amount`.
L’identifiant sert uniquement aux groupes ; l’âge, les dates, la géographie et
`retention_offer_declined` ne sont pas envoyés au modèle.

## Validation

Environ 20 % des clients sont réservés au test. Les autres passent dans quatre folds
`StratifiedGroupKFold`, stratifiés sur `churn` et groupés par `customer_id`.
Toutes les observations d’un client restent ensemble dans chaque découpage.
Cette validation mesure la généralisation à d’autres clients, sans garantie temporelle.

Chaque CatBoost effectue **500 itérations**, avec des poids calculés sur son train.
Il n’y a ni early stopping ni optimisation du seuil : le seuil reste **0,5**.
Dans les folds, la validation sert au suivi des courbes et aux scores, sans choisir
le nombre d’arbres. Le modèle final apprend sur tout le développement ; le test
n’est jamais transmis à `fit`.

Les scores incluent MCC, Average Precision, ROC-AUC, Logloss, F1, précision et rappel.
Les matrices Matplotlib présentent validation CV et test côte à côte, avec effectifs
et pourcentages par classe réelle. Les courbes viennent du premier fold ;
l’importance décroissante vient du modèle final et ne prouve pas la causalité.
Ne pas ajuster les choix de variables à partir des résultats du test.

Pour vérifier les contrôles et les types :

```bash
poetry run pytest tests/test_grouped_cv_copy.py -q
poetry run python scripts/check_notebooks.py
```

Les sorties d’exécution sont conservées dans le notebook. Aucun CSV source n’est
modifié. Aucune dépendance ni CI/CD n’est ajoutée.
