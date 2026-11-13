# Analyse approfondie du test technique Data Scientist — Churn Prediction & Retention Optimization

Perspective : Senior Data Scientist · ML Engineer · MLOps · Business Decision Science

Ce test technique est beaucoup plus intéressant qu'un simple exercice de classification binaire. Il évalue ta capacité à concevoir un système de décision métier fondé sur le Machine Learning, à en mesurer la valeur économique et à expliquer comment tu pourrais l'industrialiser.

Le piège serait de construire un modèle XGBoost, d'obtenir un ROC-AUC de 0,90, de présenter une matrice de confusion et de considérer le problème comme résolu.

Ce n'est pas ce que recherche réellement l'entreprise.

Elle veut savoir si tu es capable de répondre à cette question :

> Parmi 1,2 million de clients, quels sont les 300 clients que nous devrions contacter chaque mois pour maximiser le bénéfice économique de notre campagne de rétention, tout en respectant les contraintes légales et opérationnelles ?

Pour répondre correctement, il faut articuler six disciplines : Data Engineering, Machine Learning, Decision Science, Causal Inference, statistiques et MLOps.

## 1. Comprendre le véritable problème métier

Architecture conceptuelle de la solution

Données clients mensuelles

Historique, facturation, contrat, comportement

Prédiction

Probabilité de churn

Économie

Espérance de gain par contact

Optimisation de campagne

Classement par gain attendu, maximum 300 contacts

Mesure des résultats

Churn évité, rentabilité, incertitude, monitoring

Il faut distinguer trois objectifs.

| Question                                | Discipline                | Résultat                     |
| --------------------------------------- | ------------------------- | ---------------------------- |
| Qui risque de résilier ?                | Predictive ML             | \\(P(churn=1\mid X)\\)       |
| Qui bénéficie réellement d'un contact ? | Causal inference / uplift | Effet incrémental du contact |
| Qui contacter en priorité ?             | Decision optimization     | Gain net attendu             |

Ces questions ne sont pas équivalentes.

Un client avec 90 % de risque de churn peut être impossible à retenir. Un autre avec 50 % de risque peut être très réceptif à une offre.

Le meilleur client à contacter n'est donc pas nécessairement celui qui présente le plus fort risque de résiliation.

C'est une distinction fondamentale à expliquer pendant ton entretien.

## 2. Comprendre précisément les deux datasets

### 2.1 `churn_clients.csv` — apprendre à prédire

Ce fichier constitue la source principale pour construire le modèle de classification.

L'unité statistique est :

\\[ \boxed{\text{Une observation}=\text{un client à une fin de mois}} \\]

Un même client peut apparaître à plusieurs dates.

Par exemple :

| customer_id | snapshot_date | monthly_fee | support_calls_3m | churn |
| ----------- | ------------- | ----------- | ---------------- | ----- |
| 100127      | 2024-01-31    | 48 €        | 0                | 0     |
| 100127      | 2024-02-29    | 48 €        | 1                | 0     |
| 100127      | 2024-03-31    | 48 €        | 4                | 1     |

Exemple fictif illustrant la structure temporelle, et non trois lignes observées dans ton fichier.

Le client est présent en janvier, février et mars. À la fin de mars, il résilie dans les 30 jours suivants.

Cela implique deux choses essentielles :

- Les observations ne sont pas indépendantes : plusieurs lignes peuvent concerner le même client.
- L'ordre temporel est important : le modèle doit apprendre à partir du passé pour prédire le futur.

### 2.2 Analyse des variables

| Variable                   | Utilisation recommandée          | Attention                               |
| -------------------------- | -------------------------------- | --------------------------------------- |
| `customer_id`              | Identifiant de jointure et suivi | Ne pas l'utiliser comme prédicteur brut |
| `snapshot_date`            | Découpage temporel               | Disponibilité des labels                |
| `signup_date`              | Calculer l'ancienneté            | Ne pas utiliser de dates futures        |
| `age`                      | Audit légal uniquement           | Exclure du ciblage                      |
| `contract`                 | Prédicteur catégoriel            | Types mensuel, annuel, biannuel         |
| `region`, `dept`           | Variables géographiques          | Qualité et proxies possibles            |
| `channel`                  | Canal d'acquisition              | Catégorie à encoder                     |
| `monthly_fee`              | Prédiction et gain économique    | Valeurs manquantes                      |
| `data_gb`                  | Usage du service                 | Valeurs extrêmes                        |
| `support_calls_3m`         | Insatisfaction potentielle       | Fenêtre historique correcte             |
| `late_payments`            | Comportement de paiement         | Définition exacte à vérifier            |
| `nps`                      | Satisfaction client              | Missingness potentiellement informative |
| `last_invoice_amount`      | Facturation                      | Risque de fuite temporelle              |
| `retention_offer_declined` | Historique de rétention          | Date réelle du refus à vérifier         |
| `churn`                    | Variable cible                   | Label disponible à J+30                 |

Il y a plusieurs pièges intéressants.

Premier piège : `last_invoice_amount`. Si cette information correspond à une facture produite après la date de scoring, elle crée une fuite de données. Il faut vérifier son horodatage métier.

Deuxième piège : `retention_offer_declined`. Si le refus est intervenu avant le snapshot, la variable peut être exploitable. S'il s'agit d'un refus lié à une campagne future, elle doit être exclue.

Troisième piège : `dept`. Dans ton extrait, `dept` ressemble à un code à deux caractères. Il ne faut pas supposer automatiquement qu'il s'agit d'un département administratif français cohérent avec `region`. Il faut consulter le dictionnaire de données ou contrôler les correspondances.

Quatrième piège : l'ancienneté client. Il est préférable de créer :

\\[ tenure\\\_months \approx \frac{snapshot\\\_date-signup\\\_date}{30.44} \\]

En production, utilise plutôt un calcul calendaire explicite, stable et documenté.

### 2.3 `pilote_campagne_2023.csv` — comprendre l'effet du contact

Ce deuxième fichier ne sert pas simplement à améliorer la prédiction.

Il sert surtout à examiner si contacter un client change réellement son comportement.

Ses variables importantes sont :

- `contacted` : traitement, 1 si contacté, 0 sinon.
- `churn_30d` : résultat observé après 30 jours.
- Les autres colonnes : caractéristiques permettant de comparer les groupes.

On peut poser :

\\[ T_i=contacted_i \\]

\\[ Y_i=churn\\\_30d_i \\]

Le pilote permet de calculer :

\\[ P(Y=1\mid T=1) \quad\text{et}\quad P(Y=1\mid T=0) \\]

Mais attention : la différence entre ces taux n'est pas automatiquement l'effet causal du contact, puisque le mode de sélection des clients contactés n'est pas documenté.

Il pourrait y avoir un biais de sélection important.

Par exemple, si les conseillers ont surtout contacté les clients les plus susceptibles de partir, le groupe contacté peut avoir un taux de churn supérieur au groupe non contacté, même si la campagne est bénéfique.

C'est l'un des principaux pièges du test.

## 3. Construire une stratégie d'évaluation temporelle correcte

Le sujet précise que les labels sont connus seulement 30 jours après le snapshot.

Cela signifie qu'au moment où tu scores les clients au 31 octobre, tu ne connais pas encore leur churn à 30 jours.

Un `train_test_split` aléatoire serait donc une mauvaise stratégie d'évaluation opérationnelle : il mélange les périodes et peut exploiter indirectement des informations futures.

Je recommande un backtesting temporel avec une fenêtre de disponibilité des labels.

Exemple de simulation d'un scoring au 31 octobre 2024

Janvier – août 2024

Entraînement

Septembre 2024

Non disponible pour l'entraînement

31 octobre 2024

Scoring

Novembre 2024

Observation du churn

Schéma conservateur : l'entraînement s'arrête aux labels dont la disponibilité est garantie au moment du scoring.

La disponibilité exacte des labels de septembre dépend de leur horodatage et de la convention de clôture. Elle doit être vérifiée, pas supposée.

### Mes recommandations pour les métriques

Je n'utiliserais pas l'accuracy comme métrique principale. Dans un problème de churn déséquilibré, un modèle peut obtenir une accuracy élevée en prédisant presque toujours « pas de churn ».

Je privilégierais :

| Métrique                | Pourquoi                                                                |
| ----------------------- | ----------------------------------------------------------------------- |
| PR-AUC                  | Évaluer la capacité à retrouver les churners lorsque la classe est rare |
| ROC-AUC                 | Mesurer la discrimination globale                                       |
| Brier Score             | Mesurer la qualité des probabilités                                     |
| Calibration curve       | Vérifier que 30 % de risque signifie environ 30 % de churn observé      |
| Precision\@300          | Évaluer la qualité des 300 clients sélectionnés                         |
| Expected Net Value\@300 | Mesurer directement la valeur métier                                    |

La métrique la plus proche de la décision finale est Expected Net Value\@300, calculée avec les probabilités calibrées et les hypothèses économiques.

Mais il faudra bien la présenter comme une estimation fondée sur des hypothèses, et non comme un gain causal déjà démontré.

## 4. Le cœur du test : transformer la prédiction en décision économique

C'est probablement la partie sur laquelle je consacrerais le plus d'efforts.

L'entreprise ne cherche pas simplement une probabilité de churn. Elle souhaite une règle de décision rentable.

Nous allons donc construire mathématiquement la fonction de gain attendu.

### 4.1 Comprendre l'économie d'un contact

On définit :

- \\(p\\) : probabilité estimée que le client résilie.
- \\(f\\) : montant de son abonnement mensuel (`monthly_fee`).
- \\(q_1=0.30\\) : probabilité qu'un client qui allait partir accepte l'offre.
- \\(q_0=0.10\\) : probabilité qu'un client qui ne serait pas parti accepte l'offre.
- \\(C=10\\) € : coût de contact.

Un client qui allait partir et qui accepte l'offre génère une marge conservée :

\\[ 12\times 0.40\times f=4.8f \\]

La remise coûte :

\\[ 3\times 0.30\times f=0.9f \\]

Le gain net lié à l'acceptation d'un client qui allait partir est donc :

\\[ 4.8f-0.9f=3.9f \\]

Mais un client qui ne serait pas parti peut également accepter l'offre. Dans ce cas, on lui accorde inutilement une remise : c'est l'effet d'aubaine.

### 4.2 Les scénarios possibles

| Situation                    | Probabilité     | Gain hors coût de contact |
| ---------------------------- | --------------- | ------------------------- |
| Allait partir, accepte       | \\(0.30p\\)     | \\(+3.9f\\)               |
| Allait partir, refuse        | \\(0.70p\\)     | 0 €                       |
| N'allait pas partir, accepte | \\(0.10(1-p)\\) | \\(-0.9f\\)               |
| N'allait pas partir, refuse  | \\(0.90(1-p)\\) | 0 €                       |

Le coût de contact de 10 € s'applique dans tous les scénarios.

L'espérance de gain par client est donc :

\\[ EV(p,f)=0.30p(3.9f)-0.10(1-p)(0.9f)-10 \\]

En développant :

\\[ \boxed{EV(p,f)=1.26pf-0.09f-10} \\]

C'est une formule fondamentale de ton projet.

Elle suppose que l'acceptation d'un client qui allait partir entraîne effectivement sa rétention pendant 12 mois et que la marge indiquée représente bien une valeur incrémentale pertinente. Ces hypothèses devront être signalées.

### 4.3 Calculateur interactif du gain attendu

Modifie la probabilité de churn et l'abonnement pour voir comment la décision économique évolue.

Probabilité de churn

## 70 %

Abonnement mensuel

## 65 €

Gain net espéré par contact

# +41,48 €

Seuil de rentabilité

# 19,4 %

Contact économiquement favorable selon les hypothèses.

Ce calcul est une espérance théorique, pas un effet de campagne mesuré. Il ne tient pas compte d'éventuels coûts supplémentaires ou de l'incertitude sur les taux d'acceptation.

### 4.4 Pourquoi il ne faut pas simplement contacter les 300 plus gros scores de churn

Imaginons trois clients :

| Client | Risque de churn | Abonnement | Gain attendu |
| ------ | --------------- | ---------- | ------------ |
| A      | 90 %            | 30 €       | 21,32 €      |
| B      | 70 %            | 100 €      | 69,20 €      |
| C      | 40 %            | 80 €       | 23,12 €      |

Le classement par risque de churn est :

A → B → C.

Mais le classement par gain économique est :

B → C → A.

Le client B est plus intéressant économiquement que le client A, même si son risque de résiliation est inférieur.

Cela montre pourquoi il faut distinguer la prédiction du risque et l'optimisation de la décision.

### 4.5 La règle métier que je proposerais

Pour chaque client éligible :

1. Calculer sa probabilité de churn calibrée.
2. Calculer son gain net espéré.
3. Exclure les clients dont le gain est négatif ou nul.
4. Classer les autres par gain décroissant.
5. Sélectionner au maximum les 300 premiers.

La règle est :

\\[ \max\_{z_i\in\\{0,1\\}}\sum_i z_iEV_i \\]

sous les contraintes :

\\[ \sum_i z_i\leq 300 \\]

\\[ z_i=0 \quad\text{si le client est inéligible} \\]

Ici, \\(z_i=1\\) signifie que le client est sélectionné pour être contacté.

Comme chaque contact consomme une unité de capacité, il suffit de trier les gains attendus et de prendre les 300 meilleurs gains strictement positifs.

Une implémentation Python simple :

```
import numpy as npimport pandas as pddef expected_net_value(    churn_probability: np.ndarray,    monthly_fee: np.ndarray,) -> np.ndarray:    p = np.asarray(churn_probability, dtype=float)    fee = np.asarray(monthly_fee, dtype=float)    return 1.26 * p * fee - 0.09 * fee - 10.0def select_campaign(    customers: pd.DataFrame,    capacity: int = 300,) -> pd.DataFrame:    required = ["customer_id", "churn_probability", "monthly_fee"]    candidates = customers.dropna(subset=required).copy()    candidates["expected_net_value"] = expected_net_value(        candidates["churn_probability"].to_numpy(),        candidates["monthly_fee"].to_numpy(),    )    return (        candidates.loc[candidates["expected_net_value"] > 0]        .sort_values(            ["expected_net_value", "customer_id"],            ascending=[False, True],        )        .head(capacity)        .reset_index(drop=True)    )
```

Ce code suppose que les probabilités sont déjà calibrées, que les clients sont uniques pour le mois considéré et que les règles d'éligibilité ont été appliquées.

Il faudra aussi prévoir les contraintes réelles : consentement, fréquence maximale de sollicitation, offres déjà actives et clients précédemment contactés.

## 5. Comment estimer le gain annuel avec son incertitude ?

La direction demande un chiffrage annuel.

La capacité maximale est de :

Contacts par mois

# 300

Contacts par an

# 3 600

Maximum théorique

Le gain annuel attendu est :

\\[ \widehat G\_{\text{annuel}}= \sum\_{m=1}^{12} \sum\_{i\in S_m}EV\_{i,m} \\]

où \\(S_m\\) est l'ensemble des clients sélectionnés au mois \\(m\\).

Attention : 3 600 contacts ne signifient pas nécessairement 3 600 clients distincts, car un même client peut être sollicité plusieurs fois si aucune règle ne l'interdit.

### Pourquoi une simple multiplication par 12 est insuffisante

Le dataset principal couvre 2024. Tu peux simuler les campagnes sur les mois évaluables de 2024, mais tu ne peux pas affirmer que le résultat obtenu représente directement le bénéfice annuel futur du parc de 1,2 million de clients.

Il faut tenir compte de :

- La différence entre le parc observé et le parc réel.
- La saisonnalité du churn.
- La variabilité des abonnements.
- L'incertitude sur les probabilités prédites.
- L'incertitude sur les taux d'acceptation de 30 % et 10 %.
- La variabilité des effets réels de la campagne.

Je recommanderais deux niveaux d'estimation.

Niveau 1 — Estimation économique conditionnelle. Calculer les gains attendus sur les campagnes simulées et les annualiser en explicitant les hypothèses de stabilité et de représentativité.

Niveau 2 — Analyse d'incertitude. Utiliser une simulation Monte Carlo avec des distributions justifiées pour les taux d'acceptation, ainsi qu'un bootstrap temporel ou par client adapté à la dépendance des observations.

Exemple purement illustratif

Supposons un gain net espéré moyen de 25 € par contact et 300 contacts rentables chaque mois.

# 90 000 € / an

\\[ 25\times300\times12=90\\,000 \\]

Ce chiffre n'est pas un résultat calculé sur les fichiers fournis. Il illustre uniquement la méthode de projection.

Pour une présentation rigoureuse, tu pourrais communiquer une estimation centrale, un intervalle d'incertitude à 95 % et la probabilité que le gain soit négatif.

Il faut distinguer l'incertitude de modèle, l'incertitude sur les hypothèses économiques et l'incertitude causale. Un bootstrap ne résout pas à lui seul les trois.

## 6. La campagne pilote de 2023 : le piège de la causalité

C'est à mon avis l'une des parties les plus discriminantes du test.

Le fichier `pilote_campagne_2023.csv` contient 4 000 clients, dont certains ont été contactés et d'autres non.

La question est : est-ce que le contact a réellement permis d'éviter des résiliations ?

### 6.1 Corrélation et causalité

Imaginons que tu calcules les résultats suivants :

| Groupe        | Nombre de clients | Taux de churn observé |
| ------------- | ----------------- | --------------------- |
| Contactés     | 2 000             | 12 %                  |
| Non contactés | 2 000             | 8 %                   |

Chiffres fictifs, uniquement pour expliquer le raisonnement.

Tu pourrais conclure que la campagne augmente le churn de quatre points.

Ce serait potentiellement faux.

Pourquoi ? Parce que les conseillers ont peut-être sélectionné les clients présentant déjà un risque élevé.

Le problème du biais de sélection

Le contact n'a pas été attribué de manière aléatoire documentée. Il est donc impossible d'interpréter directement la différence brute de churn comme un effet causal.

### 6.2 Ce que je ferais avec le pilote

Je procéderais en quatre étapes.

1. Audit des groupes : comparer les caractéristiques des clients contactés et non contactés (contrat, ancienneté, facturation, incidents de paiement, etc.).
2. Mesure du déséquilibre : calculer les Standardized Mean Differences (SMD) pour identifier les différences de profils.
3. Estimation exploratoire ajustée : envisager propensity score weighting, régression ajustée ou doubly robust estimation.
4. Analyse des limites : vérifier le recouvrement des groupes, discuter les confondeurs non observés et présenter une analyse de sensibilité.

Même après ajustement, l'interprétation causale repose sur des hypothèses fortes, notamment l'absence de confondeurs non observés.

Je ne prétendrais donc pas avoir démontré l'efficacité de la campagne avec ce seul pilote.

### 6.3 Pourquoi un modèle d'uplift serait une évolution pertinente

Le modèle de churn prédit :

\\[ P(Y=1\mid X) \\]

Un modèle d'uplift cherche plutôt à estimer :

\\[ \tau(X)=P(Y=1\mid X,T=0)-P(Y=1\mid X,T=1) \\]

Autrement dit, la réduction de probabilité de churn attribuable au contact.

C'est précisément l'information qu'on aimerait exploiter pour une campagne de rétention.

Mais il faut disposer de données permettant une identification causale crédible.

Ma recommandation : proposer une expérimentation randomisée pour les futures campagnes, avec un groupe de contrôle, puis comparer la politique fondée sur le risque de churn à une politique fondée sur l'uplift économique.

Il faudra également mesurer l'acceptation réelle de l'offre, sa durée d'effet, la marge effectivement conservée et les éventuelles résiliations différées.

## 7. La contrainte légale : exclure l'âge ne suffit pas

Le service juridique interdit l'utilisation directe de l'âge dans la décision de ciblage.

Je retirerais donc `age` des variables utilisées par le modèle opérationnel, et je m'assurerais qu'il n'intervient pas non plus dans les règles de sélection ou dans le calcul du gain.

Mais il existe une subtilité : d'autres variables peuvent être corrélées à l'âge.

Par exemple, l'ancienneté client, le canal d'acquisition ou le type de contrat peuvent indirectement reproduire certaines différences entre groupes d'âge.

Il faut distinguer :

- Conformité explicite : l'âge n'est pas utilisé directement.
- Analyse des effets indirects : vérifier si le système produit des écarts de sélection importants selon les groupes d'âge.
- Explicabilité : être capable de justifier les raisons opérationnelles du ciblage.

Je recommanderais de conserver l'âge dans un jeu d'audit séparé, avec accès contrôlé et sous réserve de validation juridique, mais jamais dans le pipeline de scoring.

### Explicabilité du modèle

Pour un modèle de type Logistic Regression, les coefficients fournissent une interprétation relativement simple, sous réserve des transformations et interactions.

Pour LightGBM ou XGBoost, tu peux utiliser SHAP pour expliquer les facteurs qui contribuent à une prédiction.

Mais attention : SHAP explique la prédiction du modèle, pas la cause réelle de la résiliation.

Le message remis aux conseillers devrait rester compréhensible :

> Ce client a été priorisé en raison d'un risque de résiliation élevé, notamment associé à son historique de sollicitations du support et à son type de contrat. Le ciblage respecte les règles d'éligibilité et de rentabilité définies pour la campagne.

Les motifs doivent naturellement correspondre aux véritables facteurs du modèle pour ce client.

## 8. Quelle stratégie de Machine Learning adopter ?

Je ne commencerais pas directement par XGBoost.

Je construirais plusieurs niveaux de modèles pour démontrer une démarche scientifique.

| Modèle              | Objectif                  | Intérêt                                       |
| ------------------- | ------------------------- | --------------------------------------------- |
| Dummy Classifier    | Baseline minimale         | Vérifier la valeur ajoutée                    |
| Logistic Regression | Baseline interprétable    | Explication et calibration                    |
| LightGBM            | Modèle candidat principal | Performance et efficacité                     |
| CatBoost            | Challenger                | Variables catégorielles et valeurs manquantes |

Je privilégierais LightGBM ou CatBoost comme candidats sérieux pour ce type de données tabulaires.

Le choix final dépendrait des résultats du backtesting temporel, de la calibration, de la stabilité et de la valeur économique, et non uniquement du ROC-AUC.

### Une pipeline professionnelle

```
from sklearn.pipeline import Pipelinefrom sklearn.compose import ColumnTransformerfrom sklearn.preprocessing import OneHotEncoder, StandardScalerfrom sklearn.impute import SimpleImputerfrom sklearn.linear_model import LogisticRegressionnumeric_features = [    "tenure_months",    "monthly_fee",    "data_gb",    "support_calls_3m",    "late_payments",    "nps",    "last_invoice_amount",    "retention_offer_declined",]categorical_features = [    "contract",    "region",    "dept",    "channel",]numeric_pipeline = Pipeline([    ("imputer", SimpleImputer(strategy="median")),    ("scaler", StandardScaler()),])categorical_pipeline = Pipeline([    ("imputer", SimpleImputer(        strategy="most_frequent"    )),    ("encoder", OneHotEncoder(        handle_unknown="ignore"    )),])preprocessor = ColumnTransformer([    ("num", numeric_pipeline, numeric_features),    ("cat", categorical_pipeline, categorical_features),])model = Pipeline([    ("preprocessor", preprocessor),    ("classifier", LogisticRegression(        max_iter=1000,        class_weight="balanced",        random_state=42,    )),])
```

C'est une baseline, pas encore une pipeline de production complète.

Deux précautions importantes :

- `class_weight="balanced"` peut améliorer certains comportements du classifieur, mais peut aussi dégrader la calibration brute des probabilités. Il faut le comparer à une version non pondérée et recalibrer si nécessaire.
- Le traitement des valeurs manquantes, l'encodage et la calibration doivent être ajustés uniquement sur les périodes d'entraînement autorisées.

Je mettrais particulièrement l'accent sur la calibration, car une probabilité incorrecte fausse directement le calcul de rentabilité.

## 9. MLOps : comment industrialiser cette solution ?

La direction indique deux contraintes structurantes :

- Scorer 1,2 million de clients.
- Réentraîner le modèle chaque mois.

Il faut donc proposer une architecture reproductible, observable et maintenable.

### 9.1 Architecture cible

Pipeline mensuel de production

Data Warehouse / Data Lake

Snapshots mensuels, facturation, historique clients

Validation et Feature Engineering

Schéma, qualité, données disponibles à date, exclusion de l'âge

Training

Backtesting, calibration, MLflow

Validation

Tests, comparaison champion/challenger

Batch Scoring — 1,2 M clients

Probabilités calibrées, gains attendus, Top 300

CRM / Conseillers

Liste priorisée, justification, suivi des contacts

Monitoring et Feedback Loop

Qualité, drift, calibration, résultats économiques, expérimentation

### 9.2 Choix technologiques

Pour ce test, je ne proposerais pas une infrastructure Kubernetes complète.

Cela risquerait de donner l'impression d'une solution surdimensionnée.

Je proposerais plutôt :

| Besoin                   | Technologie possible              |
| ------------------------ | --------------------------------- |
| Traitement des données   | Python, Pandas, Parquet           |
| Modélisation             | scikit-learn, LightGBM, CatBoost  |
| Tracking des expériences | MLflow                            |
| Validation des données   | Pandera ou Great Expectations     |
| Orchestration mensuelle  | Airflow ou orchestrateur existant |
| Packaging                | Docker                            |
| Tests                    | pytest                            |
| CI/CD                    | GitHub Actions ou GitLab CI       |
| Monitoring               | Evidently + métriques métier      |
| Stockage des résultats   | Base SQL ou Data Warehouse        |

À cette échelle, 1,2 million de lignes avec quelques dizaines de variables ne nécessite pas automatiquement Spark ou une architecture distribuée.

Je commencerais par mesurer le temps de scoring batch sur un environnement CPU représentatif, avec une contrainte de durée définie avec le métier.

### 9.3 Le piège du réentraînement mensuel

Le test indique que le modèle est réentraîné chaque mois.

Cela ne signifie pas que le nouveau modèle doit automatiquement remplacer celui en production.

Je distinguerais :

1. Réentraînement du modèle candidat.
2. Évaluation sur des données temporellement valides.
3. Vérification de la calibration et des métriques métier.
4. Comparaison avec le modèle champion.
5. Promotion seulement si les critères de validation sont satisfaits.
6. Conservation d'une possibilité de rollback.

C'est un point important de maturité MLOps.

### 9.4 Les métriques de monitoring

Je suivrais quatre catégories.

| Dimension     | Indicateurs                                          |
| ------------- | ---------------------------------------------------- |
| Data quality  | Nulls, schéma, fraîcheur, distributions              |
| Model quality | PR-AUC, calibration, Brier Score                     |
| Business      | Contacts, acceptations, churn évité estimé, gain net |
| Operations    | Durée de scoring, échecs, disponibilité des données  |

Les métriques dépendant du churn réel seront disponibles avec le décalage de 30 jours.

Le gain causal réel exigera une stratégie d'évaluation appropriée, notamment un groupe de contrôle.

## 10. Structure de projet que je présenterais à l'entreprise

Je te recommande un dépôt GitHub propre, relativement léger, avec une séparation claire entre exploration, modélisation, décision et production.

telecom-retention-intelligence/

```
telecom-retention-intelligence/
├── README.md
├── pyproject.toml
├── Dockerfile
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_eda_churn.ipynb
│   ├── 04_temporal_modeling.ipynb
│   ├── 05_campaign_optimization.ipynb
│   └── 06_pilot_causal_analysis.ipynb
│
├── src/
│   └── retention/
│       ├── data/
│       │   ├── ingestion.py
│       │   └── validation.py
│       ├── features/
│       │   └── engineering.py
│       ├── modeling/
│       │   ├── training.py
│       │   ├── evaluation.py
│       │   └── calibration.py
│       ├── decision/
│       │   ├── economics.py
│       │   └── targeting.py
│       ├── causal/
│       │   └── pilot_analysis.py
│       └── monitoring/
│           └── drift.py
│
├── pipelines/
│   ├── train.py
│   ├── score.py
│   └── campaign.py
│
├── tests/
│   ├── test_features.py
│   ├── test_temporal_split.py
│   ├── test_economics.py
│   └── test_targeting.py
│
├── configs/
│   └── config.yaml
│
├── reports/
│   └── figures/
│
└── .github/
    └── workflows/
        └── ci.yml
```

Les notebooks servent à expliquer les analyses, tandis que le code réutilisable doit vivre dans `src/`.

Je mettrais dans le `README.md` les hypothèses métier, les résultats, les limites, les commandes d'exécution et les décisions techniques.

Les données brutes potentiellement sensibles ne devraient pas être publiées dans le dépôt sans autorisation.

## 11. Les dix pièges que je préparerais pour l'entretien

| Piège                                       | Réponse attendue d'un profil expérimenté                    |
| ------------------------------------------- | ----------------------------------------------------------- |
| 1. Split aléatoire                          | Backtesting temporel respectant la disponibilité des labels |
| 2. Accuracy comme KPI principal             | PR-AUC, calibration, métriques Top-K et gain économique     |
| 3. Top 300 par churn                        | Top 300 par gain net attendu positif                        |
| 4. Probabilités non calibrées               | Évaluation et calibration hors entraînement                 |
| 5. Pilote non randomisé                     | Prudence sur l'identification causale                       |
| 6. Acceptation de 30 % supposée certaine    | Analyse de sensibilité et mesure future                     |
| 7. Âge supprimé sans audit                  | Contrôle des variables proxy et des effets indirects        |
| 8. Extrapolation naïve à 1,2 M              | Analyse de représentativité et hypothèses explicites        |
| 9. Réentraînement automatique sans contrôle | Validation champion/challenger                              |
| 10. Architecture trop complexe              | Batch ML simple, reproductible, mesuré et monitoré          |

## 12. Comment je prioriserais ton travail

Le sujet comporte aussi une partie 2 consacrée à la base de connaissances et à l'évaluation RAG. Mais l'énoncé désigne explicitement la partie churn comme le cœur du test.

Je répartirais donc le temps de la manière suivante.

Compréhension et qualité des données

15 %

Modélisation et validation temporelle

25 %

Optimisation économique et incertitude

25 %

Analyse causale du pilote

15 %

MLOps et conformité

10 %

Documentation et restitution

10 %

Cette répartition concerne la partie 1 et sa restitution. Il faudra réserver séparément du temps pour la partie RAG selon le temps total dont tu disposes.

## 13. Ce que je présenterais oralement au recruteur

Voici le message central que je défendrais :

> J'ai abordé ce problème comme un système d'aide à la décision, plutôt que comme un simple exercice de classification. J'ai d'abord construit une approche de prédiction du churn respectant la temporalité des données et la disponibilité différée des labels.
>
> J'ai ensuite relié les probabilités calibrées à une fonction de valeur économique afin de prioriser les clients dans la limite des 300 contacts mensuels.
>
> J'ai distingué la rentabilité théorique de la campagne de son efficacité causale réelle, qui ne peut pas être démontrée directement à partir d'un pilote dont l'affectation au traitement n'est pas documentée.
>
> Enfin, j'ai proposé une architecture de scoring batch, avec traçabilité, contrôle de qualité, réentraînement supervisé et monitoring, adaptée à un parc de 1,2 million de clients.

Ce discours est pertinent à condition que le projet livré démontre réellement les éléments annoncés.

## 14. Mon évaluation du niveau de difficulté

Difficulté technique globale

Mon appréciation : 4/5, surtout à cause des aspects décisionnels, temporels et causaux.

Ce qui différenciera un bon rendu d'un excellent rendu

Un bon candidat produit un modèle prédictif correctement évalué.

Un candidat expérimenté démontre que le modèle peut soutenir une décision rentable, explique les limites de l'estimation, respecte les contraintes légales et propose un dispositif réaliste pour mesurer son impact en production.

### La prochaine étape que je recommande

Avant de coder les modèles, il faut réaliser un audit statistique complet de `churn_clients.csv` et `pilote_campagne_2023.csv`.

Les extraits que tu as partagés permettent de comprendre la structure, mais pas de calculer les véritables distributions, taux de churn, déséquilibres du pilote ou performances prédictives.

Je commencerais par un notebook `01_data_understanding.ipynb`, puis `02_data_quality.ipynb`, avec une attention particulière à la granularité client-mois, aux doublons, à la disponibilité temporelle des variables, à la distribution des abonnements et aux différences entre les groupes du pilote.

C'est sur cette base que nous pourrons construire un modèle et une stratégie de campagne scientifiquement défendables, plutôt que de choisir des algorithmes avant de comprendre les données.