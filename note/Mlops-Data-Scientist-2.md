# Analyse approfondie du test technique Data Scientist — Churn Prediction & Retention Optimization

Perspective : Senior Data Scientist / Machine Learning Engineer — Modélisation, causalité, décision économique et MLOps

Ce test est particulièrement intéressant parce qu'il ne demande pas simplement de construire un modèle de Machine Learning. Il demande de concevoir un système de décision métier capable de générer de la valeur économique, tout en respectant des contraintes temporelles, juridiques et opérationnelles.

Je vais l'analyser comme je le ferais lors d'une revue technique dans une équipe Data Science expérimentée : en distinguant la problématique métier, les hypothèses statistiques, les choix de modélisation, les risques méthodologiques et l'industrialisation.

Une distinction essentielle guidera toute notre analyse :

Prédire qu'un client va partir ne signifie pas que le contacter permettra de le retenir.

C'est probablement le point le plus important de ce test.

## 1. Comprendre la véritable problématique métier

### 1.1 Ce que demande explicitement l'entreprise

L'opérateur télécom possède 1,2 million de clients.

Chaque mois, certains clients résilient leur abonnement. Cette résiliation est appelée churn.

L'entreprise souhaite anticiper les départs pour proposer une offre commerciale aux clients susceptibles de partir.

Mais elle dispose de ressources limitées :

- Seulement 300 clients peuvent être contactés par mois.
- Chaque contact coûte 10 €.
- Tous les clients contactés n'acceptent pas l'offre.
- Certains clients acceptent une remise alors qu'ils n'avaient aucune intention de partir.
- Les prédictions doivent être disponibles à la fin de chaque mois.

L'objectif n'est donc pas de contacter tous les clients susceptibles de résilier.

L'objectif est de sélectionner les 300 clients qui présentent la meilleure espérance de gain économique grâce à une intervention commerciale.

### 1.2 Exemple concret

Imaginons trois clients.

| Client | Probabilité de churn | Abonnement mensuel |
| ------ | -------------------- | ------------------ |
| Alice  | 85 %                 | 20 €               |
| Bruno  | 65 %                 | 100 €              |
| Clara  | 30 %                 | 150 €              |

Une approche classique consisterait à sélectionner Alice, puis Bruno, puis Clara, en classant les clients selon leur probabilité de résiliation.

Mais est-ce réellement la meilleure décision ?

Pas nécessairement.

Bruno et Clara peuvent représenter une valeur économique plus importante, malgré une probabilité de départ plus faible.

Il faut donc intégrer simultanément :

1. Le risque de résiliation.
2. La valeur financière du client.
3. La probabilité d'acceptation de l'offre.
4. Le coût de la campagne.
5. L'effet d'aubaine.

C'est ce qui transforme un problème de classification binaire en problème d'optimisation de décision sous contrainte de capacité.

## 2. Ce que les recruteurs cherchent réellement à évaluer

| Dimension          | Question implicite du recruteur                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| Data understanding | Comprends-tu la granularité client × mois et les données disponibles au moment de la prédiction ? |
| Machine Learning   | Sais-tu construire un modèle sans fuite de données ?                                              |
| Évaluation         | Sais-tu différencier performance statistique et performance économique ?                          |
| Business           | Sais-tu transformer des probabilités en décisions rentables ?                                     |
| Causal inference   | Sais-tu distinguer prédiction de churn et effet réel d'une campagne ?                             |
| Statistiques       | Sais-tu quantifier l'incertitude de tes résultats ?                                               |
| Légal              | Sais-tu construire une décision explicable et contrôler les risques de discrimination ?           |
| MLOps              | Sais-tu maintenir un modèle mensuel sur 1,2 million de clients ?                                  |

Un candidat junior pourrait se limiter à entraîner XGBoost, calculer un ROC-AUC et sélectionner les 300 plus gros scores.

Un candidat expérimenté doit aller beaucoup plus loin : démontrer que son système prend de bonnes décisions, expliquer sous quelles hypothèses elles sont bonnes et identifier ce qui n'est pas encore démontré.

## 3. Comprendre l'architecture du problème

Architecture fonctionnelle du système de rétention

Cette architecture montre pourquoi il ne faut pas confondre le modèle de churn avec la solution complète.

Le modèle prédictif n'est qu'un composant. La décision finale est produite par un moteur économique, puis soumise aux contraintes de capacité.

Le suivi de campagne permet ensuite de confronter les hypothèses aux résultats observés.

## 4. Comprendre les données et les deux fichiers de la partie 1

### 4.1 `churn_clients.csv` — apprendre à prédire

Ce fichier contient des observations client × mois pendant l'année 2024.

Par exemple :

| client_id | snapshot_date | monthly_fee | churn |
| --------- | ------------- | ----------- | ----- |
| C001      | 2024-01-31    | 50 €        | 0     |
| C001      | 2024-02-29    | 50 €        | 0     |
| C001      | 2024-03-31    | 50 €        | 1     |
| C002      | 2024-01-31    | 80 €        | 0     |
| C002      | 2024-02-29    | 80 €        | 0     |

Exemple fictif destiné à expliquer la structure, et non extrait du CSV réel.

Le client C001 est présent à plusieurs fins de mois.

À la fin du mois de mars, `churn = 1` signifie qu'il résilie pendant les 30 jours suivants.

Le même client peut donc contribuer plusieurs observations au dataset.

C'est un panel longitudinal, et non une simple collection d'individus indépendants.

Cette distinction a plusieurs conséquences importantes :

- Les observations d'un même client sont corrélées.
- Le risque de résiliation peut évoluer dans le temps.
- Les variables doivent représenter l'information disponible au `snapshot_date`.
- Il faut respecter l'ordre chronologique pour évaluer le modèle.

### 4.2 `pilote_campagne_2023.csv` — étudier les effets de la campagne

Ce deuxième fichier contient 4 000 clients observés lors d'une campagne pilote en novembre 2023.

Il comporte notamment :

- `contacted` : indique si le client a été contacté.
- `churn_30d` : indique si le client a résilié dans les 30 jours.

Son rôle est fondamentalement différent.

|                        | `churn_clients.csv`                   | `pilote_campagne_2023.csv`                   |
| ---------------------- | ------------------------------------- | -------------------------------------------- |
| Période                | 2024                                  | Novembre 2023                                |
| Objectif               | Prédire le churn                      | Étudier l'impact potentiel du contact        |
| Variable cible         | `churn`                               | `churn_30d`                                  |
| Variable de traitement | Non précisée                          | `contacted`                                  |
| Usage principal        | Entraînement et validation prédictive | Analyse observationnelle et limites causales |
| Risque majeur          | Fuite temporelle                      | Biais de sélection                           |

Attention : le pilote n'est pas présenté comme un essai randomisé. Le mécanisme ayant déterminé qui était contacté n'est pas documenté.

On ne peut donc pas affirmer que la différence de churn entre les deux groupes représente l'effet causal de la campagne.

C'est un piège méthodologique majeur.

## 5. Le piège majeur : le Data Leakage temporel

C'est l'un des premiers sujets sur lesquels je challengerais un candidat pendant l'entretien.

### 5.1 Pourquoi un `train_test_split` aléatoire est problématique

Supposons que nous utilisions :

```
from sklearn.model_selection import train_test_splitX_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.2,    random_state=42,    stratify=y)
```

Ce code est parfaitement valide techniquement, mais il ne respecte pas la chronologie du problème métier.

Il peut entraîner le modèle sur des observations de novembre 2024, puis l'évaluer sur des observations de mars 2024.

Or, en production, il est impossible d'utiliser le futur pour prédire le passé.

Le modèle risque de profiter indirectement d'informations qui n'auraient pas été disponibles au moment de la décision.

### 5.2 La bonne approche : validation temporelle

Je recommande une stratégie de type walk-forward validation, avec une gestion explicite des 30 jours nécessaires pour connaître le label.

Exemple de validation temporelle

Jan

Fév

Mar

Avr

Mai

Juin

Train

Train

Train

Train

Gap

Test

Entraînement

Embargo

Validation

Schéma illustratif : les mois utilisables dépendent des dates exactes de disponibilité des labels.

L'idée est d'utiliser uniquement les observations dont les labels sont effectivement connus à la date de l'entraînement.

Par exemple, au 30 juin 2024, les labels des observations du 31 mai ne sont pas nécessairement tous disponibles : leur fenêtre de 30 jours n'est pas entièrement terminée.

Il faut définir :

\\[ t\_{\text{label disponible}} = t\_{\text{snapshot}}+30\text{ jours}+\delta \\]

où \\(\delta\\) représente un éventuel délai de consolidation des données.

Une règle professionnelle : chaque observation d'entraînement doit satisfaire `label_available_at <= training_cutoff`.

Le test précise également que le modèle est réentraîné chaque mois. Je proposerais donc plusieurs fenêtres de validation successives, puis un test final chronologique, jamais utilisé pour choisir les hyperparamètres.

### 5.3 Autre piège : les variables construites après le snapshot

Imaginons une variable :

`number_of_complaints_next_30d`

Elle serait très prédictive du churn, mais inutilisable en production puisqu'elle décrit le futur.

Même problème pour un montant de facture actualisé après la date de décision ou un statut de résiliation déjà enregistré.

Il faut auditer chaque feature selon sa date de disponibilité effective, et pas seulement selon sa corrélation avec le churn.

## 6. Quelle stratégie de Machine Learning choisir ?

Je ne commencerais pas directement par XGBoost.

Je proposerais trois niveaux de modèles.

| Modèle                          | Pourquoi l'utiliser ?                                                         | Limites                                                 |
| ------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------- |
| Régression logistique           | Baseline explicable, rapide, probabilités interprétables                      | Relations essentiellement linéaires sur l'échelle logit |
| HistGradientBoosting / LightGBM | Capture les non-linéarités et interactions, performant sur données tabulaires | Explicabilité moins immédiate                           |
| CatBoost                        | Intéressant avec des variables catégorielles et valeurs manquantes            | Nécessite une validation et une calibration rigoureuses |

### 6.1 Ce que je recommande

Construire une baseline de régression logistique, puis un modèle de gradient boosting, et comparer les deux.

Mais attention : le meilleur modèle n'est pas forcément celui qui possède le meilleur ROC-AUC.

Le système n'a que 300 places par mois.

Il faut donc évaluer la qualité des prédictions précisément dans la zone où les décisions sont prises.

### 6.2 Les métriques réellement importantes

Je retiendrais :

- PR-AUC : particulièrement utile lorsque le churn est rare.
- ROC-AUC : mesure globale de discrimination.
- Brier score et courbes de calibration : vérifient la qualité des probabilités.
- Precision\@300 : proportion de churners observés parmi les 300 sélectionnés.
- Recall\@300 : proportion des churners totaux capturés par ces 300 clients.
- Gain économique simulé : mesure la valeur attendue de la politique de ciblage.

Il faut cependant distinguer `Precision@300` pour un classement par risque et la performance d'une sélection par gain économique. Les 300 clients les plus rentables ne sont pas nécessairement les 300 plus susceptibles de résilier.

### 6.3 Pourquoi la calibration est cruciale

Supposons que le modèle annonce 80 % de probabilité de churn pour 1 000 clients.

Si, parmi ces 1 000 clients, seulement 50 % résilient effectivement, le modèle est mal calibré.

Il peut conserver une excellente capacité à classer les clients tout en produisant des probabilités trompeuses.

Or, notre moteur économique utilisera directement ces probabilités.

Une mauvaise calibration peut donc entraîner de mauvaises décisions financières, même avec un bon ROC-AUC.

Je vérifierais les courbes de fiabilité par période et par segment, puis testerais éventuellement une calibration sigmoïde ou isotone sur une période distincte de l'entraînement et du test.

# 7. Le cœur du test : construire la fonction de gain économique

C'est probablement la partie sur laquelle je consacrerais le plus d'efforts.

Nous devons traduire le contexte métier en une équation permettant de prendre une décision pour chaque client.

## 7.1 Définir les variables

Pour un client \\(i\\), posons :

| Symbole         | Signification                                         |
| --------------- | ----------------------------------------------------- |
| \\(p_i\\)       | Probabilité prédite de churn                          |
| \\(m_i\\)       | Montant mensuel de l'abonnement                       |
| \\(a_1=0.30\\)  | Probabilité d'acceptation si le client allait partir  |
| \\(a_0=0.10\\)  | Probabilité d'acceptation si le client ne partait pas |
| \\(c=10\\)      | Coût du contact                                       |
| \\(r=0.40\\)    | Taux de marge mensuelle                               |
| \\(d=0.90m_i\\) | Coût total de la remise                               |

### 7.2 Cas 1 : le client allait résilier

Si le client accepte l'offre, il reste 12 mois.

La marge récupérée est :

\\[ 12\times0.40\times m_i=4.8m_i \\]

Il faut retrancher le coût de la remise :

\\[ 4.8m_i-0.9m_i=3.9m_i \\]

Le client n'accepte que dans 30 % des cas.

Le gain attendu conditionnel au fait qu'il allait résilier est donc :

\\[ G\_{\text{churner}}=0.30\times3.9m_i \\]

\\[ \boxed{G\_{\text{churner}}=1.17m_i} \\]

### 7.3 Cas 2 : le client n'allait pas résilier

Ce client serait resté même sans campagne.

Il n'y a donc aucune marge supplémentaire récupérée du seul fait de son maintien.

En revanche, il a 10 % de chances d'accepter une remise inutile.

\\[ G\_{\text{non-churner}}=0.10\times(-0.9m_i) \\]

\\[ \boxed{G\_{\text{non-churner}}=-0.09m_i} \\]

C'est ce que l'énoncé appelle l'effet d'aubaine.

### 7.4 Combiner les deux scénarios

La probabilité que le client allait partir est \\(p_i\\).

La probabilité qu'il allait rester est \\(1-p_i\\).

Le gain espéré d'un contact est donc :

\\[ G_i=p_i(1.17m_i)+(1-p_i)(-0.09m_i)-10 \\]

En simplifiant :

\\[ \boxed{G_i=(1.26p_i-0.09)m_i-10} \\]

C'est la fonction économique centrale de notre solution, sous les hypothèses de l'énoncé.

Elle suppose notamment que l'offre empêche effectivement le départ pendant les 12 mois indiqués et que les coûts de remise sont comptabilisés selon la convention fournie.

## 7.5 Simulateur de rentabilité d'un contact

Tu peux modifier la probabilité de churn et le montant de l'abonnement pour comprendre l'effet de chaque variable.

Probabilité de churn

## 65 %

Abonnement mensuel

## 80 €

Gain espéré du contact

# 48,32 €

Contact rentable en espérance

Probabilité minimale rentable

# 17,1 %

Pour cet abonnement

Ce calcul représente un gain incrémental attendu selon les hypothèses économiques du test, pas un gain causal déjà mesuré.

### 7.6 Exemple comparatif de trois clients

Reprenons Alice, Bruno et Clara.

| Client | Churn prédit | Abonnement | Gain espéré |
| ------ | ------------ | ---------- | ----------- |
| Alice  | 85 %         | 20 €       | 9,62 €      |
| Bruno  | 65 %         | 100 €      | 62,90 €     |
| Clara  | 30 %         | 150 €      | 33,20 €     |

Le classement par risque de churn serait :

Alice → Bruno → Clara.

Le classement par gain économique est :

Bruno → Clara → Alice.

C'est une différence fondamentale.

Un modèle de classification permet d'estimer le risque. Mais c'est le calcul économique qui permet de transformer ce risque en décision.

## 8. Comment sélectionner les 300 clients ?

Mathématiquement, nous avons un problème d'optimisation sous contrainte :

\\[ \max\_{x_1,\ldots,x_N}\sum\_{i=1}^{N}x_iG_i \\]

avec :

\\[ \sum\_{i=1}^{N}x_i\leq300 \\]

et :

\\[ x_i\in\\{0,1\\} \\]

où \\(x_i=1\\) signifie que le client est sélectionné pour être contacté.

Puisque tous les contacts consomment une unité de capacité, la solution est simple :

1. Calculer le gain espéré de chaque client éligible.
2. Exclure les clients dont le gain espéré est négatif ou nul.
3. Trier les autres par gain décroissant.
4. Sélectionner les 300 premiers au maximum.

Voici une implémentation Python simplifiée :

```
import numpy as npdef select_retention_targets(    clients,    churn_model,    capacity=300,):    result = clients.copy()    # Features identiques à celles de l'entraînement    X = result.drop(        columns=["client_id", "monthly_fee"],        errors="ignore",    )    # Probabilités idéalement calibrées    result["p_churn"] = churn_model.predict_proba(X)[:, 1]    # Gain incrémental espéré    result["expected_gain"] = (        (1.26 * result["p_churn"] - 0.09)        * result["monthly_fee"]        - 10    )    # Décision sous contrainte    selected = (        result.loc[result["expected_gain"] > 0]        .sort_values("expected_gain", ascending=False)        .head(capacity)    )    return selected
```

Ce code illustre la logique métier, mais ce n'est pas encore du code de production : les features doivent être préparées par le même pipeline qu'à l'entraînement, et les contrôles d'éligibilité doivent être appliqués avant la sélection.

Une précision importante : la contrainte est 300 contacts au maximum, pas une obligation de contacter exactement 300 personnes.

Si seulement 220 clients ont un gain attendu positif, contacter 80 clients supplémentaires détruirait de la valeur selon notre modèle.

# 9. Comment estimer le gain annuel et son incertitude ?

Le recruteur ne demande pas uniquement un gain mensuel.

Il souhaite connaître le gain annuel attendu, ainsi que l'incertitude associée.

Cela nécessite une démarche statistique rigoureuse.

## 9.1 Première approche : estimation du gain annuel

Pour un mois \\(t\\), on calcule :

\\[ G_t=\sum\_{i\in S_t}G_i \\]

où \\(S_t\\) représente les clients sélectionnés.

Le gain annuel attendu est :

\\[ \boxed{G\_{\text{annuel}}=\sum\_{t=1}^{12}G_t} \\]

Attention : on ne peut pas simplement multiplier par 12 un résultat mensuel et prétendre avoir validé une politique annuelle.

Le portefeuille de clients évolue, les probabilités de churn changent et les mêmes clients peuvent apparaître plusieurs mois.

Il faut également définir les règles de recontact et tenir compte des effets d'une campagne précédente.

### Exemple illustratif

Supposons que les 300 clients sélectionnés présentent un gain moyen espéré de 35 €.

Gain mensuel

# 10 500 €

Gain annuel

# 126 000 €

Contacts/an

# 3 600

Scénario purement illustratif, avec 300 contacts chaque mois et un gain moyen stable de 35 €.

## 9.2 Pourquoi ajouter un intervalle d'incertitude ?

Le gain annuel dépend de plusieurs quantités incertaines :

- Les probabilités de churn estimées par le modèle.
- Le taux d'acceptation de 30 %, issu d'une enquête.
- Le taux d'acceptation des non-churners.
- La stabilité des comportements des clients.
- Les conditions économiques et la saisonnalité.

Un chiffre de 126 000 € sans intervalle d'incertitude pourrait donner une impression de précision injustifiée.

### Méthode recommandée : bootstrap temporel et simulation Monte Carlo

Je proposerais deux analyses complémentaires.

Premièrement, une validation historique de la politique de ciblage. Pour chaque période de test, nous simulons les décisions que nous aurions prises à partir des seules informations disponibles à cette date.

Deuxièmement, une analyse de sensibilité probabiliste. Nous faisons varier les paramètres économiques et comportementaux dans des plages plausibles.

Par exemple, pour le taux d'acceptation des churners, nous pourrions comparer les scénarios suivants :

Sensibilité du gain attendu au taux d'acceptation

Exemple pour un client avec p(churn) = 65 % et monthly_fee = 100 €.

20 €40 €60 €80 €100 €15 %20 %25 %30 %35 %40 %

Les autres hypothèses sont maintenues constantes. Ces scénarios ne constituent pas un intervalle de confiance.

Il faut être rigoureux : un bootstrap sur les clients ne suffit pas à quantifier l'incertitude sur l'efficacité réelle de l'offre.

Un bootstrap temporel peut être pertinent, mais avec seulement douze mois de données, son estimation de la variabilité annuelle restera fragile.

Je distinguerais donc explicitement l'incertitude statistique, l'incertitude sur les hypothèses économiques et l'incertitude causale.

# 10. Le pilote de novembre 2023 : la partie la plus subtile du test

Cette section est essentielle pour démontrer une maturité de Data Scientist.

Le pilote contient 4 000 clients, dont certains ont été contactés et d'autres non.

L'idée intuitive serait de comparer les taux de churn des deux groupes.

Mais il existe un problème.

## 10.1 Corrélation n'est pas causalité

Imaginons que nous observions :

| Groupe        | Clients | Churn observé |
| ------------- | ------- | ------------- |
| Contactés     | 2 000   | 18 %          |
| Non contactés | 2 000   | 10 %          |

Chiffres fictifs pour illustrer le raisonnement.

Une interprétation naïve serait :

« La campagne augmente le churn de 8 points ! »

Mais ce serait une conclusion incorrecte.

Pourquoi ?

Parce que les conseillers ont peut-être volontairement contacté les clients les plus susceptibles de résilier.

Le groupe contacté était alors potentiellement plus risqué dès le départ.

Nous sommes face à un biais de sélection.

## 10.2 Introduire le raisonnement causal

Définissons :

\\[ Y_i(1)=\text{churn du client s'il est contacté} \\]

\\[ Y_i(0)=\text{churn du client s'il n'est pas contacté} \\]

L'effet individuel du contact est :

\\[ \tau_i=Y_i(1)-Y_i(0) \\]

Le problème fondamental est que nous ne pouvons observer qu'un seul de ces deux résultats pour chaque client.

Un client ne peut pas simultanément être contacté et non contacté dans les mêmes conditions.

L'objectif d'une analyse causale est d'estimer ce qui se serait passé en l'absence d'intervention.

C'est ce qu'on appelle le contrefactuel.

### 10.3 Pourquoi le pilote ne permet pas automatiquement de mesurer cet effet

Le tirage des clients contactés n'est pas documenté.

Nous ne savons pas si la décision dépendait de leur historique de churn, de leur abonnement, de leur ancienneté ou d'autres facteurs.

Une régression ajustée, un propensity score matching ou une méthode doubly robust peuvent être envisagés si les variables nécessaires sont disponibles.

Mais ces méthodes nécessitent des hypothèses, notamment l'absence de facteurs de confusion non mesurés et un recouvrement suffisant entre les groupes.

Elles ne remplacent pas une randomisation.

Par ailleurs, le pilote ne mesure pas directement le taux d'acceptation de 30 % mentionné dans l'énoncé.

Il serait donc incorrect de présenter ce taux comme un résultat expérimental.

## 10.4 Ce que je proposerais pour améliorer la campagne

Mettre en place un essai contrôlé randomisé.

Population éligible

Clients sélectionnés selon des règles préétablies

Randomisation

Groupe traité

Contact et proposition commerciale

Groupe contrôle

Aucun contact de rétention

Mesurer la différence de churn et de profit

À 30 jours, puis sur un horizon économique plus long

La randomisation doit être conçue en respectant la capacité de 300 contacts. Par exemple, nous pouvons constituer une population éligible, randomiser l'attribution des places disponibles et conserver un groupe témoin comparable.

L'analyse en intention de traiter permettrait d'évaluer l'effet de l'attribution à la campagne, y compris lorsque certains contacts échouent.

Il faudrait aussi mesurer les acceptations, les remises réellement accordées, les marges et la rétention à plus long terme.

## 10.5 L'étape suivante : l'Uplift Modeling

Un modèle classique prédit :

\\[ P(Y=1\mid X) \\]

C'est le risque de churn.

Un modèle d'uplift cherche à estimer :

\\[ U(X)=P(Y=1\mid X,T=0)-P(Y=1\mid X,T=1) \\]

C'est la réduction de probabilité de churn attribuable au contact, conditionnellement aux caractéristiques du client.

Pourquoi est-ce important ?

Un client peut avoir une probabilité de churn de 95 %, mais être impossible à convaincre.

Un autre peut avoir une probabilité de churn de 55 %, mais être très sensible à une offre commerciale.

Le second client peut représenter une meilleure opportunité de rétention.

À terme, le système idéal serait donc un modèle de profit incrémental fondé sur l'effet causal du traitement, plutôt qu'un simple classement par risque de churn.

Je ne construirais toutefois pas directement un uplift model sur le pilote de 2023 sans examiner les conditions d'identification causale.

# 11. La contrainte juridique : ne pas utiliser directement l'âge

L'énoncé interdit explicitement l'utilisation directe de l'âge dans la décision de ciblage.

Il serait insuffisant de simplement supprimer la colonne `age` sans autre vérification.

Pourquoi ?

Parce que certaines variables peuvent être corrélées à l'âge et agir comme des variables proxy.

Par exemple, l'ancienneté, certaines habitudes d'utilisation ou le type d'abonnement peuvent être associés à des groupes d'âge.

Je proposerais donc :

1. Exclure l'âge des features utilisées pour produire les décisions, y compris les transformations directes de l'âge.
2. Auditer les variables susceptibles d'agir comme proxies.
3. Évaluer les disparités de sélection et les erreurs entre groupes d'âge lorsque cette analyse est légalement autorisée.
4. Fournir des explications individuelles fondées sur les variables réellement utilisées.
5. Documenter les arbitrages, la finalité du traitement et les contrôles juridiques avec les équipes compétentes.

Une régression logistique peut faciliter l'interprétation globale, tandis que SHAP peut aider à comprendre les contributions des variables dans un modèle plus complexe.

Cependant, une valeur SHAP n'est pas une explication causale.

Il faut aussi distinguer explicabilité technique et explication compréhensible pour le client.

Une explication destinée au conseiller pourrait être :

> Le client présente un risque élevé de résiliation, notamment en raison d'une diminution récente de son utilisation et de plusieurs incidents de service. Une offre de rétention peut être proposée conformément aux critères d'éligibilité.

Cette formulation reste à adapter aux variables réellement disponibles et aux exigences juridiques.

# 12. Mise en production : architecture MLOps recommandée

Le système doit traiter 1,2 million de clients chaque mois.

Ce volume n'impose pas automatiquement Spark, Kubernetes ou une architecture distribuée complexe.

Un traitement batch avec Python, Parquet et un moteur performant peut être parfaitement suffisant selon le nombre de features, les ressources disponibles et le temps d'exécution exigé.

Je privilégierais une architecture simple, testable et reproductible.

### 12.1 Les composants techniques

| Composant           | Solution possible                       |
| ------------------- | --------------------------------------- |
| Ingestion           | SQL, Python, Parquet                    |
| Feature engineering | pandas, Polars ou PySpark si nécessaire |
| Modélisation        | scikit-learn, LightGBM, CatBoost        |
| Expérimentation     | MLflow                                  |
| Orchestration       | Airflow ou jobs planifiés               |
| Tests               | pytest, contrôles de schéma et qualité  |
| Packaging           | Package Python, Docker                  |
| Déploiement         | Batch job conteneurisé                  |
| Monitoring          | Métriques, logs, alertes et rapports    |

### 12.2 Les contrôles indispensables

Je mettrais en place des contrôles à plusieurs niveaux.

Avant le scoring : unicité client × snapshot, cohérence des dates, données manquantes, schéma, disponibilité des features et exclusion des clients déjà résiliés.

Pendant le scoring : version du modèle, validité des probabilités, temps de calcul, qualité des entrées et reproductibilité des résultats.

Avant la campagne : exclusion des clients non éligibles, vérification du plafond de 300 contacts, absence de doublons, respect des règles juridiques et traçabilité des décisions.

Après la campagne : taux de contact effectif, acceptations, résiliations, coût, marge et mesure de l'effet incrémental.

Je conserverais également une politique de repli : si le nouveau modèle échoue à la validation, le système peut réutiliser une version précédemment approuvée, à condition que ses données d'entrée restent compatibles.

### 12.3 Le calendrier opérationnel mérite une attention particulière

L'énoncé indique que le modèle est réentraîné et scoré le dernier jour du mois.

Il faut clarifier l'ordre des opérations :

- À la date \\(t\\), seules les observations dont les labels sont disponibles peuvent servir à l'entraînement.
- Les features de scoring doivent correspondre à l'état du client à la date \\(t\\).
- Le modèle doit produire la liste des clients à contacter.
- Les contacts et leurs résultats doivent être historisés.
- Les labels et résultats de campagne seront intégrés lorsqu'ils deviendront disponibles.

C'est une logique de point-in-time correctness.

Un autre sujet important est le changement de distribution des données provoqué par les campagnes elles-mêmes. Une campagne efficace modifie les comportements futurs : le churn observé chez les clients traités n'est plus nécessairement leur churn naturel sans intervention.

Cela doit être pris en compte lors des réentraînements.

# 13. Les dix pièges que j'identifierais lors de la revue technique

| Piège                                                 | Gravité  | Approche attendue                      |
| ----------------------------------------------------- | -------- | -------------------------------------- |
| Split aléatoire sur des données temporelles           | Critique | Validation chronologique               |
| Utilisation de features futures                       | Critique | Audit de disponibilité des données     |
| Confondre risque de churn et effet du contact         | Critique | Distinguer prédiction et causalité     |
| Présenter le pilote comme randomisé                   | Critique | Analyser le biais de sélection         |
| Classer uniquement par probabilité de churn           | Élevée   | Optimiser le gain économique           |
| Ignorer la calibration                                | Élevée   | Évaluer les probabilités               |
| Présenter un gain attendu comme un gain causal prouvé | Critique | Expliciter les hypothèses              |
| Retirer l'âge sans analyser les proxies               | Élevée   | Audit juridique et de biais            |
| Ignorer la dépendance entre mois d'un même client     | Élevée   | Validation et incertitude adaptées     |
| Surdimensionner l'architecture MLOps                  | Modérée  | Privilégier une solution proportionnée |

# 14. Comment je structurerais le projet GitHub

Je proposerais un dépôt nommé `telco-retention-optimization`, suffisamment explicite pour un recruteur et suffisamment professionnel pour un portfolio.

```
telco-retention-optimization/
│
├── README.md
├── pyproject.toml
├── .env.example
├── Dockerfile
├── Makefile
│
├── configs/
│   ├── model.yaml
│   ├── campaign.yaml
│   └── validation.yaml
│
├── data/
│   ├── raw/
│   │   ├── churn_clients.csv
│   │   └── pilote_campagne_2023.csv
│   ├── processed/
│   └── outputs/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_temporal_validation.ipynb
│   ├── 03_modeling_calibration.ipynb
│   ├── 04_business_optimization.ipynb
│   └── 05_pilot_causal_analysis.ipynb
│
├── src/
│   └── telco_retention/
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
│       ├── monitoring/
│       │   └── drift.py
│       └── pipelines/
│           ├── train.py
│           └── score.py
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_temporal_leakage.py
│   ├── test_economics.py
│   └── test_targeting.py
│
├── reports/
│   ├── figures/
│   └── technical_report.md
│
└── docs/
    ├── assumptions.md
    ├── architecture.md
    └── production_plan.md
```

Pour un test technique, je ne chercherais pas à implémenter toutes les fonctionnalités possibles.

Je privilégierais un pipeline complet, même simple, avec des tests sur les points les plus sensibles.

Par exemple, les tests unitaires de la fonction économique devraient vérifier les scénarios de churn, de non-churn, les seuils de rentabilité et la limite de capacité.

# 15. Mes recommandations pour réussir ce test

Je répartirais l'effort de la manière suivante.

Données et validation temporelle

20 %

Modélisation et calibration

20 %

Optimisation économique

25 %

Pilote et analyse causale

15 %

MLOps, légal et restitution

20 %

Répartition indicative de l'effort de préparation, et non barème officiel de l'entreprise.

Je proposerais une progression en quatre phases :

1. Comprendre et auditer les données : granularité, qualité, variables, déséquilibre de classes, chronologie et faisabilité des validations.
2. Construire un modèle prédictif fiable : baseline, gradient boosting, calibration et évaluation temporelle.
3. Construire la décision métier : calcul du gain, sélection des 300 clients, comparaison de politiques et analyse d'incertitude.
4. Analyser le pilote et préparer l'industrialisation : limites causales, protocole expérimental, conformité et pipeline MLOps.

## 16. Les arbitrages que je présenterais explicitement au recruteur

Puisque le test indique que le sujet est volontairement large, je préciserais les décisions suivantes.

| Ce que je réaliserais                      | Ce que je laisserais en perspective           |
| ------------------------------------------ | --------------------------------------------- |
| Régression logistique et gradient boosting | Deep learning complexe                        |
| Validation temporelle rigoureuse           | Recherche exhaustive d'hyperparamètres        |
| Calibration et optimisation du gain        | Optimisation multi-offres                     |
| Analyse observationnelle du pilote         | Uplift causal affirmé sans identification     |
| Simulation de scénarios économiques        | Garantie de gain réel sans expérimentation    |
| Pipeline batch reproductible               | Infrastructure Kubernetes complète            |
| Tests unitaires et documentation           | Déploiement cloud opérationnel si non demandé |

Ce tableau permet de montrer que les omissions sont des choix d'ingénierie justifiés, et non des oublis.

## 17. La conclusion que je défendrais pendant l'entretien

Je présenterais le projet en trois niveaux.

Premier niveau — Prédiction : nous estimons la probabilité qu'un client résilie dans les 30 jours suivants, avec une évaluation temporelle et une attention particulière à la calibration.

Deuxième niveau — Décision : nous transformons cette probabilité en gain économique espéré, puis sélectionnons les clients les plus rentables sous la contrainte de 300 contacts mensuels.

Troisième niveau — Validation de l'impact : nous reconnaissons que les gains simulés reposent sur des hypothèses non encore démontrées. Nous proposons donc une expérimentation contrôlée pour mesurer l'effet réel des contacts et, à terme, optimiser le profit incrémental.

C'est cette articulation entre Machine Learning, optimisation économique et inférence causale qui donnerait de la profondeur à ta présentation.

Il reste toutefois une limite importante : nous avons analysé l'énoncé, mais pas encore les données réelles. Nous ne connaissons donc ni les colonnes exactes, ni la distribution du churn, ni le nombre de clients par mois, ni la qualité des observations, ni les caractéristiques du pilote.

La prochaine étape que je recommande est d'analyser réellement `churn_clients.csv` et `pilote_campagne_2023.csv`. Envoie-moi ces deux fichiers. Nous commencerons par un audit Data Understanding rigoureux, puis nous déterminerons la stratégie de validation, les modèles et la politique de ciblage les plus adaptés à leurs caractéristiques réelles.