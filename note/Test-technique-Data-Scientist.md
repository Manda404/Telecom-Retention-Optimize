# Test technique Data Scientist

Ce test servira de **support de discussion** lors de notre entretien.

- Vous êtes libre du format, de la structure du projet et des outils (Python recommandé).
- Le sujet est volontairement large : faites des arbitrages et indiquez explicitement ce que vous avez laissé de côté.

## Contenu fourni

| Fichier | Description |
|---|---|
| `data/churn_clients.csv` | Données de la partie 1 (churn télécom) |
| `data/pilote_campagne_2023.csv` | Données du pilote de novembre 2023 (partie 1) |

## Partie 1 : prédiction de churn et décision de campagne (cœur du test)

### Contexte
Un opérateur télécom souhaite identifier, à chaque fin de mois, les clients qui résilieront dans les 30 jours suivants (`churn`), afin de cibler une campagne de rétention.

Économie de la campagne (par client contacté) :
- coût de contact : 10 € ;
- un client qui allait partir accepte l'offre avec une probabilité de 30 % ; il reste alors 12 mois, ce qui rapporte une marge de 40 % de son `monthly_fee` par mois ;
- l'offre est une remise de 30 % pendant 3 mois (coût = 0,9 × `monthly_fee`), accordée à tout client qui l'accepte, y compris ceux qui ne seraient pas partis (effet d'aubaine : 10 % d'entre eux l'acceptent) ;
- capacité : les conseillers ne peuvent contacter que 300 clients par mois.

### Données
Table client × mois (2024). Chaque ligne décrit l'état d'un client à une fin de mois (`snapshot_date`) ; `churn` vaut 1 s'il résilie dans les 30 jours qui suivent. Un client résilié n'apparaît plus ensuite.

### Contraintes et informations complémentaires
- Les labels ne sont connus que 30 jours après l'instantané. Le modèle est réentraîné chaque mois et scoré le dernier jour de chaque mois.
- Une campagne pilote a été menée en novembre 2023 (`data/pilote_campagne_2023.csv`, hors du jeu principal) : 4 000 clients, certains contactés (`contacted`) et d'autres non, tirage non documenté. `churn_30d` est le résultat à 30 jours. Le taux d'acceptation de 30 % provient d'une enquête interne et n'a jamais été mesuré sur ce pilote.
- Le service juridique impose que la décision soit explicable au client et interdit l'utilisation directe de l'âge dans la décision de ciblage.
- La direction souhaite une règle simple à transmettre aux conseillers (seuil, score, liste) et un chiffrage du gain annuel attendu avec son incertitude.
- Le parc réel compte 1,2 million de clients ; le scoring doit rester exécutable dans un délai raisonnable sur ce volume.

### Ce que nous attendons
1. **Compréhension des données** : explorez les données et justifiez vos choix de traitement.
2. **Modélisation** : construisez un modèle de prédiction du churn et justifiez votre démarche d'évaluation.
3. **Décision métier** : dites qui contacter chaque mois et estimez le gain annuel attendu, avec son incertitude.
4. **Contrainte légale** : expliquez comment vous respectez la contrainte juridique et ce qu'elle implique.
5. **Campagne de rétention** : discutez de la pertinence de votre approche pour mesurer et améliorer l'efficacité de la campagne, y compris au regard du pilote de 2023.
6. **Mise en production** : décrivez ce qu'il faudrait pour exploiter votre solution dans la durée.


## Partie 3 : scénarios d'évolution (analyse écrite, 1 page maximum par scénario)

Traitez au moins deux des scénarios suivants dans la note de synthèse, en précisant ce que vous changeriez dans votre solution :
1. Un concurrent lance une offre agressive en région parisienne : les performances chutent de 30 % en deux mois. Diagnostic, actions immédiates, actions de fond.
2. La direction veut aussi prédire la valeur future du client et proposer trois offres différentes selon le profil. Comment reformuler le problème ?
3. Le métier signale que les conseillers contactent surtout les clients qu'ils connaissent déjà et ignorent les autres. Quels impacts sur les données futures et quelles mesures ?
4. Un audit demande de prouver que le modèle ne discrimine pas certaines populations. Quelle démarche ?
5. Le budget est divisé par deux et la capacité passe à 150 contacts par mois. Que devient votre décision et comment le justifier ?

## Partie 4 : industrialisation (à intégrer dans votre projet)

Livrez la partie 1 comme un **projet Python réutilisable**, pas seulement comme un notebook. Il n'est pas demandé de tout industrialiser : choisissez le périmètre le plus pertinent et expliquez-le.

Bonus: suivi d'expériences, validation de schéma, intégration continue, conteneur, API, plan de monitoring.

## Livrables

Veuillez à ne pas publier les données, l'énoncer ou vos travaux (pas de dépôt en ligne, public ou privé, pas de service tiers de partage). Travaillez avec un dépôt Git local (`git init`) afin que l'historique de vos commits nous soit transmis, et envoyez-nous une archive `.zip` incluant le dossier `.git`, par e-mail ou via le lien sécurisé que nous vous indiquerons. Supprimez les données de vos environnements à la fin du processus.

L'archive doit contenir :

1. **Le projet**
2. **Une note de synthèse** (1 à 2 pages, dans le dépôt) : vos principaux constats, vos décisions, vos hypothèses, vos limites et ce que vous feriez avec plus de temps.
3. **Une présentation** (PowerPoint ou PDF) pour une restitution de 10 minutes : problème, démarche, choix clés et alternatives écartées, résultats, recommandation métier, risques et prochaines étapes.

## Déroulé de la partie test technique de l'entretien
1. Présentation de vos slides (10 min).
2. Discussion sur votre projet et vos choix, avec des questions sur des points précis du code et des variantes (« que se passe-t-il si… »).
3. Échanges sur la mise en production.

