# Diagnostic historique de la sélection groupée

**Ancienne méthode :** les ablations et l’optimisation de seuil décrites dans cet
historique ont depuis été retirées de la copie groupée. Celle-ci utilise désormais
un choix explicite des colonnes avant entraînement et un seuil fixe de 0,5.


Lecture des sorties enregistrées par l’utilisateur avant la modification ; aucun
nouvel entraînement n’a été exécuté pour ce diagnostic.

La CV avait entraîné les quatre folds. Les quatre candidats proposés par la
permutation ont tous été testés puis refusés par les critères d’ablation :

| Retrait | Plus forte perte AP d’un fold | Variation AP globale | Variation Logloss |
| --- | ---: | ---: | ---: |
| nps | 0.004510 | −0.003207 | +0.004225 |
| data_gb | 0.008384 | −0.006048 | +0.006851 |
| channel | 0.001722 | −0.000406 | +0.000188 |
| late_payments | 0.002383 | +0.000448 | −0.000723 |

La tolérance était de 0.001 et devait être respectée dans chaque fold. Même
`channel` et `late_payments`, acceptables sur les critères globaux, échouaient
sur le pire fold. La référence était donc conservée, avec les mêmes huit colonnes
et les mêmes modèles CV. La seconde ligne ne représentait pas un modèle réduit.

Les modifications ajoutent : 500 itérations maximum, MCC dans les scores,
journal détaillé des refus, liste réelle des colonnes entraînées, contrôle du
budget de référence et affichage explicite d’une sélection sans retrait. Les
seuils de retrait ne sont pas assouplis pour forcer une baisse du nombre de colonnes.
Un paramètre explicite permet de définir le nombre de folds stables requis avant
les essais. Les visuels gardent leur code ; leurs anciennes sorties sont vidées.

Les chiffres ci-dessus proviennent de l’ancienne exécution, à 1 800 itérations
maximum. Ils ne préjugent pas des résultats de la nouvelle version à 500 itérations.
Le notebook, les nouveaux tests et les modèles n’ont pas été exécutés après cette
modification. Redémarrer le noyau puis relancer depuis le début est nécessaire.
