IL MANQUE EXPLICATION DE LA FFT, BANDE PASSANTE DOIT ETRE MIS AVANT LE DE
EXPLICATION DE COMMENT LANCER LE CODE
EXPLICATION DE CHECK_FFT ET CHECK_PREPROCESS
PREREQUIS
DONNÉES (EEG ET PERCLOS)
DONNER LE NOM DES CODES
ORGANISATION DU PROJET (AVEC LE DATASET)
POURQUOI NPY et NPZ ?
# Projet de détection de fatigue basée sur l'EEG

Ce projet vise à développer un modèle de détection de fatigue basé sur les signaux EEG (électroencéphalographie) en utilisant des données de laboratoire et de scénarios réels. Le modèle utilise un **Random Forest Regressor** pour prédire la fatigue mesurée par l'indice PERCLOS.

---

## Table des matières
1. [Objectif du projet](#objectif-du-projet)
2. [Données utilisées](#données-utilisées)
3. [Prétraitement des données](#prétraitement-des-données)
4. [Extraction des caractéristiques](#extraction-des-caractéristiques)
5. [Modélisation](#modélisation)
6. [Résultats](#résultats)
7. [Comparaison avec l'article](#comparaison-avec-larticle)
8. [Améliorations possibles](#améliorations-possibles)
9. [Instructions pour exécuter le code](#instructions-pour-exécuter-le-code)

---

## Objectif du projet

L'objectif de ce projet est de prédire la fatigue (mesurée par l'indice PERCLOS) à partir des signaux EEG. Nous avons utilisé des données provenant de deux scénarios :
- **Données de laboratoire** : Données EEG collectées dans un environnement contrôlé.
- **Données réelles** : Données EEG collectées dans des conditions réelles de conduite.

Le modèle développé est un **Random Forest Regressor**, et nous avons évalué ses performances en utilisant les métriques suivantes :
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² (Coefficient de détermination)**
- **PCC (Pearson Correlation Coefficient)**

---

## Données utilisées

Les données utilisées dans ce projet sont les suivantes :
- **Fichiers EEG** : `{sujet}_eeg_windows.npy` (fenêtres de données EEG).
- **Caractéristiques DE** : `{sujet}_de_results.npz` (Differential Entropy pour les bandes de fréquence delta, theta, alpha, beta, gamma).
- **Étiquettes PERCLOS** : `{sujet}_perclos.npy` (valeurs de fatigue mesurées par PERCLOS).

Les données sont divisées en deux ensembles :
- **Laboratoire** : 20 sujets.
- **Réel** : 14 sujets.

---

## Prétraitement des données

Le prétraitement des données comprend les étapes suivantes :

1. **Chargement des données** :
   - Les fichiers `.edf` (EEG) et `.mat` (PERCLOS) sont chargés pour chaque sujet.
   - Les données EEG sont extraites et les étiquettes PERCLOS sont alignées avec les données EEG.

2. **Suppression des premières secondes** :
   - Les premières 24 secondes des données EEG sont supprimées pour éliminer le bruit initial.
   - **Pourquoi ?** Les premières secondes des enregistrements EEG contiennent souvent du bruit dû à l'initialisation du système ou à l'adaptation du sujet à l'environnement expérimental.

3. **Filtrage passe-bande** :
   - Un filtre passe-bande (0.1 Hz - 70 Hz) est appliqué pour éliminer les fréquences indésirables.
   - **Pourquoi ?** Les signaux EEG utiles se situent généralement dans cette plage de fréquences. Les fréquences en dehors de cette plage (comme le bruit de ligne à 50/60 Hz) sont filtrées pour améliorer la qualité des données.

4. **Filtre notch** :
   - Un filtre notch est appliqué pour supprimer les interférences de la fréquence du secteur (50 Hz ou 60 Hz).
   - **Pourquoi ?** Le bruit de ligne à 50 Hz ou 60 Hz est courant dans les enregistrements EEG et peut masquer les signaux physiologiques. Le filtre notch permet de supprimer cette interférence sans affecter les autres fréquences.

5. **Découpage en fenêtres** :
   - Les données EEG sont découpées en fenêtres de 2400 points.
   - **Pourquoi ?** Une fenêtre de 2400 points correspond à environ 8 secondes de données EEG (à une fréquence d'échantillonnage de 300 Hz). Cette taille est suffisamment grande pour capturer des motifs significatifs dans les signaux EEG, tout en étant suffisamment petite pour permettre une analyse temporelle fine. C'est un compromis entre résolution et stabilité.

6. **Normalisation** :
   - Les données sont normalisées pour améliorer la stabilité du modèle.
   - **Pourquoi ?** La normalisation permet de mettre toutes les caractéristiques à la même échelle, ce qui améliore la convergence et les performances du modèle.

---

## Extraction des caractéristiques

Les caractéristiques suivantes sont extraites pour chaque fenêtre EEG :

1. **Bandes de fréquence** :
   - **Differential Entropy (DE)** est calculée pour les bandes de fréquence delta, theta, alpha, beta, gamma.
   - **Pourquoi la DE ?** La Differential Entropy est une mesure de l'information contenue dans un signal continu. Elle est souvent utilisée pour analyser les signaux EEG car elle capture bien la complexité des signaux. Elle est particulièrement utile pour les bandes de fréquence spécifiques, car elle résume les propriétés spectrales des signaux EEG.

2. **Filtrage passe-bande dans `fft_de.py`** :
   - Un filtre passe-bande est appliqué à nouveau pour chaque bande de fréquence (delta, theta, alpha, beta, gamma) avant de calculer la DE.
   - **Pourquoi ?** Le filtre passe-bande permet d'isoler les fréquences spécifiques à chaque bande, ce qui est essentiel pour calculer la DE de manière précise. Cela garantit que la DE reflète uniquement l'information contenue dans la bande de fréquence cible.

3. **Caractéristiques supplémentaires** :
   - **Variance** : Mesure de la variabilité du signal.
   - **Énergie** : Intégrale du carré du signal.
   - **Pourquoi ces caractéristiques ?** La variance et l'énergie fournissent des informations supplémentaires sur la dynamique du signal EEG, ce qui peut aider le modèle à mieux prédire la fatigue.

---

## Modélisation

Le modèle utilisé est un **Random Forest Regressor** avec les hyperparamètres suivants :
- **n_estimators** : 300
- **max_depth** : 20
- **min_samples_split** : 10
- **min_samples_leaf** : 2

### **Pourquoi un Random Forest Regressor ?**
- **Robustesse** : Le Random Forest est un modèle robuste qui résiste au surajustement (overfitting) grâce à l'ensemblage d'arbres de décision.
- **Non-linéarité** : Le Random Forest peut capturer des relations non linéaires entre les caractéristiques et la cible, ce qui est important pour les données EEG complexes.
- **Importance des caractéristiques** : Le Random Forest permet d'évaluer l'importance des caractéristiques, ce qui aide à comprendre quelles bandes de fréquence sont les plus pertinentes pour la prédiction.

### **Choix des hyperparamètres**
- **n_estimators** : 300 arbres ont été choisis pour équilibrer la performance et le temps de calcul.
- **max_depth** : Une profondeur maximale de 20 permet de capturer des relations complexes sans surajuster.
- **min_samples_split** : Un seuil de 10 échantillons pour diviser un nœud garantit que les divisions sont basées sur suffisamment de données.
- **min_samples_leaf** : Un seuil de 2 échantillons par feuille permet d'éviter des feuilles trop petites, ce qui réduit le surajustement.

### **Optimisation des hyperparamètres**
- Nous avons utilisé **GridSearchCV** pour optimiser les hyperparamètres du modèle.
- **Pourquoi ?** GridSearchCV permet de tester systématiquement différentes combinaisons d'hyperparamètres pour trouver celles qui maximisent les performances du modèle.

---

## Résultats

Les performances du modèle sont les suivantes :
- **RMSE** : **0.1468**
- **MAE** : **0.1046**
- **R²** : **0.5390**
- **PCC** : **0.7342**

### **Pourquoi ces métriques ?**
- **RMSE (Root Mean Squared Error)** : Cette métrique mesure l'écart moyen entre les prédictions et les valeurs réelles. Elle est sensible aux erreurs importantes, ce qui en fait un bon indicateur de la précision du modèle.
- **MAE (Mean Absolute Error)** : Cette métrique mesure l'erreur absolue moyenne. Elle est moins sensible aux erreurs importantes que le RMSE, ce qui permet d'évaluer la performance globale du modèle.
- **R² (Coefficient de détermination)** : Cette métrique indique la proportion de la variance des données expliquée par le modèle. Un R² proche de 1 signifie que le modèle explique bien la variance des données.
- **PCC (Pearson Correlation Coefficient)** : Cette métrique mesure la corrélation linéaire entre les prédictions et les valeurs réelles. Un PCC élevé indique une forte corrélation, ce qui est souhaitable pour un modèle de prédiction.

### Importance des caractéristiques
- **alpha** : 0.3575 (la plus importante)
- **gamma** : 0.2372
- **theta** : 0.1162
- **beta** : 0.1157
- **variance** : 0.0697
- **energy** : 0.0683
- **delta** : 0.0353 (la moins importante)

---

## Comparaison avec l'article

Les résultats de l'article sont les suivants :
- **PCC** : **0.6636 ± 0.1321**
- **RMSE** : **0.1365 ± 0.0689**

### **Analyse des différences**
- **PCC** : Notre **PCC de 0.7342** est **supérieur** à celui de l'article (**0.6636**). Cela signifie que notre modèle a une corrélation plus forte entre les prédictions et les valeurs réelles.
- **RMSE** : Notre **RMSE de 0.1468** est **légèrement supérieur** à celui de l'article (**0.1365**). Cela signifie que les prédictions de notre modèle sont légèrement moins précises que celles de l'article.

### **Pourquoi nos résultats sont-ils meilleurs ou moins bons ?**
1. **PCC supérieur** :
   - Nous avez peut-être utilisé des caractéristiques supplémentaires (comme la variance et l'énergie) qui capturent mieux les informations pertinentes sur la fatigue.
   - Notre modèle Random Forest est bien optimisé et capture mieux les relations non linéaires dans les données.

2. **RMSE légèrement supérieur** :
   - L'article utilise une méthode de domain adaptation (**CS2DA**) pour atténuer les problèmes de décalage de domaine entre les données de laboratoire et de scénarios réels. Cela pourrait expliquer leur meilleur RMSE.
   - Notre modèle pourrait bénéficier de l'ajout d'une méthode de domain adaptation pour réduire le RMSE.

---

## Améliorations possibles

Pour améliorer les performances du modèle et surpasser les résultats de l'article, voici quelques suggestions :
1. **Implémenter une méthode de fusion** : Utiliser une règle de fusion (Max rule) pour combiner les prédictions de plusieurs modèles.
2. **Ajouter des caractéristiques supplémentaires** : Intégrer des mesures comme l'asymétrie frontale ou la cohérence inter-hémisphérique.
3. **Améliorer le prétraitement** : Appliquer des techniques de filtrage des artefacts et de normalisation spécifique.
4. **Utiliser des modèles plus sophistiqués** : Tester des modèles comme XGBoost, LightGBM, ou des réseaux de neurones.
5. **Implémenter une méthode de domain adaptation** : Atténuer les problèmes de décalage de domaine entre les données de laboratoire et de scénarios réels.

---

## Instructions pour exécuter le code

### Prérequis
- Python 3.x
- Bibliothèques Python : `numpy`, `scipy`, `scikit-learn`, `mne`, `matplotlib`, `logging`

### Installation des dépendances
```bash
pip install numpy scipy scikit-learn mne matplotlib
```

### Structure des fichiers
- **`preprocessing.py`** : Prétraitement des données EEG et PERCLOS.
- **`fft_de.py`** : Extraction des caractéristiques DE (Differential Entropy).
- **`random_forest.py`** : Entraînement et évaluation du modèle Random Forest.

### Exécution du code
1. Placez les fichiers de données dans les dossiers `sortie_preprocess/lab` et `sortie_preprocess/real`.
2. Exécutez les scripts dans l'ordre suivant :
   ```bash
   python preprocessing.py
   python fft_de.py
   python random_forest.py
   ```

---

## Auteurs
- ALEMANY Clarisse
- ASSOUANE Inès
- BOUKHEDRA Khitam