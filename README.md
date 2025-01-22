# Projet de détection de fatigue basée sur l'EEG

Ce projet vise à développer un modèle de détection de fatigue en utilisant des signaux EEG (électroencéphalographie) et des données PERCLOS (Percentage of Eye Closure). Le modèle utilise un **Random Forest Regressor** pour prédire le niveau de fatigue à partir des caractéristiques extraites des signaux EEG.

---

## Table des matières
1. [Objectif du projet](#objectif-du-projet)
2. [Données utilisées](#données-utilisées)
3. [Prérequis](#prérequis)
4. [Organisation du projet](#organisation-du-projet)
5. [Prétraitement des données](#prétraitement-des-données)
6. [Extraction des caractéristiques](#extraction-des-caractéristiques)
7. [Matrice de Pearson](#matrice-de-pearson)
8. [Modélisation](#modélisation)
9. [Résultats](#résultats)
10. [Comparaison avec l'article](#comparaison-avec-larticle)
11. [Améliorations possibles](#améliorations-possibles)
12. [Instructions pour exécuter le code](#instructions-pour-exécuter-le-code)
13. [Explication des scripts](#explication-des-scripts)
14. [Auteurs](#auteurs)

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

### Pourquoi utiliser des fichiers NPY et NPZ ?
- **NPY** : Format binaire pour stocker des tableaux NumPy. Il est rapide à charger et efficace pour stocker des données numériques.
- **NPZ** : Format compressé pour stocker plusieurs tableaux NumPy dans un seul fichier. Il est utile pour regrouper des données associées, comme les caractéristiques DE et les métadonnées.

---

## Prérequis

Pour exécuter ce projet, les éléments suivants sont nécessaires :
- **Python 3.x**
- **Bibliothèques Python** :
  - `numpy`, `scipy`, `scikit-learn`, `mne`, `matplotlib`, `logging`

**Installation des dépendances** :
```bash
pip install numpy scipy scikit-learn mne matplotlib
```

---

## Organisation du projet

Le projet est organisé comme suit :
```
BCI/
│
├── README.md                  # Documentation principale du projet
├── Visuels/                   # Dossier contenant les images et graphiques générés
│   ├── alpha.png              # Visualisation de la bande de fréquence alpha
│   ├── avant_notch.png        # Signal EEG avant application du filtre notch
│   ├── apres_notch.png        # Signal EEG après application du filtre notch
│   ├── beta.png               # Visualisation de la bande de fréquence beta
│   ├── delta.png              # Visualisation de la bande de fréquence delta
│   ├── gamma.png              # Visualisation de la bande de fréquence gamma
│   ├── theta.png              # Visualisation de la bande de fréquence theta
│   ├── pearson_matrix.png     # Matrice de Pearson des corrélations
│   ├── random_forest_1.png    # Graphique des performances du modèle (1/2)
│   └── random_forest_2.png    # Graphique des performances du modèle (2/2)
│
├── article.pdf                # Article de référence pour le projet
├── Codes/                     # Dossier contenant les scripts Python
│   ├── check_fft_de.py        # Vérification des résultats DE
│   ├── check_preprocess.py    # Vérification des données prétraitées
│   ├── fft_de.py              # Extraction des caractéristiques DE
│   ├── matrice.py             # Calcul de la matrice de Pearson
│   ├── preprocessing.py       # Prétraitement des données EEG et PERCLOS
│   └── random_forest.py       # Entraînement et évaluation du modèle Random Forest
│
├── VLA_VRW/                   # Dossier contenant la base de données (BDD)
│   ├── README.txt             # Documentation spécifique à la BDD
│   ├── lab/                   # Données de laboratoire
│   │   ├── EEG/               # Fichiers EEG (.edf) pour chaque sujet
│   │   └── perclos/           # Fichiers PERCLOS (.mat) pour chaque sujet
│   └── real/                  # Données réelles (structure similaire à lab/)
│
└── sortie_preprocess/         # Dossier de sortie pour les données prétraitées
```

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

## Matrice de Pearson

La **matrice de Pearson** est utilisée pour mesurer les relations linéaires entre les caractéristiques (bandes de fréquence EEG) et la cible (PERCLOS). Elle est calculée **après le prétraitement des données** et **avant la modélisation** pour :

1. **Comprendre les données** :
   - Identifier les relations entre les variables.
   - Détecter des problèmes comme la multicollinéarité.

2. **Guider la modélisation** :
   - Sélectionner les caractéristiques les plus pertinentes pour la prédiction.
   - Choisir un modèle adapté (par exemple, un modèle non linéaire si les corrélations sont faibles).

### **Résultats de la matrice de Pearson**
Voici un exemple de visualisation de la matrice de Pearson :

![Matrice de Pearson](Visuels/pearson_matrix.png)

---

## Modélisation

Le modèle utilisé est un **Random Forest Regressor** avec les hyperparamètres suivants :
- **n_estimators** : 300
- **max_depth** : 20
- **min_samples_split** : 10
- **min_samples_leaf** : 2

Voici une explication détaillée du **Random Forest**, de son utilisation dans votre projet, des hyperparamètres choisis, et des données d'entraînement et de test.

---

## **Random Forest : Explication détaillée**

### **1. Pourquoi utiliser un Random Forest ?**

Le **Random Forest** est un modèle d'apprentissage automatique basé sur l'**ensemblage d'arbres de décision**. Voici pourquoi il est bien adapté à votre projet :

#### **a. Robustesse**
- Le Random Forest est **peu sensible au surajustement (overfitting)** grâce à l'ensemblage de plusieurs arbres de décision. Chaque arbre est entraîné sur un sous-ensemble aléatoire des données, ce qui réduit le risque de surajustement.

#### **b. Non-linéarité**
- Les signaux EEG ont des relations **complexes et non linéaires** avec la fatigue (PERCLOS). Le Random Forest peut capturer ces relations grâce à sa structure d'arbre de décision.

#### **c. Importance des caractéristiques**
- Le Random Forest permet de calculer l'**importance des caractéristiques**, ce qui vous aide à comprendre quelles bandes de fréquence EEG sont les plus pertinentes pour prédire la fatigue.

#### **d. Facilité d'utilisation**
- Le Random Forest nécessite peu de prétraitement des données (par exemple, pas besoin de normalisation stricte) et est facile à implémenter avec des bibliothèques comme `scikit-learn`.

---

### **2. Comment fonctionne un Random Forest ?**

Le Random Forest est un **ensemble d'arbres de décision**. Voici comment il fonctionne :

#### **a. Construction des arbres**
1. **Bootstrap** :
   - Pour chaque arbre, un sous-ensemble aléatoire des données d'entraînement est sélectionné (avec remise). Cela signifie que certaines données peuvent être utilisées plusieurs fois, tandis que d'autres ne sont pas utilisées.

2. **Sélection aléatoire des caractéristiques** :
   - À chaque division d'un nœud, un sous-ensemble aléatoire des caractéristiques est considéré. Cela garantit que les arbres sont diversifiés.

3. **Construction de l'arbre** :
   - Chaque arbre est construit en divisant récursivement les données en sous-ensembles basés sur les caractéristiques. Le critère de division est généralement l'**impureté de Gini** ou l'**entropie**.

#### **b. Prédiction**
- Pour la prédiction, chaque arbre du Random Forest donne une prédiction. La prédiction finale est la **moyenne** des prédictions de tous les arbres (pour la régression) ou le **vote majoritaire** (pour la classification).

#### **c. Importance des caractéristiques**
- L'importance d'une caractéristique est calculée en mesurant à quel point elle réduit l'impureté (Gini ou entropie) dans l'ensemble des arbres. Les caractéristiques qui réduisent le plus l'impureté sont considérées comme les plus importantes.

---

### **3. Hyperparamètres du Random Forest**

Voici les hyperparamètres que vous avez choisis et leur justification :

#### **a. `n_estimators` : 300**
- **Description** : Nombre d'arbres dans la forêt.
- **Pourquoi 300 ?** :
  - Un nombre élevé d'arbres améliore la stabilité et la précision du modèle.
  - 300 est un bon compromis entre performance et temps de calcul.

#### **b. `max_depth` : 20**
- **Description** : Profondeur maximale de chaque arbre.
- **Pourquoi 20 ?** :
  - Une profondeur maximale permet de limiter la complexité des arbres et d'éviter le surajustement.
  - 20 est suffisamment profond pour capturer des relations complexes sans surajuster.

#### **c. `min_samples_split` : 10**
- **Description** : Nombre minimum d'échantillons requis pour diviser un nœud.
- **Pourquoi 10 ?** :
  - Cela garantit que les divisions sont basées sur suffisamment de données, ce qui réduit le surajustement.

#### **d. `min_samples_leaf` : 2**
- **Description** : Nombre minimum d'échantillons requis pour être une feuille.
- **Pourquoi 2 ?** :
  - Cela évite des feuilles trop petites, ce qui réduit le surajustement et améliore la généralisation.

#### **e. `random_state` : 42**
- **Description** : Graine aléatoire pour garantir la reproductibilité.
- **Pourquoi 42 ?** :
  - C'est une valeur arbitraire couramment utilisée pour garantir que les résultats sont reproductibles.

---

### **4. Données d'entraînement et de test**

#### **a. Division des données**
- Les données sont divisées en deux ensembles :
  - **Ensemble d'entraînement** : 80 % des données.
  - **Ensemble de test** : 20 % des données.
- Cette division est faite de manière aléatoire mais stratifiée pour garantir que les deux ensembles ont une distribution similaire de PERCLOS.

#### **b. Caractéristiques utilisées**
- Les caractéristiques d'entraînement incluent :
  - Les bandes de fréquence EEG (delta, theta, alpha, beta, gamma).
  - Les caractéristiques supplémentaires (variance et énergie).
- La cible est la valeur PERCLOS.

#### **c. Normalisation**
- Les caractéristiques sont normalisées (moyenne = 0, écart-type = 1) pour améliorer la convergence du modèle.

---

### **5. Évaluation du modèle**

Le modèle est évalué sur l'ensemble de test en utilisant les métriques suivantes :

#### **a. RMSE (Root Mean Squared Error)**
- Mesure l'écart moyen entre les prédictions et les valeurs réelles.
- **Interprétation** : Plus le RMSE est faible, plus les prédictions sont précises.

#### **b. MAE (Mean Absolute Error)**
- Mesure l'erreur absolue moyenne.
- **Interprétation** : Moins sensible aux erreurs importantes que le RMSE.

#### **c. R² (Coefficient de détermination)**
- Mesure la proportion de la variance des données expliquée par le modèle.
- **Interprétation** : Un R² proche de 1 signifie que le modèle explique bien la variance des données.

#### **d. PCC (Pearson Correlation Coefficient)**
- Mesure la corrélation linéaire entre les prédictions et les valeurs réelles.
- **Interprétation** : Un PCC proche de 1 indique une forte corrélation.

---

### **6. Résultats du modèle**

Voici les performances de votre modèle Random Forest :
- **RMSE** : **0.1468**
- **MAE** : **0.1046**
- **R²** : **0.5390**
- **PCC** : **0.7342**

#### **Interprétation**
- Le **RMSE** et le **MAE** montrent que les prédictions sont relativement proches des valeurs réelles.
- Le **R²** indique que le modèle explique environ 54 % de la variance des données.
- Le **PCC** montre une forte corrélation entre les prédictions et les valeurs réelles.

---

## Comparaison avec l'article

Les résultats de l'article de référence sont les suivants :
- **PCC** : **0.6636 ± 0.1321**
- **RMSE** : **0.1365 ± 0.0689**

### **Analyse des différences**
- **PCC** : Notre **PCC de 0.7342** est **supérieur** à celui de l'article (**0.6636**). Cela signifie que notre modèle a une corrélation plus forte entre les prédictions et les valeurs réelles.
- **RMSE** : Notre **RMSE de 0.1468** est **légèrement supérieur** à celui de l'article (**0.1365**). Cela signifie que les prédictions de notre modèle sont légèrement moins précises que celles de l'article.

### **Pourquoi nos résultats sont-ils meilleurs ou moins bons ?**
1. **PCC supérieur** :
   - Nous avons utilisé des caractéristiques supplémentaires (comme la variance et l'énergie) qui capturent mieux les informations pertinentes sur la fatigue.
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
- **`pearson_matrix.py`** : Calcul de la matrice de Pearson.
- **`random_forest.py`** : Entraînement et évaluation du modèle Random Forest.
- **`check_preprocess.py`** : Vérification des données prétraitées.
- **`check_fft_de.py`** : Vérification des résultats DE.

### Exécution du code
1. Placez les fichiers de données dans les dossiers `sortie_preprocess/lab` et `sortie_preprocess/real`.
2. Exécutez les scripts dans l'ordre suivant :
   ```bash
   python scripts/preprocessing.py
   python scripts/fft_de.py
   python scripts/pearson_matrix.py
   python scripts/random_forest.py
   python scripts/check_preprocess.py
   python scripts/check_fft_de.py
   ```

---

## Explication des scripts

### **`preprocessing.py`**
- **Objectif** : Prétraite les données EEG et PERCLOS.
- **Fonctionnalités** :
  - Chargement des fichiers `.edf` (EEG) et `.mat` (PERCLOS).
  - Suppression des premières secondes pour éliminer le bruit initial.
  - Filtrage passe-bande et filtre notch pour nettoyer les signaux EEG.
  - Découpage en fenêtres de 2400 points.
  - Normalisation des données.

### **`fft_de.py`**
- **Objectif** : Extrait les caractéristiques DE (Differential Entropy) des signaux EEG.
- **Fonctionnalités** :
  - Applique un filtre passe-bande pour chaque bande de fréquence (delta, theta, alpha, beta, gamma).
  - Calcule la DE pour chaque bande de fréquence.
  - Ajoute des caractéristiques supplémentaires (variance et énergie).

### **`pearson_matrix.py`**
- **Objectif** : Calcule et visualise la matrice de Pearson.
- **Fonctionnalités** :
  - Calcule les corrélations entre les bandes de fréquence EEG et PERCLOS.
  - Génère une heatmap de la matrice de Pearson.
  - Sauvegarde l'image dans le dossier `Visuels/`.

### **`random_forest.py`**
- **Objectif** : Entraîne et évalue un modèle Random Forest Regressor.
- **Fonctionnalités** :
  - Charge les données prétraitées et les caractéristiques DE.
  - Divise les données en ensembles d'entraînement et de test.
  - Optimise les hyperparamètres avec GridSearchCV.
  - Évalue les performances du modèle (RMSE, MAE, R², PCC).

### **`check_preprocess.py`**
- **Objectif** : Vérifie l'intégrité des données prétraitées.
- **Fonctionnalités** :
  - Vérifie que les fichiers prétraités existent.
  - Vérifie que les dimensions des données EEG et PERCLOS sont cohérentes.

### **`check_fft_de.py`**
- **Objectif** : Vérifie les résultats DE.
- **Fonctionnalités** :
  - Charge les résultats DE.
  - Vérifie l'absence de valeurs NaN ou infinies.
  - Visualise les résultats DE sous forme de graphiques.

---

## Auteurs
- **ALEMANY Clarisse**
- **ASSOUANE Inès**
- **BOUKHEDRA Khitam**
