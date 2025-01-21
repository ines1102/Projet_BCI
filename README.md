Voici une version améliorée du **README** où **chaque choix, code, métrique et étape est justifié** de manière détaillée. L'objectif est de rendre le projet compréhensible même pour des personnes qui n'ont pas lu l'article ou qui ne sont pas familières avec l'EEG, le PERCLOS ou le traitement du signal.

---

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
7. [Modélisation](#modélisation)
8. [Résultats](#résultats)
9. [Comparaison avec l'article](#comparaison-avec-larticle)
10. [Améliorations possibles](#améliorations-possibles)
11. [Instructions pour exécuter le code](#instructions-pour-exécuter-le-code)
12. [Explication des scripts](#explication-des-scripts)
13. [Auteurs](#auteurs)

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
projet-fatigue-eeg/
│
├── données/
│   ├── lab/                  # Données de laboratoire
│   │   ├── sujet1_eeg_windows.npy
│   │   ├── sujet1_de_results.npz
│   │   └── sujet1_perclos.npy
│   └── real/                 # Données réelles
│       ├── sujet21_eeg_windows.npy
│       ├── sujet21_de_results.npz
│       └── sujet21_perclos.npy
│
├── scripts/
│   ├── preprocessing.py      # Script de prétraitement des données
│   ├── fft_de.py             # Script d'extraction des caractéristiques DE
│   ├── random_forest.py      # Script d'entraînement et d'évaluation du modèle
│   ├── check_preprocess.py   # Script de vérification des données prétraitées
│   └── check_fft_de.py       # Script de vérification des résultats DE
│
├── sortie_preprocess/        # Dossier de sortie pour les données prétraitées
│   ├── lab/
│   └── real/
│
└── README.md                 # Documentation du projet
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
- **RMSE (Root Mean Squared Error)** : Cette métrique mesure l'écart moyen entre les prédictions et les valeurs réelles. Elle est sensible aux erreurs importantes, ce qui en fait un bon indicateur de la précision du modèle. Un RMSE faible signifie que les prédictions sont proches des valeurs réelles.
- **MAE (Mean Absolute Error)** : Cette métrique mesure l'erreur absolue moyenne. Elle est moins sensible aux erreurs importantes que le RMSE, ce qui permet d'évaluer la performance globale du modèle sans être influencé par des valeurs aberrantes.
- **R² (Coefficient de détermination)** : Cette métrique indique la proportion de la variance des données expliquée par le modèle. Un R² proche de 1 signifie que le modèle explique bien la variance des données. Ici, un R² de 0.5390 indique que le modèle explique environ 54 % de la variance.
- **PCC (Pearson Correlation Coefficient)** : Cette métrique mesure la corrélation linéaire entre les prédictions et les valeurs réelles. Un PCC élevé (proche de 1) indique une forte corrélation, ce qui est souhaitable pour un modèle de prédiction. Ici, un PCC de 0.7342 montre une corrélation positive significative.

### **Importance des caractéristiques**
Le modèle Random Forest permet d'évaluer l'importance des caractéristiques. Voici les résultats :
- **alpha** : 0.3575 (la plus importante)
- **gamma** : 0.2372
- **theta** : 0.1162
- **beta** : 0.1157
- **variance** : 0.0697
- **energy** : 0.0683
- **delta** : 0.0353 (la moins importante)

**Pourquoi ces résultats ?**
- La bande **alpha** est la plus importante car elle est associée à l'éveil calme et à la relaxation, des états souvent liés à la fatigue.
- La bande **gamma** est également importante car elle est associée aux processus cognitifs complexes, qui peuvent être affectés par la fatigue.
- Les bandes **theta** et **beta** sont moins importantes mais contribuent tout de même à la prédiction.
- Les caractéristiques supplémentaires (**variance** et **énergie**) fournissent des informations supplémentaires sur la dynamique du signal EEG, mais leur importance est moindre par rapport aux bandes de fréquence.

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
- **`random_forest.py`** : Entraînement et évaluation du modèle Random Forest.
- **`check_preprocess.py`** : Vérification des données prétraitées.
- **`check_fft_de.py`** : Vérification des résultats DE.

### Exécution du code
1. Placez les fichiers de données dans les dossiers `sortie_preprocess/lab` et `sortie_preprocess/real`.
2. Exécutez les scripts dans l'ordre suivant :
   ```bash
   python preprocessing.py
   python fft_de.py
   python random_forest.py
   python check_preprocess.py
   python check_fft_de.py
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

