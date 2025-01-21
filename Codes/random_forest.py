import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import variation
from scipy.integrate import simpson  # Utiliser simpson au lieu de simps

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins vers les dossiers lab et real
dossier_lab = os.path.join('sortie_preprocess', 'lab')
dossier_real = os.path.join('sortie_preprocess', 'real')

# Fonction pour extraire des caractéristiques supplémentaires (variance et énergie)
def extract_features(eeg_windows):
    features = []
    for window in eeg_windows:
        # Variance
        var = np.var(window)
        # Energie (intégrale du carré du signal)
        energy = simpson(window**2, dx=1)  # Utiliser simpson au lieu de simps
        features.append([var, energy])
    return np.array(features)

# Fonction pour charger les données d'un sujet
def charger_donnees_sujet(dossier, sujet):
    # Charger les fenêtres EEG (eeg_windows)
    chemin_eeg_windows = os.path.join(dossier, f'{sujet}_eeg_windows.npy')
    if not os.path.exists(chemin_eeg_windows):
        raise FileNotFoundError(f"Fichier {chemin_eeg_windows} non trouvé")
    eeg_windows = np.load(chemin_eeg_windows)

    # Charger les caractéristiques DE (de_results)
    chemin_de = os.path.join(dossier, f'{sujet}_de_results.npz')
    if not os.path.exists(chemin_de):
        raise FileNotFoundError(f"Fichier {chemin_de} non trouvé")
    de_data = np.load(chemin_de)
    
    # Combiner les bandes de fréquence en une seule matrice
    X = np.hstack([
        de_data['delta'].reshape(-1, 1),  # Bande delta
        de_data['theta'].reshape(-1, 1),  # Bande theta
        de_data['alpha'].reshape(-1, 1),  # Bande alpha
        de_data['beta'].reshape(-1, 1),   # Bande beta
        de_data['gamma'].reshape(-1, 1)   # Bande gamma
    ])

    # Extraire des caractéristiques supplémentaires
    X_additional = extract_features(eeg_windows)

    # Combiner les caractéristiques
    X = np.hstack([X, X_additional])

    # Charger les étiquettes PERCLOS (fichier .npy)
    chemin_perclos = os.path.join(dossier, f'{sujet}_perclos.npy')
    if not os.path.exists(chemin_perclos):
        raise FileNotFoundError(f"Fichier {chemin_perclos} non trouvé")
    y = np.load(chemin_perclos)  # Charger les étiquettes PERCLOS

    # Vérifier les dimensions de X et y
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Dimensions incohérentes pour le sujet {sujet}: X a {X.shape[0]} échantillons, y a {y.shape[0]} échantillons")

    return X, y

# Fonction pour charger les données de tous les sujets
def charger_donnees_tous_sujets(dossier, sujets):
    X, y = [], []
    for sujet in sujets:
        try:
            X_sujet, y_sujet = charger_donnees_sujet(dossier, sujet)
            # Vérifier les dimensions avant d'ajouter
            if X_sujet.shape[0] == y_sujet.shape[0]:
                X.append(X_sujet)
                y.append(y_sujet)
            else:
                logging.warning(f"Dimensions incohérentes pour le sujet {sujet}: X a {X_sujet.shape[0]} échantillons, y a {y_sujet.shape[0]} échantillons")
        except FileNotFoundError as e:
            logging.warning(f"Données manquantes pour le sujet {sujet}: {e}")
        except ValueError as e:
            logging.warning(f"Erreur de dimensions pour le sujet {sujet}: {e}")
    
    # Vérifier si des données ont été chargées
    if not X or not y:
        raise ValueError("Aucune donnée valide n'a été chargée.")
    
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

# Liste des sujets (20 sujets pour lab, 14 sujets pour real)
sujets_lab = range(1, 21)  # Sujets 1 à 20 pour lab
sujets_real = range(1, 15)  # Sujets 1 à 14 pour real

# Charger les données du lab
try:
    X_lab, y_lab = charger_donnees_tous_sujets(dossier_lab, sujets_lab)
except ValueError as e:
    logging.error(f"Erreur lors du chargement des données lab : {e}")
    X_lab, y_lab = np.array([]), np.array([])

# Charger les données du real
try:
    X_real, y_real = charger_donnees_tous_sujets(dossier_real, sujets_real)
except ValueError as e:
    logging.error(f"Erreur lors du chargement des données real : {e}")
    X_real, y_real = np.array([]), np.array([])

# Vérifier si des données ont été chargées
if X_lab.size == 0 and X_real.size == 0:
    raise ValueError("Aucune donnée n'a été chargée. Vérifiez les chemins des fichiers.")

# Combiner les données lab et real (si tu veux les utiliser ensemble)
X = np.concatenate([X_lab, X_real], axis=0)
y = np.concatenate([y_lab, y_real], axis=0)

# Vérifier les dimensions finales de X et y
logging.info(f"Dimensions de X : {X.shape}")
logging.info(f"Dimensions de y : {y.shape}")

# Normalisation des caractéristiques (optionnel mais recommandé)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division des données en ensembles d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Définir une grille d'hyperparamètres à tester
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres
grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train.ravel())

# Meilleurs hyperparamètres
logging.info(f"Meilleurs hyperparamètres : {grid_search.best_params_}")

# Utiliser le meilleur modèle
best_rf_regressor = grid_search.best_estimator_

# Prédire sur l'ensemble de test
y_pred = best_rf_regressor.predict(X_test)

# Évaluer les performances du modèle
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
mae = mean_absolute_error(y_test, y_pred)  # MAE
r2 = r2_score(y_test, y_pred)  # R²
pcc = np.corrcoef(y_test.ravel(), y_pred)[0, 1]  # PCC

logging.info(f"RMSE sur l'ensemble de test : {rmse:.4f}")
logging.info(f"MAE sur l'ensemble de test : {mae:.4f}")
logging.info(f"R² sur l'ensemble de test : {r2:.4f}")
logging.info(f"PCC sur l'ensemble de test : {pcc:.4f}")

# Importance des caractéristiques
importances = best_rf_regressor.feature_importances_
feature_names = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'variance', 'energy']

logging.info("Importance des caractéristiques :")
for feature, importance in zip(feature_names, importances):
    logging.info(f"{feature}: {importance:.4f}")