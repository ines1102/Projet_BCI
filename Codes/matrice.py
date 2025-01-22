# pearson_matrix.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins vers les dossiers lab et real
dossier_lab = os.path.join('sortie_preprocess', 'lab')
dossier_real = os.path.join('sortie_preprocess', 'real')

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

    # Charger les étiquettes PERCLOS (fichier .npy)
    chemin_perclos = os.path.join(dossier, f'{sujet}_perclos.npy')
    if not os.path.exists(chemin_perclos):
        raise FileNotFoundError(f"Fichier {chemin_perclos} non trouvé")
    y = np.load(chemin_perclos)  # Charger les étiquettes PERCLOS

    return X, y

# Fonction pour charger les données de tous les sujets
def charger_donnees_tous_sujets(dossier, sujets):
    X, y = [], []
    for sujet in sujets:
        try:
            X_sujet, y_sujet = charger_donnees_sujet(dossier, sujet)
            X.append(X_sujet)
            y.append(y_sujet)
        except FileNotFoundError as e:
            logging.warning(f"Données manquantes pour le sujet {sujet}: {e}")
        except ValueError as e:
            logging.warning(f"Erreur de dimensions pour le sujet {sujet}: {e}")
    
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

# Combiner les données lab et real
X = np.concatenate([X_lab, X_real], axis=0)
y = np.concatenate([y_lab, y_real], axis=0)

# Créer un DataFrame avec les caractéristiques et la cible
feature_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
X_df = pd.DataFrame(X, columns=feature_names)
X_df['PERCLOS'] = y

# Calculer la matrice de corrélation de Pearson
correlation_matrix = X_df.corr(method='pearson')

# Visualiser la matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matrice de corrélation de Pearson")
plt.savefig("Visuels/pearson_matrix.png")  # Sauvegarder la figure
plt.show()