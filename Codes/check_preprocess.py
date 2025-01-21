import os
import mne
import numpy as np

# Chemins des dossiers
base_path = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/BCI/VLA_VRW"
output_path = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/BCI/sortie_preprocess"

# Fonction pour vérifier les données prétraitées
def validate_preprocessing(subject_id, scenario):
    try:
        # Chemins des fichiers prétraités
        eeg_output_file = os.path.join(output_path, scenario, f"{subject_id}_eeg_windows.npy")
        perclos_output_file = os.path.join(output_path, scenario, f"{subject_id}_perclos.npy")
        metadata_file = os.path.join(output_path, scenario, f"{subject_id}_metadata.npz")

        # Vérifier que les fichiers existent
        if not os.path.exists(eeg_output_file):
            raise FileNotFoundError(f"Fichier EEG prétraité manquant : {eeg_output_file}")
        if not os.path.exists(perclos_output_file):
            raise FileNotFoundError(f"Fichier PERCLOS prétraité manquant : {perclos_output_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Fichier de métadonnées manquant : {metadata_file}")

        # Charger les données EEG prétraitées
        eeg_preprocessed = np.load(eeg_output_file)
        print(f"Données EEG prétraitées chargées pour {subject_id} (scénario {scenario})")
        print(f"Nombre de fenêtres EEG : {eeg_preprocessed.shape[0]}")
        print(f"Taille de chaque fenêtre EEG : {eeg_preprocessed.shape[1]} points")

        # Charger les données PERCLOS prétraitées
        perclos_preprocessed = np.load(perclos_output_file)
        print(f"Données PERCLOS prétraitées chargées pour {subject_id} (scénario {scenario})")
        print(f"Nombre de valeurs PERCLOS : {len(perclos_preprocessed)}")

        # Charger les métadonnées
        metadata = np.load(metadata_file)
        sampling_rate = metadata['sampling_rate']
        perclos_interval = metadata['perclos_interval']
        print(f"Fréquence d'échantillonnage EEG : {sampling_rate} Hz")
        print(f"Intervalle entre deux valeurs PERCLOS : {perclos_interval:.2f} secondes")

        # Calculer la durée des données EEG
        window_size = eeg_preprocessed.shape[1]
        duration_eeg = (window_size / sampling_rate) * eeg_preprocessed.shape[0]
        print(f"Durée des données EEG prétraitées : {duration_eeg:.2f} secondes")

        # Calculer la durée des données PERCLOS
        duration_perclos = len(perclos_preprocessed) * perclos_interval
        print(f"Durée des données PERCLOS prétraitées : {duration_perclos:.2f} secondes")

        # Vérifier que les durées EEG et PERCLOS sont synchronisées
        tolerance = 1.0  # Tolérance de 1 seconde
        if not np.isclose(duration_eeg, duration_perclos, atol=tolerance):
            raise ValueError(f"Les durées EEG et PERCLOS ne sont pas synchronisées : "
                            f"EEG = {duration_eeg:.2f}s, PERCLOS = {duration_perclos:.2f}s")

        print(f"Validation réussie pour {subject_id} (scénario {scenario})")

    except Exception as e:
        print(f"Erreur lors de la validation pour {subject_id} (scénario {scenario}) : {e}")

# Parcourir les dossiers lab et real
for scenario in ["lab", "real"]:
    eeg_folder = os.path.join(base_path, scenario, "EEG")
    eeg_files = [f for f in os.listdir(eeg_folder) if f.endswith('.edf')]
    for eeg_file in eeg_files:
        subject_id = os.path.basename(eeg_file).split('.')[0]
        validate_preprocessing(subject_id, scenario)