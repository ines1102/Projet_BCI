import os
import mne
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins des dossiers
base_path = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/BCI/VLA_VRW"
output_path = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/BCI/sortie_preprocess"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_path, exist_ok=True)

# Paramètres de prétraitement
initial_crop = 24  # Supprimer les premières 24 secondes
bandpass_range = (0.1, 70)  # Filtre passe-bande (0.1 - 70 Hz)
window_size = 2400  # Nombre de points EEG pour une valeur PERCLOS

# Fonction pour visualiser le spectre des fréquences
def plot_spectrum(raw, title, save_path=None):
    spectrum = raw.compute_psd(method='welch', fmin=0, fmax=100)
    psd, freqs = spectrum.get_data(return_freqs=True)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, psd.mean(axis=0), label='PSD moyenne')
    plt.axvline(x=50, color='r', linestyle='--', label='50 Hz')
    plt.axvline(x=60, color='g', linestyle='--', label='60 Hz')
    plt.title(title)
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Puissance (dB/Hz)')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# Fonction pour inspecter les clés d'un fichier .mat
def inspect_mat_file(mat_file):
    data = scipy.io.loadmat(mat_file)
    keys = list(data.keys())
    logging.info(f"Clés disponibles dans {mat_file}: {keys}")
    
    # Recherche récursive de la clé PERCLOS
    def find_perclos(data):
        if 'perclos' in data:  # Utiliser 'perclos' en minuscules
            return data['perclos']
        for key, value in data.items():
            if isinstance(value, dict):
                result = find_perclos(value)
                if result is not None:
                    return result
        return None
    
    perclos = find_perclos(data)
    if perclos is None:
        raise KeyError(f"Aucune clé PERCLOS trouvée dans {mat_file}")
    return perclos

# Fonction pour prétraiter un fichier EEG et PERCLOS
def preprocess_eeg_perclos(eeg_file, perclos_file, output_folder, notch_freq=None, plot_spectrum_flag=False):
    try:
        logging.info(f"Début du prétraitement pour {eeg_file}")
        
        # Charger les données EEG
        if not os.path.exists(eeg_file):
            raise FileNotFoundError(f"Fichier EEG {eeg_file} introuvable.")
        raw = mne.io.read_raw_edf(eeg_file, preload=True)
        sampling_rate = raw.info['sfreq']  # Fréquence d'échantillonnage réelle
        logging.info(f"Fréquence d'échantillonnage de l'EEG : {sampling_rate} Hz")
        
        # Durée totale des données EEG
        initial_duration = raw.times[-1]
        logging.info(f"Durée initiale des données EEG pour {os.path.basename(eeg_file)} : {initial_duration:.2f} secondes")

        # Charger les données PERCLOS
        if not os.path.exists(perclos_file):
            raise FileNotFoundError(f"Fichier PERCLOS {perclos_file} introuvable.")
        perclos_data = inspect_mat_file(perclos_file)
        perclos = perclos_data  # La clé 'perclos' est déjà extraite dans inspect_mat_file

        # Calculer l'intervalle entre deux valeurs PERCLOS
        perclos_interval = initial_duration / len(perclos)
        logging.info(f"Intervalle entre deux valeurs PERCLOS : {perclos_interval:.2f} secondes")

        # Supprimer les premières 24 secondes des données EEG
        raw.crop(tmin=24)
        cropped_duration = raw.times[-1]
        logging.info(f"Durée des données EEG après suppression des 24 secondes : {cropped_duration:.2f} secondes")

        # Supprimer les premières valeurs PERCLOS correspondantes
        n_perclos_to_remove = int(24 / perclos_interval)
        perclos = perclos[n_perclos_to_remove:]
        logging.info(f"Suppression des {n_perclos_to_remove} premières valeurs PERCLOS.")

        # Appliquer un filtre passe-bande
        raw.filter(l_freq=bandpass_range[0], h_freq=bandpass_range[1], method='fir', fir_window='hamming')
    
        # Appliquer le filtre notch si spécifié
        if notch_freq is not None:
            raw.notch_filter(freqs=notch_freq)
            logging.info(f"Filtre notch appliqué à {notch_freq} Hz pour {os.path.basename(eeg_file)}")
            if plot_spectrum_flag:
                plot_spectrum(raw, title=f"Spectre EEG après filtre notch - {os.path.basename(eeg_file)}")

        # Découper les données EEG en fenêtres de 2400 points
        eeg_data = raw.get_data().flatten()
        n_windows = int(len(eeg_data) / window_size)
        
        # Tronquer les données EEG pour correspondre au nombre de fenêtres
        eeg_data = eeg_data[:n_windows * window_size]
        eeg_windows = np.reshape(eeg_data, (n_windows, window_size))

        # Vérifier les artefacts dans les données EEG
        if np.isnan(eeg_windows).any() or np.isinf(eeg_windows).any():
            logging.warning("Des artefacts (NaN ou inf) ont été détectés dans les données EEG. Suppression...")
            eeg_windows = np.nan_to_num(eeg_windows, nan=0.0, posinf=0.0, neginf=0.0)

        # Normaliser les données EEG
        eeg_windows = (eeg_windows - np.mean(eeg_windows)) / np.std(eeg_windows)

        # Vérifier que le nombre de fenêtres EEG correspond au nombre de valeurs PERCLOS
        if len(eeg_windows) != len(perclos):
            logging.warning(f"Le nombre de fenêtres EEG ({len(eeg_windows)}) ne correspond pas exactement au nombre de valeurs PERCLOS ({len(perclos)}).")
            # Tronquer les données EEG pour correspondre au nombre de valeurs PERCLOS
            min_length = min(len(eeg_windows), len(perclos))
            eeg_windows = eeg_windows[:min_length]
            perclos = perclos[:min_length]
            logging.info(f"Données EEG et PERCLOS tronquées à {min_length} fenêtres pour correspondre.")

        # Sauvegarder les données prétraitées
        subject_id = os.path.basename(eeg_file).split('.')[0]
        eeg_output_file = os.path.join(output_folder, f"{subject_id}_eeg_windows.npy")
        np.save(eeg_output_file, eeg_windows)

        perclos_output_file = os.path.join(output_folder, f"{subject_id}_perclos.npy")
        np.save(perclos_output_file, perclos)

        # Sauvegarder les métadonnées
        metadata = {
            'eeg_duration': raw.times[-1],
            'perclos_duration': len(perclos) * perclos_interval,
            'sampling_rate': sampling_rate,
            'notch_freq': notch_freq,
            'window_size': window_size,
            'perclos_interval': perclos_interval
        }
        np.savez(os.path.join(output_folder, f"{subject_id}_metadata.npz"), **metadata)

        logging.info(f"Prétraitement terminé pour {subject_id}")

    except Exception as e:
        logging.error(f"Erreur lors du prétraitement de {eeg_file}: {e}")

# Parcourir les dossiers lab et real
for scenario in ["lab", "real"]:
    eeg_folder = os.path.join(base_path, scenario, "EEG")
    perclos_folder = os.path.join(base_path, scenario, "perclos")
    output_folder = os.path.join(output_path, scenario)
    os.makedirs(output_folder, exist_ok=True)

    eeg_files = [f for f in os.listdir(eeg_folder) if f.endswith('.edf')]
    for i, eeg_file in enumerate(eeg_files):
        eeg_file_path = os.path.join(eeg_folder, eeg_file)
        perclos_file = eeg_file.replace('.edf', '.mat')
        perclos_file_path = os.path.join(perclos_folder, perclos_file)

        if os.path.exists(perclos_file_path):
            if i == 0:  # Pour le premier fichier, afficher le spectre et demander à l'utilisateur
                plot_spectrum_flag = True
                plot_spectrum(mne.io.read_raw_edf(eeg_file_path, preload=True), title=f"Spectre EEG avant filtre notch - {eeg_file}")
                apply_notch = input("Voulez-vous appliquer un filtre notch ? (oui/non) : ").strip().lower()
                if apply_notch == 'oui':
                    notch_freq = input("À quelle fréquence appliquer le filtre notch ? (50/60) : ").strip()
                    notch_freq = int(notch_freq) if notch_freq in ['50', '60'] else None
                else:
                    notch_freq = None
            else:  # Pour les fichiers suivants, utiliser le même filtre notch
                plot_spectrum_flag = False

            preprocess_eeg_perclos(eeg_file_path, perclos_file_path, output_folder, notch_freq=notch_freq, plot_spectrum_flag=plot_spectrum_flag)
        else:
            logging.warning(f"Fichier PERCLOS manquant pour {eeg_file}")

logging.info("Prétraitement terminé pour tous les fichiers.")
