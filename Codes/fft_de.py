import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.integrate import simpson as simps
import os
import logging
import matplotlib.pyplot as plt

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Bandes de fréquence pour le filtrage
freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# Fonction pour appliquer un filtre passe-bande
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Fonction pour calculer la Differential Entropy (DE)
def compute_de(data, band, fs):
    # Calculer la PSD (Power Spectral Density) avec la méthode de Welch
    f, psd = welch(data, fs=fs, nperseg=1024)
    
    # Trouver les indices de la bande de fréquence
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    
    # Calculer la DE comme l'intégrale de la PSD dans la bande de fréquence
    de = simps(psd[idx_band], dx=f[1] - f[0])
    return de

# Fonction pour afficher les données filtrées
# def plot_filtered_data(data, title, save_path=None):
#     plt.figure(figsize=(10, 5))
#     plt.plot(data, label="Données filtrées")
#     plt.title(title)
#     plt.xlabel("Temps (échantillons)")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid()
#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.show()
#     plt.close()

# Charger les données prétraitées
def load_preprocessed_data(eeg_windows_file):
    subject_id = os.path.basename(eeg_windows_file).replace('_eeg_windows.npy', '')
    metadata_file = eeg_windows_file.replace('_eeg_windows.npy', '_metadata.npz')
    
    if not os.path.exists(eeg_windows_file):
        raise FileNotFoundError(f"Fichier EEG {eeg_windows_file} introuvable.")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Fichier de métadonnées {metadata_file} introuvable.")
    
    eeg_windows = np.load(eeg_windows_file)
    metadata = np.load(metadata_file, allow_pickle=True)
    
    return eeg_windows, metadata, subject_id

# Fonction principale pour appliquer le filtrage, la FFT et calculer la DE
def process_bands(eeg_windows_file, output_folder):
    try:
        # Charger les données prétraitées
        eeg_windows, metadata, subject_id = load_preprocessed_data(eeg_windows_file)
        fs = metadata['sampling_rate']
        
        # Normaliser les données EEG
        eeg_windows = (eeg_windows - np.mean(eeg_windows)) / np.std(eeg_windows)
        
        # Initialiser un dictionnaire pour stocker les résultats
        de_results = {band: [] for band in freq_bands.keys()}
        
        # Traiter chaque bande de fréquence individuellement
        for band_name, band_range in freq_bands.items():
            logging.info(f"Traitement de la bande {band_name} ({band_range[0]} - {band_range[1]} Hz)...")
            
            # Initialiser une liste pour stocker les DE de chaque fenêtre
            de_values = []
            
            # Traiter chaque fenêtre EEG
            for window in eeg_windows:
                # Appliquer le filtre passe-bande
                filtered_data = bandpass_filter(window, band_range[0], band_range[1], fs)
                
                # Afficher les données filtrées pour la première fenêtre
                #if len(de_values) == 0:
                #    plot_filtered_data(filtered_data, title=f"Données filtrées pour la bande {band_name} - {subject_id}")
                
                # Calculer la DE
                de = compute_de(filtered_data, band_range, fs)
                de_values.append(de)
            
            # Ajouter les résultats DE pour cette bande
            de_results[band_name] = de_values
            logging.info(f"DE moyenne pour la bande {band_name}: {np.mean(de_values):.4f}")
        
        # Sauvegarder les résultats
        output_file = os.path.join(output_folder, f"{subject_id}_de_results.npz")
        np.savez(output_file, **de_results)
        logging.info(f"Résultats DE sauvegardés pour {subject_id} dans {output_file}")

    except Exception as e:
        logging.error(f"Erreur lors du traitement de {eeg_windows_file}: {e}")

# Parcourir les dossiers lab et real
def process_all_folders(base_folder):
    for scenario in ["lab", "real"]:
        scenario_folder = os.path.join(base_folder, scenario)
        if not os.path.exists(scenario_folder):
            logging.warning(f"Dossier {scenario_folder} introuvable. Ignoré.")
            continue
        
        # Parcourir tous les fichiers EEG prétraités
        for file in os.listdir(scenario_folder):
            if file.endswith('_eeg_windows.npy'):
                eeg_windows_file = os.path.join(scenario_folder, file)
                process_bands(eeg_windows_file, scenario_folder)

# Exécution du script
if __name__ == "__main__":
    base_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/BCI/sortie_preprocess"
    process_all_folders(base_folder)
    logging.info("Traitement terminé pour tous les fichiers.")