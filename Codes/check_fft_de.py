import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins des dossiers
base_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/BCI/sortie_preprocess"

# Fonction pour vérifier les résultats DE
def check_de_results(de_results_file):
    # Charger les résultats DE
    if not os.path.exists(de_results_file):
        logging.error(f"Le fichier {de_results_file} n'existe pas. Assurez-vous que le script fft_de.py a été exécuté.")
        return

    de_results = np.load(de_results_file)
    subject_id = os.path.basename(de_results_file).replace('_de_results.npz', '')

    # Afficher les résultats
    print(f"\nRésultats DE pour {subject_id} :")
    for band, values in de_results.items():
        print(f"{band}: {values}")

    # Vérifier les erreurs potentielles
    for band, values in de_results.items():
        if np.isnan(values).any() or np.isinf(values).any():
            print(f"Attention : des valeurs NaN ou infinies ont été trouvées dans la bande {band}.")

    # Visualiser les résultats
    plt.figure(figsize=(10, 6))
    for band, values in de_results.items():
        plt.plot(values, label=band)

    plt.title(f"Differential Entropy (DE) pour chaque bande de fréquence - {subject_id}")
    plt.xlabel("Fenêtre")
    plt.ylabel("DE")
    plt.legend()
    plt.grid()
    plt.show()

    # Vérifier la cohérence des résultats
    print("\nVérification de la cohérence des résultats :")
    for band, values in de_results.items():
        mean_de = np.mean(values)
        std_de = np.std(values)
        print(f"{band}: Moyenne = {mean_de:.4f}, Écart-type = {std_de:.4f}")

# Parcourir les dossiers lab et real
def check_all_folders(base_folder):
    for scenario in ["lab", "real"]:
        scenario_folder = os.path.join(base_folder, scenario)
        if not os.path.exists(scenario_folder):
            logging.warning(f"Dossier {scenario_folder} introuvable. Ignoré.")
            continue

        # Parcourir tous les fichiers DE résultats
        for file in os.listdir(scenario_folder):
            if file.endswith('_de_results.npz'):
                de_results_file = os.path.join(scenario_folder, file)
                logging.info(f"Vérification des résultats pour {file}...")
                check_de_results(de_results_file)

# Exécution du script
if __name__ == "__main__":
    check_all_folders(base_folder)
    logging.info("Vérification terminée pour tous les fichiers.")