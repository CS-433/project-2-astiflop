import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import os
import shutil

# ==========================================
# 1. CONFIGURATION
# ==========================================
IMG_SIZE = 224  # Taille standard pour ResNet/MobileNet
DPI = 100  # Qualité de l'image
SEGMENT_LENGTH = 100  # Nombre de points par image (votre "slice")
OUTPUT_DIR = "worm_trajectories"
# INPUT dir est à la racine du projet, contenant les CSV originaux
INPUT_DIR = "preprocessed_data"
CONTROL = "TERBINAFINE- (control)"
TREATED = "TERBINAFINE+"

ACTUAL_PATH = Path(__file__).parent.resolve()
DATA_PATH = ACTUAL_PATH.parent / INPUT_DIR

# REGLAGE CRITIQUE DE LA COULEUR
# Si vous avez des outliers à 10 mais que la vraie vitesse est 0.5,
# baissez ce chiffre (ex: 90 ou 85) pour saturer le rouge plus tôt.
ROBUST_PERCENTILE = 80


# Simulation de données pour l'exemple (A REMPLACER par votre chargement de CSV)
# Structure attendue : un DataFrame avec colonnes ['worm_id', 'x', 'y', 'class_label']
def generate_dummy_data(n_worms=10):
    data = []
    print("Génération de données simulées...")
    for i in range(n_worms):
        # 50% normaux (0), 50% drogués (1)
        is_drugged = i % 2 != 0
        label = "drogue" if is_drugged else "normal"

        # Simuler 8000 points de trajectoire
        t = np.linspace(0, 100, 8000)

        if is_drugged:
            # Drogué : Mouvements lents, erratiques (petit rayon), vitesse faible
            speed_factor = 0.5 + np.random.rand() * 0.5  # Vitesse entre 0.5 et 1.0
            x = np.cumsum(np.random.randn(8000)) * 0.5  # Random walk serré
            y = np.cumsum(np.random.randn(8000)) * 0.5
        else:
            # Normal : Rapide, mouvements amples
            speed_factor = 2.0 + np.random.rand() * 2.0  # Vitesse entre 2.0 et 4.0
            x = np.sin(t) * 10 + np.cumsum(np.random.randn(8000) * 0.2)
            y = np.cos(t) * 10 + np.cumsum(np.random.randn(8000) * 0.2)

        # Création des lignes pour ce ver
        for step in range(len(t)):
            # On ajoute un peu de variation de vitesse locale
            local_speed = speed_factor * (0.8 + 0.4 * np.random.rand())

            # AJOUT D'UN OUTLIER ARTIFICIEL (Bug de tracking) pour tester la robustesse
            if np.random.rand() < 0.001:
                local_speed = 50.0  # Vitesse absurde

            data.append(
                {
                    "worm_id": i,
                    "time": step,
                    "x": x[step],
                    "y": y[step],
                    "speed": local_speed,  # On suppose qu'on a déjà la vitesse, sinon on la calcule
                    "class_label": label,
                }
            )
    return pd.DataFrame(data)


# ==========================================
# 2. FONCTION CŒUR : TRAJECTOIRE -> IMAGE
# ==========================================
def save_trajectory_image(segment_df, output_path, global_max_speed):
    """
    Transforme un segment de données X,Y,Speed en image PNG colorée.
    """
    x = segment_df["X"].values
    y = segment_df["Y"].values
    speed = segment_df["Speed"].values

    # Préparer les segments pour LineCollection
    # On crée des paires de points (x0, y0) -> (x1, y1)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Création de la figure sans bordures
    fig = plt.figure(figsize=(IMG_SIZE / DPI, IMG_SIZE / DPI), dpi=DPI)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])  # Occupe tout l'espace
    ax.set_axis_off()
    fig.add_axes(ax)

    # Normalisation de la vitesse pour la couleur (0 = Bleu, 1 = Rouge)
    # vmin=0, vmax=global_max_speed.
    # Tout ce qui est > vmax sera clampé à 1.0 (Rouge pur) par matplotlib automatiquement.
    norm = plt.Normalize(vmin=0, vmax=global_max_speed, clip=True)

    # Création de la ligne colorée
    # cmap='turbo' a souvent un meilleur contraste perceptuel que 'jet' pour les données scientifiques
    lc = LineCollection(segments, cmap="turbo", norm=norm)

    # La couleur est déterminée par la vitesse moyenne de chaque petit segment
    # On aligne la taille de 'speed' avec 'segments' (un point de moins)
    speed_values = (speed[:-1] + speed[1:]) / 2
    lc.set_array(speed_values)
    lc.set_linewidth(2)  # Épaisseur du trait

    ax.add_collection(lc)

    # Centrage automatique sur le segment
    if len(x) > 0:
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        # Garder l'aspect ratio (pour ne pas déformer les virages)
        ax.set_aspect("equal", "datalim")

    # Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI)
    plt.close(fig)  # Très important pour libérer la mémoire


# ==========================================
# 3. PIPELINE PRINCIPAL
# ==========================================
def process_dataset():
    global_max_speed = 0
    # B. Charger les données
    # les données sont dans un dossier à la racine du projet, sachant que je ne suis acutellement pas à la racine du projet
    for treatment in [CONTROL, TREATED]:
        treatment_path = DATA_PATH / treatment
        for file_name in os.listdir(treatment_path):
            file_path = treatment_path / file_name
            df = pd.read_csv(file_path)
            if df["Speed"].quantile(ROBUST_PERCENTILE / 100.0) > global_max_speed:
                global_max_speed = df["Speed"].quantile(ROBUST_PERCENTILE / 100.0)
            # C. ANALYSE STATISTIQUE DE LA VITESSE
            # C'est ici qu'on évite le piège des outliers
            # print("\n--- ANALYSE DE LA DISTRIBUTION DES VITESSES ---")
            # print(f"Vitesse Min      : {df['Speed'].min():.4f}")
            # print(f"Vitesse Médiane  : {df['Speed'].median():.4f}")
            # print(f"Vitesse Moyenne  : {df['Speed'].mean():.4f}")
            # print(f"Percentile 90%   : {df['Speed'].quantile(0.90):.4f}")
            # print(f"Percentile 95%   : {df['Speed'].quantile(0.95):.4f}")
            # print(f"Vitesse Max (Brut): {df['Speed'].max():.4f}")

    print(f"\n>>> SEUIL CHOISI (Rouge Max) : {global_max_speed:.4f}")
    print(f"Tout point > {global_max_speed:.4f} sera affiché en ROUGE MAXIMAL.")
    print("-----------------------------------------------\n")
    # D. Traitement par ver
    print(f"Début du traitement ...")
    total_images = 0
    for treatment in [CONTROL, TREATED]:
        treatment_path = DATA_PATH / treatment
        for file_name in os.listdir(treatment_path):
            file_path = treatment_path / file_name
            df = pd.read_csv(file_path)
            segments = df["Segment"].unique()
            for segment in segments:
                segment_df = df[df["Segment"] == segment]
                worm_id = file_name.split(".csv")[0]
                label = treatment
                # Nom de fichier : IMPORTANT pour le 'Group Split' plus tard
                # On garde l'ID du ver dans le nom pour savoir qui est qui
                filename = f"{worm_id}_seg_{segment}.png"
                output_path = f"{OUTPUT_DIR}/{label}/{worm_id}/{filename}"
                save_trajectory_image(segment_df, output_path, global_max_speed)
                total_images += 1
            print(f"Fichier {file_name} traité -> {len(segments)} images générées.")

    print(f"\nTraitement terminé ! {total_images} images générées dans '{OUTPUT_DIR}'")


# Lancer le script
if __name__ == "__main__":
    process_dataset()
