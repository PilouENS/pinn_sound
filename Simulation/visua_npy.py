from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# Visualisation temporelle d'un dataset .npy de la forme :
# [sample_id, x, y, t, pression]
# ============================================================

def load_dataset(path: str) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(
            "Le fichier .npy doit contenir un tableau 2D avec 5 colonnes : "
            "[sample_id, x, y, t, pression]"
        )
    return data


def select_sample(data: np.ndarray, sample_id: int) -> np.ndarray:
    sample = data[data[:, 0] == sample_id]
    if sample.size == 0:
        raise ValueError(f"Aucun sample_id={sample_id} trouvé dans le dataset.")
    return sample


def build_frames(sample: np.ndarray):
    """
    Reconstruit les frames 2D p(x,y,t) à partir du dataset tabulaire.
    Retourne :
    - times : liste triée des temps
    - x_unique : coordonnées x triées
    - y_unique : coordonnées y triées
    - frames : tableau (nt, nx, ny)
    """
    x_unique = np.unique(sample[:, 1])
    y_unique = np.unique(sample[:, 2])
    times = np.unique(sample[:, 3])

    nx = len(x_unique)
    ny = len(y_unique)
    nt = len(times)

    x_to_i = {x: i for i, x in enumerate(x_unique)}
    y_to_j = {y: j for j, y in enumerate(y_unique)}

    frames = np.zeros((nt, nx, ny), dtype=float)

    for k, t in enumerate(times):
        block = sample[sample[:, 3] == t]

        if block.shape[0] != nx * ny:
            raise ValueError(
                f"Au temps t={t}, le nombre de points ({block.shape[0]}) "
                f"ne correspond pas à nx*ny = {nx*ny}."
            )

        for row in block:
            _, x, y, _, p = row
            i = x_to_i[x]
            j = y_to_j[y]
            frames[k, i, j] = p

    return times, x_unique, y_unique, frames


def animate_dataset(path: str, sample_id: int = 0, interval_ms: int = 80):
    data = load_dataset(path)
    sample = select_sample(data, sample_id)
    times, x_unique, y_unique, frames = build_frames(sample)

    ########################################################
    frames_log = np.sign(frames) * np.log1p(np.abs(frames))
    vabs = np.percentile(np.abs(frames_log), 99)
    ########################################################

    if vabs == 0:
        vabs = 1.0

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        frames_log[0].T,
        origin="lower",
        extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
        aspect="equal",
        cmap="seismic",
        vmin=-vabs,
        vmax=vabs,
        animated=True,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pression")

    title = ax.set_title(f"sample {sample_id} — t = {times[0]:.6e} s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame_idx: int):
        
        im.set_data(frames_log[frame_idx].T)

        title.set_text(f"sample {sample_id} — t = {times[frame_idx]:.6e} s")
        return im, title

    ani = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    ani.save(
    "Simulation_exemple.gif",
    writer="pillow",
    fps=1000 // interval_ms
)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python visualiser_npy.py fichier.npy [sample_id] [interval_ms]")
        sys.exit(1)

    path = sys.argv[1]
    sample_id = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    interval_ms = int(sys.argv[3]) if len(sys.argv) >= 4 else 80

    animate_dataset(path, sample_id=sample_id, interval_ms=interval_ms)