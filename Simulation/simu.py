from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
#  Simulation acoustique 2D
#  - source impulsionnelle gaussienne au centre
#  - bords absorbants (Mur ordre 1)
#  - un obstacle rigide (Neumann approx. = miroir)
#  - sorties : dataset [sample_id, x, y, t, pression]
# ============================================================================

# -----------------------------
# 1) Paramètres physiques/numeriques
# -----------------------------
c = 343.0                        # vitesse du son (m/s)

nx = ny = 201                    # points de grille
Lx = Ly = 1.0                    # domaine [0,1] x [0,1]
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
assert abs(dx - dy) < 1e-15, "Ce script suppose dx = dy"

# CFL stable pour schéma 2D explicite
dt_cfl = dx / (c * np.sqrt(2))
dt = 0.95 * dt_cfl

T_final = 0.006                  # durée totale (s)
n_steps = int(np.ceil(T_final / dt))

save_every = 8                   # on ne sauvegarde pas tous les pas pour alléger
preview = os.environ.get("SIM_PREVIEW", "1") != "0"
seed = int(os.environ.get("SIM_SEED", "0"))
n_samples = int(os.environ.get("SIM_SAMPLES", "1"))

rng = np.random.default_rng(seed)

print(f"Grille       : {nx} x {ny}")
print(f"dx = dy      : {dx:.6f} m")
print(f"dt           : {dt:.3e} s")
print(f"Durée totale : {T_final:.4f} s")
print(f"Pas de temps : {n_steps}")

# -----------------------------
# 2) Grille spatiale
# -----------------------------
x = np.linspace(0.0, Lx, nx)
y = np.linspace(0.0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing="ij")

c2dt2_dx2 = (c * dt / dx) ** 2

# Coefficient Mur ordre 1
# Forme 1D appliquée sur chaque bord :
# p^{n+1}_bord = p^n_voisin + alpha * (p^{n+1}_voisin - p^n_bord)
alpha = (c * dt - dx) / (c * dt + dx)

# -----------------------------
# 3) Source initiale
# -----------------------------
def gaussian_pulse_centered(X: np.ndarray, Y: np.ndarray,
                            x0: float = 0.5, y0: float = 0.5,
                            sigma: float = 0.04,
                            amplitude: float = 1.0) -> np.ndarray:
    """
    Impulsion initiale de pression centrée.
    Pas de réémission ensuite : la source n'agit qu'à t=0.
    """
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return amplitude * np.exp(-r2 / (2.0 * sigma**2))

# -----------------------------
# 4) Obstacle rigide unique
# -----------------------------
def make_single_rect_obstacle_mask(
    rng: np.random.Generator,
    width_cells: int = 24,
    height_cells: int = 18,
    margin_cells: int = 20,
    center_exclusion_radius: float = 0.16,
) -> np.ndarray:
    """
    Crée un seul obstacle rectangulaire de taille fixe, position aléatoire.
    L'obstacle est évité près du centre pour ne pas recouvrir la source.
    """
    mask = np.zeros((nx, ny), dtype=bool)

    while True:
        ix0 = rng.integers(margin_cells, nx - width_cells - margin_cells)
        iy0 = rng.integers(margin_cells, ny - height_cells - margin_cells)

        cx_obs = (ix0 + width_cells / 2) * dx
        cy_obs = (iy0 + height_cells / 2) * dy

        if np.hypot(cx_obs - 0.5, cy_obs - 0.5) > center_exclusion_radius:
            break

    mask[ix0:ix0 + width_cells, iy0:iy0 + height_cells] = True
    return mask

# -----------------------------
# 5) Laplacien avec obstacle rigide
# -----------------------------
def laplacian_with_rigid_obstacle(p: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
    """
    Approximation du Laplacien 2D avec obstacle rigide.

    Idée :
    - dans le fluide : schéma différences finies centré
    - si un voisin est dans l'obstacle, on impose approx. d/dn p = 0
      en remplaçant la valeur voisine par la valeur locale.
      => réflexion type paroi rigide

    Remarque :
    c'est une approximation simple mais cohérente pour un prototype.
    """
    lap = np.zeros_like(p)

    fluid = ~obstacle_mask
    interior = np.zeros_like(fluid, dtype=bool)
    interior[1:-1, 1:-1] = True
    active = fluid & interior

    p_c = p[1:-1, 1:-1]

    # Voisins géométriques
    p_l = p[:-2, 1:-1].copy()
    p_r = p[2:, 1:-1].copy()
    p_d = p[1:-1, :-2].copy()
    p_u = p[1:-1, 2:].copy()

    # Masques obstacle chez les voisins
    obs_l = obstacle_mask[:-2, 1:-1]
    obs_r = obstacle_mask[2:, 1:-1]
    obs_d = obstacle_mask[1:-1, :-2]
    obs_u = obstacle_mask[1:-1, 2:]

    # Condition rigide approx. : voisin obstacle -> on remplace par la cellule courante
    p_l[obs_l] = p_c[obs_l]
    p_r[obs_r] = p_c[obs_r]
    p_d[obs_d] = p_c[obs_d]
    p_u[obs_u] = p_c[obs_u]

    lap_local = (p_l + p_r + p_d + p_u - 4.0 * p_c) / dx**2

    lap[1:-1, 1:-1] = lap_local

    # Dans l'obstacle lui-même, on ne résout rien
    lap[obstacle_mask] = 0.0

    # Hors domaine intérieur, valeur laissée à 0 ; les bords seront traités par Mur
    lap[~active] = lap[~active]
    return lap

# -----------------------------
# 6) Conditions absorbantes de Mur
# -----------------------------
def apply_mur_boundaries(p_next: np.ndarray, p_curr: np.ndarray) -> None:
    """
    Applique Mur ordre 1 sur les 4 bords extérieurs.
    Cela approxime une onde sortante avec peu de réflexion.
    """

    # Bord gauche x = 0
    p_next[0, 1:-1] = (
        p_curr[1, 1:-1]
        + alpha * (p_next[1, 1:-1] - p_curr[0, 1:-1])
    )

    # Bord droit x = Lx
    p_next[-1, 1:-1] = (
        p_curr[-2, 1:-1]
        + alpha * (p_next[-2, 1:-1] - p_curr[-1, 1:-1])
    )

    # Bord bas y = 0
    p_next[1:-1, 0] = (
        p_curr[1:-1, 1]
        + alpha * (p_next[1:-1, 1] - p_curr[1:-1, 0])
    )

    # Bord haut y = Ly
    p_next[1:-1, -1] = (
        p_curr[1:-1, -2]
        + alpha * (p_next[1:-1, -2] - p_curr[1:-1, -1])
    )

    # Coins : moyenne simple des deux formules adjacentes
    p_next[0, 0] = 0.5 * (p_next[0, 1] + p_next[1, 0])
    p_next[0, -1] = 0.5 * (p_next[0, -2] + p_next[1, -1])
    p_next[-1, 0] = 0.5 * (p_next[-2, 0] + p_next[-1, 1])
    p_next[-1, -1] = 0.5 * (p_next[-2, -1] + p_next[-1, -2])

# -----------------------------
# 7) Sauvegarde dataset
# -----------------------------
def append_snapshot(records: list[np.ndarray],
                    sample_idx: int,
                    t: float,
                    field: np.ndarray) -> None:
    """
    Sauvegarde snapshot sous forme tabulaire :
    [sample_id, x, y, t, pression]
    """
    v = field.ravel()
    records.append(
        np.column_stack([
            np.full(v.shape, sample_idx, dtype=float),
            X.ravel(),
            Y.ravel(),
            np.full(v.shape, t, dtype=float),
            v
        ])
    )

# -----------------------------
# 8) Une simulation
# -----------------------------
def run_one_simulation(sample_idx: int,
                       obstacle_mask: np.ndarray,
                       sigma: float = 0.04,
                       amplitude: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Lance une simulation avec :
    - source gaussienne au centre à t=0
    - obstacle rigide
    - bords absorbants
    """
    # Condition initiale de pression
    p0 = gaussian_pulse_centered(X, Y, x0=0.5, y0=0.5,
                                 sigma=sigma, amplitude=amplitude)

    # Pas de vitesse initiale :
    # p_prev = p_curr = p0
    p_prev = p0.copy()
    p_curr = p0.copy()
    p_next = np.zeros_like(p0)

    # Pas de champ dans l'obstacle
    p_prev[obstacle_mask] = 0.0
    p_curr[obstacle_mask] = 0.0

    records: list[np.ndarray] = []
    append_snapshot(records, sample_idx, 0.0, p_curr)

    for step in range(n_steps):
        lap = laplacian_with_rigid_obstacle(p_curr, obstacle_mask)

        # Mise à jour explicite sur tout le domaine
        p_next[:, :] = 2.0 * p_curr - p_prev + (c * dt) ** 2 * lap

        # L'obstacle reste exclu du domaine fluide
        p_next[obstacle_mask] = 0.0

        # Bords absorbants espace libre approx.
        apply_mur_boundaries(p_next, p_curr)

        # Re-forcer l'obstacle après les bords
        p_next[obstacle_mask] = 0.0

        t = (step + 1) * dt
        if (step % save_every == 0) or (step == n_steps - 1):
            append_snapshot(records, sample_idx, t, p_next)

        # Rotation temporelle
        p_prev, p_curr, p_next = p_curr, p_next, p_prev

    return np.vstack(records), p_curr

# -----------------------------
# 9) Boucle dataset
# -----------------------------
all_data: list[np.ndarray] = []
all_obstacles_meta: list[np.ndarray] = []

for sample_idx in range(n_samples):
    obstacle_mask = make_single_rect_obstacle_mask(rng)

    data, last_field = run_one_simulation(sample_idx, obstacle_mask)
    all_data.append(data)

    # Sauvegarde aussi le masque pour post-traitement
    np.save(f"obstacle_mask_sample{sample_idx}.npy", obstacle_mask.astype(np.uint8))

    # Metadata simple : bbox obstacle
    coords = np.argwhere(obstacle_mask)
    ix_min, iy_min = coords.min(axis=0)
    ix_max, iy_max = coords.max(axis=0)

    meta = np.array([
        sample_idx,
        x[ix_min], y[iy_min],
        x[ix_max], y[iy_max]
    ], dtype=float)
    all_obstacles_meta.append(meta)

    print(
        f"sample {sample_idx}: obstacle "
        f"x=[{x[ix_min]:.3f},{x[ix_max]:.3f}] "
        f"y=[{y[iy_min]:.3f},{y[iy_max]:.3f}]"
    )

    if preview:
        # Pression finale
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        im0 = axes[0].imshow(
            last_field.T,
            origin="lower",
            extent=[0, Lx, 0, Ly],
            aspect="equal",
            cmap="coolwarm"
        )
        axes[0].set_title(f"Champ final — sample {sample_idx}")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        fig.colorbar(im0, ax=axes[0], label="Pression")

        im1 = axes[1].imshow(
            obstacle_mask.T,
            origin="lower",
            extent=[0, Lx, 0, Ly],
            aspect="equal",
            cmap="Greys"
        )
        axes[1].plot(0.5, 0.5, "r*", markersize=12, label="source")
        axes[1].set_title(f"Obstacle — sample {sample_idx}")
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("y (m)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"preview_sample{sample_idx}.png", dpi=150)
        plt.close(fig)

# -----------------------------
# 10) Sauvegardes finales
# -----------------------------
dataset = np.vstack(all_data)
np.save("pressure_dataset.npy", dataset)

obstacles_meta = np.vstack(all_obstacles_meta)
# colonnes : [sample_id, x_min, y_min, x_max, y_max]
np.save("obstacles_metadata.npy", obstacles_meta)

print("\nFichiers générés :")
print("  - pressure_dataset.npy")
print("  - obstacles_metadata.npy")
print("  - obstacle_mask_sample{i}.npy")
print("  - preview_sample{i}.png")

print("\nColonnes pressure_dataset.npy :")
print("  [sample_id, x, y, t, pression]")

print("\nTerminé.")