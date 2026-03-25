# IA Project, M2FESUP 2026
# @Pilou&Nietzsche
# 24/03/2026
# ---------- Simulation acoustique 2D — NumPy only (sans MPI/FEniCSx) ----------
# Variable sources, fixed obstacle
# no wall rebound 
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_center(raw: str) -> tuple[float, float]:
    """Convertit 'x,y' en tuple, fallback (0.65, 0.5) en cas d'erreur."""
    try:
        a, b = raw.split(",")[:2]
        return float(a), float(b)
    except Exception:
        return 0.65, 0.5

# ── 1. Paramètres ─────────────────────────────────────────────────────────────

c        = 343.0              # vitesse du son (m/s)
nx = ny  = 81                 # points de grille (80 cellules)
dx       = 1.0 / (nx - 1)
dt       = min(2e-5, 0.9 * dx / (c * np.sqrt(2)))   # condition CFL
T_final  = 0.01               # durée totale (s)
n_steps  = int(np.ceil(T_final / dt))
save_every = 5                # fréquence de sauvegarde des pas

n_samples     = int(os.environ.get("SIM_SAMPLES",   "3"))
n_obs_samples = int(os.environ.get("SIM_OBS_SAMPLES", "3"))  # simulations avec obstacles
n_obstacles   = int(os.environ.get("SIM_N_OBS",     "4"))   # obstacles par simulation
_fixed_center_env = os.environ.get("SIM_FIXED_CENTER", "0.65,0.5")
fixed_obs_center = _parse_center(_fixed_center_env)
fixed_obs_size = int(os.environ.get("SIM_FIXED_SIZE", "8"))    # largeur en cellules pour l'obstacle fixe
n_fixed_samples = int(os.environ.get("SIM_FIXED_SAMPLES", "3"))
seed          = int(os.environ.get("SIM_SEED",       "0"))
preview       = os.environ.get("SIM_PREVIEW", "1") != "0"

rng = np.random.default_rng(seed)

print(f"Grille : {nx}×{ny} | dx={dx:.4f} | dt={dt:.2e} | steps={n_steps}")

# ── 2. Grille spatiale ────────────────────────────────────────────────────────

X, Y = np.meshgrid(
    np.linspace(0.0, 1.0, nx),
    np.linspace(0.0, 1.0, ny),
    indexing="ij"
)

inv_dx2 = 1.0 / dx**2
c2dt2   = (c * dt)**2

# ── 3. Génération d'obstacles aléatoires ─────────────────────────────────────

def make_obstacle_mask(n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """
    Retourne un masque booléen (nx, ny) : True = obstacle (pression forcée à 0).
    Chaque obstacle est un carré de taille aléatoire placé aléatoirement,
    en évitant les bords et la zone centrale (source).
    """
    mask = np.zeros((nx, ny), dtype=bool)
    for _ in range(n_obs):
        # Taille de l'obstacle : entre 3 et 10 cellules
        size = rng.integers(3, 11)
        # Position : évite les 10% de bord et la zone centrale (±0.15)
        while True:
            ix = rng.integers(5, nx - size - 5)
            iy = rng.integers(5, ny - size - 5)
            # Centre en coordonnées normalisées
            cx_obs = (ix + size / 2) * dx
            cy_obs = (iy + size / 2) * dx
            if abs(cx_obs - 0.5) > 0.15 or abs(cy_obs - 0.5) > 0.15:
                break
        mask[ix:ix+size, iy:iy+size] = True
    return mask


def make_fixed_obstacle(center_xy: tuple[float, float], size_cells: int) -> np.ndarray:
    """Obstacle carré fixé par son centre (coordonnées normalisées) et sa taille en cellules."""
    mask = np.zeros((nx, ny), dtype=bool)
    size_cells = max(1, size_cells)
    cx, cy = center_xy

    ix_center = int(round(cx / dx))
    iy_center = int(round(cy / dx))

    ix0 = int(np.clip(ix_center - size_cells // 2, 1, nx - 2))
    ix1 = int(np.clip(ix0 + size_cells,             1, nx - 1))
    iy0 = int(np.clip(iy_center - size_cells // 2, 1, ny - 2))
    iy1 = int(np.clip(iy0 + size_cells,             1, ny - 1))

    mask[ix0:ix1, iy0:iy1] = True
    return mask

# ── 4. Simulation (une source, masque d'obstacles optionnel) ──────────────────

def run(sample_idx: int, cx: float, cy: float,
        obstacle_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Schéma différences finies centré, conditions de Dirichlet aux bords et obstacles."""
    p0     = np.exp(-200.0 * ((X - cx)**2 + (Y - cy)**2))
    p_prev = p0.copy()
    p_curr = p0.copy()
    p_next = np.zeros_like(p0)
    records: list[np.ndarray] = []

    def save(t: float, field: np.ndarray) -> None:
        v = field.ravel()
        records.append(np.column_stack([
            np.full_like(v, sample_idx),
            X.ravel(), Y.ravel(),
            np.full_like(v, t),
            v
        ]))

    save(0.0, p_curr)

    for step in range(n_steps):
        lap = (
            (p_curr[2:, 1:-1] - 2*p_curr[1:-1, 1:-1] + p_curr[:-2, 1:-1]) +
            (p_curr[1:-1, 2:] - 2*p_curr[1:-1, 1:-1] + p_curr[1:-1, :-2])
        ) * inv_dx2

        p_next[1:-1, 1:-1] = 2*p_curr[1:-1, 1:-1] - p_prev[1:-1, 1:-1] + c2dt2 * lap

        # Bords Dirichlet
        p_next[0,:] = p_next[-1,:] = p_next[:,0] = p_next[:,-1] = 0.0

        # Obstacles : pression nulle sur les cellules masquées
        if obstacle_mask is not None:
            p_next[obstacle_mask] = 0.0

        if step % save_every == 0 or step == n_steps - 1:
            save((step + 1) * dt, p_next)

        p_prev, p_curr, p_next = p_curr, p_next, p_prev

    return np.vstack(records), p_curr


# ── 7. Boucle C — obstacle fixe ─────────────────────────────────────────────

fixed_mask = make_fixed_obstacle(fixed_obs_center, fixed_obs_size)
all_data_fixed: list[np.ndarray] = []

print(f"\n── Boucle C : {n_fixed_samples} simulations avec obstacle fixe ──")
print(f"  obstacle : centre=({fixed_obs_center[0]:.3f}, {fixed_obs_center[1]:.3f}) | taille={fixed_obs_size} cellules | surface={fixed_mask.sum()} cases")

for i in range(n_fixed_samples):
    cx, cy = rng.uniform(0.1, 0.9, size=2)
    print(f"  sample {i} : source=({cx:.3f}, {cy:.3f})")
    data, last_field_C = run(i, cx, cy, obstacle_mask=fixed_mask)
    all_data_fixed.append(data)

    if preview:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Champ de pression
        im = axes[0].imshow(last_field_C, origin="lower", extent=[0,1,0,1],
                            cmap="coolwarm", aspect="equal")
        axes[0].set_title(f"Pression — obstacle fixe — sample {i}")
        axes[0].set_xlabel("x") ; axes[0].set_ylabel("y")
        fig.colorbar(im, ax=axes[0], label="Pression")
        # Superposer obstacle (gris transparent) et source (étoile rouge)
        axes[0].imshow(fixed_mask.T, origin="lower", extent=[0,1,0,1],
                       cmap="Greys", alpha=0.25, aspect="equal")
        axes[0].plot(cx, cy, "r*", markersize=10, label="source")
        axes[0].legend(loc="upper right", frameon=True)

        # Masque fixe
        axes[1].imshow(fixed_mask.T, origin="lower", extent=[0,1,0,1],
                       cmap="Greys", aspect="equal")
        axes[1].set_title("Obstacle fixe")
        axes[1].set_xlabel("x") ; axes[1].set_ylabel("y")
        axes[1].plot(cx, cy, "r*", markersize=12, label="source")
        axes[1].legend()

        plt.suptitle(f"Simulation obstacle fixe — sample {i}", fontsize=11)
        plt.tight_layout()
        fname = f"preview_fixed_obstacle_sample{i}.png"
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  → {fname} sauvegardé")

np.save("pinn_ground_truth_fixed_obstacle.npy", np.vstack(all_data_fixed))
print(f"  Dataset C sauvegardé : {np.vstack(all_data_fixed).shape}")
print("\nColonnes datasets : [sample_id, x, y, t, pression]")
print("Simulation terminée ✓")
