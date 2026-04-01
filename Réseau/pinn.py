# IA Project, M2FESUP 2026
# @Pilou&Nietzsche
# 24/03/2026
# ---------- PINN — Équation d'onde acoustique 2D ----------

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm 
# ── 1. Architecture ───────────────────────────────────────────────────────────

class FCN(nn.Module):
    """
    Fully Connected Network pour PINN acoustique 2D.
    Entrée  : (x, y, t)  ∈ [0,1]² × [0, T]
    Sortie  : p(x, y, t) ∈ ℝ

    Choix tanh : dérivées d'ordre 2 non nulles partout → requis pour ∂²p/∂t² et Δp.
    Choix largeur uniforme : évite les goulots d'étranglement qui écrasent le gradient.
    """
    def __init__(self, hidden_dim: int = 128, n_layers: int = 5):
        super().__init__()

        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]   # pas d'activation : régression

        self.net = nn.Sequential(*layers)

        # Initialisation Xavier — stabilise les gradients avec tanh
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, y, t], dim=1)   # (N, 3)
        return self.net(inputs)                 # (N, 1)


# ── 2. Résidu physique (équation d'onde) ──────────────────────────────────────

def wave_residual(model: FCN, x: torch.Tensor, y: torch.Tensor,
                  t: torch.Tensor, c: float) -> torch.Tensor:
    """
    Calcule le résidu : ∂²p/∂t² − c²·(∂²p/∂x² + ∂²p/∂y²) = 0
    via autograd (différentiation automatique).
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    p = model(x, y, t)

    # Dérivées premières
    grad_p = torch.autograd.grad(p.sum(), [x, y, t], create_graph=True)
    dp_dx, dp_dy, dp_dt = grad_p

    # Dérivées secondes
    dp_dxx = torch.autograd.grad(dp_dx.sum(), x, create_graph=True)[0]
    dp_dyy = torch.autograd.grad(dp_dy.sum(), y, create_graph=True)[0]
    dp_dtt = torch.autograd.grad(dp_dt.sum(), t, create_graph=True)[0]

    residual = dp_dtt - c**2 * (dp_dxx + dp_dyy)
    return residual


# ── 3. Loss totale ────────────────────────────────────────────────────────────

def compute_loss(model: FCN,
                 # Points de données (ground truth NumPy)
                 x_data: torch.Tensor, y_data: torch.Tensor,
                 t_data: torch.Tensor, p_data: torch.Tensor,
                 # Points de collocation (physique)
                 x_col: torch.Tensor, y_col: torch.Tensor, t_col: torch.Tensor,
                 # Points de bord
                 x_bc: torch.Tensor, y_bc: torch.Tensor, t_bc: torch.Tensor,
                 c: float,
                 w_data: float = 1.0,
                 w_phys: float = 1e-3,
                 w_bc:   float = 1.0) -> dict[str, torch.Tensor]:
    """
    L_total = w_data · L_data + w_phys · L_physique + w_bc · L_bords

    Les poids w_* sont à ajuster selon la convergence :
    - Si L_phys domine → baisser w_phys
    - Si le réseau ignore la physique → monter w_phys
    """
    # L_data : erreur sur les observations
    p_pred = model(x_data, y_data, t_data)
    loss_data = torch.mean((p_pred - p_data) ** 2)

    # L_physique : résidu de l'équation d'onde sur points de collocation
    res = wave_residual(model, x_col, y_col, t_col, c)
    loss_phys = torch.mean(res ** 2)

    # L_bords : condition de Dirichlet p = 0 sur les bords (et obstacles)
    p_bc = model(x_bc, y_bc, t_bc)
    loss_bc = torch.mean(p_bc ** 2)

    loss_total = w_data * loss_data + w_phys * loss_phys + w_bc * loss_bc

    return {
        "total": loss_total,
        "data":  loss_data,
        "phys":  loss_phys,
        "bc":    loss_bc,
    }


# ── 4. Chargement du dataset ──────────────────────────────────────────────────

def load_dataset(path: str, n_points: int = 5000) -> tuple:
    """
    Charge pinn_ground_truth.npy et tire n_points aléatoires.
    Colonnes attendues : [sample_id, x, y, t, pression]
    """
    data = np.load(path)                     # (N, 5)
    idx  = np.random.choice(len(data), size=min(n_points, len(data)), replace=False)
    data = data[idx]

    def t(col): return torch.tensor(data[:, col], dtype=torch.float32).unsqueeze(1)
    return t(1), t(2), t(3), t(4)           # x, y, t, p


# ── 5. Points de collocation & bords ─────────────────────────────────────────

def sample_collocation(n: int, T: float) -> tuple:
    x = torch.rand(n, 1)
    y = torch.rand(n, 1)
    t = torch.rand(n, 1) * T
    return x, y, t

def sample_boundary(n: int, T: float) -> tuple:
    """Points sur les 4 bords du domaine [0,1]²."""
    n4 = n // 4
    def edge(fixed_dim, fixed_val):
        pts = torch.rand(n4, 1)
        t   = torch.rand(n4, 1) * T
        if fixed_dim == "x0": return torch.zeros(n4,1), pts, t
        if fixed_dim == "x1": return torch.ones(n4,1),  pts, t
        if fixed_dim == "y0": return pts, torch.zeros(n4,1), t
        if fixed_dim == "y1": return pts, torch.ones(n4,1),  t

    parts = [edge(d, v) for d, v in [("x0",0),("x1",1),("y0",0),("y1",1)]]
    x_bc = torch.cat([p[0] for p in parts])
    y_bc = torch.cat([p[1] for p in parts])
    t_bc = torch.cat([p[2] for p in parts])
    return x_bc, y_bc, t_bc


# ── 6. Boucle d'entraînement ──────────────────────────────────────────────────

if __name__ == "__main__":
    c       = 343.0
    T_final = 0.01
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = FCN(hidden_dim=128, n_layers=5).to(device)

    # Données
    x_d, y_d, t_d, p_d = load_dataset("simu/pinn_ground_truth_fixed_obstacle.npy", n_points=8000)
    x_d, y_d, t_d, p_d = x_d.to(device), y_d.to(device), t_d.to(device), p_d.to(device)

    # Points de collocation et bords (rééchantillonnés à chaque époque possible)
    x_col, y_col, t_col = sample_collocation(10_000, T_final)
    x_bc,  y_bc,  t_bc  = sample_boundary(2_000, T_final)
    x_col, y_col, t_col = x_col.to(device), y_col.to(device), t_col.to(device)
    x_bc,  y_bc,  t_bc  = x_bc.to(device),  y_bc.to(device),  t_bc.to(device)

    # ── Phase 1 : Adam ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("\nPhase 1 — Adam (5000 époques)")

    for epoch in tqdm(range(5000)):
        optimizer.zero_grad()
        losses = compute_loss(model,
                              x_d, y_d, t_d, p_d,
                              x_col, y_col, t_col,
                              x_bc, y_bc, t_bc,
                              c=c)
        losses["total"].backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  [{epoch:5d}] total={losses['total'].item():.2e} "
                  f"data={losses['data'].item():.2e} "
                  f"phys={losses['phys'].item():.2e} "
                  f"bc={losses['bc'].item():.2e}")

    # ── Phase 2 : L-BFGS (affinage) ──────────────────────────────────────────
    print("\nPhase 2 — L-BFGS (500 steps)")
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0,
                               max_iter=5, history_size=50,
                               line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        losses = compute_loss(model,
                              x_d, y_d, t_d, p_d,
                              x_col, y_col, t_col,
                              x_bc, y_bc, t_bc,
                              c=c)
        losses["total"].backward()
        return losses["total"]

    lbfgs.step(closure)
    print("  L-BFGS terminé.")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "pinn_wave_model.pt")
    print("\nModèle sauvegardé : pinn_wave_model.pt")