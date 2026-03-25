# PINN — 2D Acoustic Wave

Physics-Informed Neural Network (PINN) solving the 2D acoustic wave equation on the unit square. Data come from a finite-difference simulator; training blends observation loss, physics residual, and boundary conditions.

## Repository Layout
- `pinn.py` — PINN definition, losses (data/physics/boundary), and two-stage training (Adam → L-BFGS) that saves `pinn_wave_model.pt`.
- `simu/simu1_feni.py` — NumPy finite-difference simulator creating training data (with and without obstacles) plus preview plots.
- `simu/pinn_ground_truth.npy` — dataset without obstacles; columns: `sample_id, x, y, t, pressure`.
- `simu/pinn_ground_truth_obstacles.npy` — dataset with random square obstacles; same column order.
- `simu/preview*.png` — example pressure fields and obstacle masks.

## Requirements
- Python 3.10+
- PyTorch (CUDA if available), NumPy, Matplotlib

Quick install (edit the torch line to match your CUDA/CPU build):
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
```

## Generate Synthetic Data (optional)
From the repo root:
```bash
python simu/simu1_feni.py
```
Environment knobs (defaults in parentheses):
- `SIM_SAMPLES` (3) — simulations without obstacles
- `SIM_OBS_SAMPLES` (3) — simulations with obstacles
- `SIM_N_OBS` (4) — obstacle count per simulation
- `SIM_SEED` (0) — RNG seed
- `SIM_PREVIEW` (1) — set to 0 to skip PNG previews

Outputs: refreshed `pinn_ground_truth.npy`, `pinn_ground_truth_obstacles.npy`, and preview PNGs.

## Train the PINN
```bash
python pinn.py
```
Notes:
- Automatically uses GPU if available (`torch.cuda.is_available()`).
- Dataset path expected: `simu/pinn_ground_truth.npy` (edit in `pinn.py` if you prefer the obstacle set).
- Collocation/boundary points sampled on the fly; prints losses every 500 epochs.
- Training schedule: Adam 5000 epochs (lr=1e-3) then L-BFGS (500 steps). Model saved as `pinn_wave_model.pt`.

## Model & Physics
- Network: fully connected, input `(x,y,t)`, 5 hidden layers of 128 with `tanh`, Xavier init, scalar pressure output.
- Physics term: wave residual `∂²p/∂t² - c²(∂²p/∂x² + ∂²p/∂y²)` via autograd.
- Defaults: sound speed `c=343 m/s`, time horizon `T_final=0.01 s`, domain `[0,1]²`, Dirichlet boundaries (pressure=0) on edges and obstacles.
- Loss weights: `w_data=1.0`, `w_phys=1e-3`, `w_bc=1.0` (tune `w_phys` upward if the model ignores physics; downward if it dominates).

## Tips
- Increase `n_points` in `load_dataset` or collocation/boundary counts for higher fidelity (with more compute).
- Use the obstacle dataset to test generalization to complex geometries.
- Monitor `loss_phys` for PDE satisfaction; if it stalls, consider raising `w_phys` or adding collocation points.
