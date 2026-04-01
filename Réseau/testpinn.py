#!/usr/bin/env python
# Quick evaluation script for the 2D acoustic PINN

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from PINN.pinn import FCN


def predict_field(model: FCN, t: float, nx: int, device: torch.device) -> np.ndarray:
    """Return predicted pressure grid (nx×nx) at time t on [0,1]²."""
    xs = torch.linspace(0.0, 1.0, nx, device=device)
    ys = torch.linspace(0.0, 1.0, nx, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    t_tensor = torch.full_like(X, fill_value=t)

    with torch.no_grad():
        pred = model(X.reshape(-1, 1), Y.reshape(-1, 1), t_tensor.reshape(-1, 1))
    return pred.view(nx, nx).cpu().numpy()


def load_snapshot(dataset_path: Path, sample_id: int, target_t: float, nx: int) -> tuple[np.ndarray, float]:
    """
    Wrapper that loads the .npy file then delegates to load_snapshot_array.
    """
    data = np.load(dataset_path)
    return load_snapshot_array(data, dataset_path, sample_id, target_t, nx)


def load_snapshot_array(data: np.ndarray, dataset_path: Path | None, sample_id: int, target_t: float, nx: int) -> tuple[np.ndarray, float]:
    """
    Load the ground-truth grid for a given sample_id at the time closest to target_t.
    Returns (grid, matched_t).
    """
    mask_sample = data[:, 0] == sample_id
    if not mask_sample.any():
        raise ValueError(f"sample_id {sample_id} not found in {dataset_path}")

    times = data[mask_sample, 3]
    unique_times = np.unique(times)
    matched_t = float(unique_times[np.argmin(np.abs(unique_times - target_t))])

    rows = data[mask_sample & (np.abs(data[:, 3] - matched_t) < 1e-12)]
    if len(rows) != nx * nx:
        raise ValueError(f"Expected {nx*nx} grid points, got {len(rows)}. Check nx or dataset.")

    grid = rows[:, 4].reshape(nx, nx)
    return grid, matched_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a trained PINN on the 2D acoustic wave.")
    parser.add_argument("--weights", default="pinn_wave_model.pt", help="Path to model weights.")
    parser.add_argument("--dataset", default=None, help="Optional .npy dataset to compare against.")
    parser.add_argument("--sample-id", type=int, default=0, help="Sample id to compare if dataset is provided.")
    parser.add_argument("--t", type=float, default=0.005, help="Time (seconds) for prediction.")
    parser.add_argument("--nx", type=int, default=81, help="Grid resolution (matches training sims).")
    parser.add_argument("--out", default="pred_field.png", help="Output PNG for the prediction.")
    parser.add_argument("--out-dir", default="eval_outputs", help="Output directory when evaluating multiple samples.")
    parser.add_argument("--device", default=None, help="Force device, e.g., cpu or cuda. Default: auto.")
    parser.add_argument("--eval-many", type=int, default=0, help="If >0, evaluate first N sample_ids found in dataset and save per-sample comparisons.")
    parser.add_argument("--curve", action="store_true", help="Plot MSE vs time for the selected sample-id.")
    parser.add_argument("--curve-out", default="mse_vs_time.png", help="Filename for the MSE-vs-time plot.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = FCN(hidden_dim=128, n_layers=5).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset_path: Path | None = Path(args.dataset) if args.dataset else None
    dataset_data: np.ndarray | None = np.load(dataset_path) if dataset_path else None

    # If batch evaluation requested
    if args.eval_many > 0:
        if not args.dataset:
            raise SystemExit("--eval-many requires --dataset.")
        data = dataset_data  # already loaded
        sample_ids = sorted(np.unique(data[:, 0]).astype(int))[: args.eval_many]
        os.makedirs(args.out_dir, exist_ok=True)

        mses = []
        for sid in sample_ids:
            p_pred = predict_field(model, t=args.t, nx=args.nx, device=device)
            grid_true, matched_t = load_snapshot_array(data, dataset_path, sid, args.t, args.nx)
            mse = float(np.mean((p_pred - grid_true) ** 2))
            err = np.abs(p_pred - grid_true)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            im0 = axes[0].imshow(p_pred, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm", aspect="equal")
            axes[0].set_title(f"PINN @ t={args.t:.4f}s")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(grid_true, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm", aspect="equal")
            axes[1].set_title(f"GT @ t≈{matched_t:.4f}s (sample {sid})")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(err, origin="lower", extent=[0, 1, 0, 1], cmap="magma", aspect="equal")
            axes[2].set_title(f"|Erreur| MSE={mse:.2e}")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            cs = axes[0].contour(grid_true, levels=8, colors="k", linewidths=0.6,
                                 origin="lower", extent=[0, 1, 0, 1])
            axes[0].clabel(cs, inline=True, fontsize=7, fmt="%.2f")

            out_path = Path(args.out_dir) / f"sample_{sid}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            mses.append(mse)
            print(f"[sample {sid}] MSE={mse:.4e} | saved {out_path}")

        mses = np.array(mses)
        print(f"\nSummary over {len(mses)} samples:")
        print(f"  mean MSE = {mses.mean():.4e}")
        print(f"  median   = {np.median(mses):.4e}")
        print(f"  max      = {mses.max():.4e}")
        return

    # Single-sample flow (default)
    p_pred = predict_field(model, t=args.t, nx=args.nx, device=device)
    mse = None
    matched_t = None

    if args.dataset:
        grid_true, matched_t = load_snapshot(dataset_path, args.sample_id, args.t, args.nx)
        mse = float(np.mean((p_pred - grid_true) ** 2))
        err = np.abs(p_pred - grid_true)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        im0 = axes[0].imshow(p_pred, origin="lower", extent=[0, 1, 0, 1],
                             cmap="coolwarm", aspect="equal")
        axes[0].set_title(f"PINN @ t={args.t:.4f}s")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(grid_true, origin="lower", extent=[0, 1, 0, 1],
                             cmap="coolwarm", aspect="equal")
        axes[1].set_title(f"GT @ t≈{matched_t:.4f}s (sample {args.sample_id})")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(err, origin="lower", extent=[0, 1, 0, 1],
                             cmap="magma", aspect="equal")
        axes[2].set_title(f"|Erreur| MSE={mse:.2e}")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        cs = axes[0].contour(grid_true, levels=8, colors="k", linewidths=0.6,
                             origin="lower", extent=[0, 1, 0, 1])
        axes[0].clabel(cs, inline=True, fontsize=7, fmt="%.2f")

        fig.savefig(args.out, dpi=150)
        print(f"MSE at t≈{matched_t:.4f}s (sample {args.sample_id}): {mse:.4e}")
        print(f"Saved comparison figure to {args.out}")

        # Optional MSE vs time curve on same sample
        if args.curve:
            times = np.unique(dataset_data[dataset_data[:, 0] == args.sample_id][:, 3])
            mses_curve = []
            for tt in times:
                p_pred_t = predict_field(model, t=float(tt), nx=args.nx, device=device)
                grid_true_t, _ = load_snapshot_array(dataset_data, dataset_path, args.sample_id, float(tt), args.nx)
                mses_curve.append(float(np.mean((p_pred_t - grid_true_t) ** 2)))

            fig_curve, ax_curve = plt.subplots(figsize=(6, 4))
            ax_curve.plot(times, mses_curve, marker="o")
            ax_curve.set_xlabel("Temps (s)")
            ax_curve.set_ylabel("MSE")
            ax_curve.set_title(f"MSE vs temps — sample {args.sample_id}")
            ax_curve.grid(True, alpha=0.3)
            fig_curve.tight_layout()
            fig_curve.savefig(args.curve_out, dpi=150)
            plt.close(fig_curve)
            print(f"MSE curve saved to {args.curve_out}")
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(p_pred, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm", aspect="equal")
        ax.set_title(f"PINN prediction at t={args.t:.4f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="Pression (a.u.)")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"Saved prediction figure to {args.out}")

    if mse is None:
        print("No dataset provided → only prediction plotted.")


if __name__ == "__main__":
    main()
