from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loop import TrainHistory


def plot_opf_history(*, history, plots_dir: str) -> None:
    """Save OPF training curves: total loss + component losses."""
    os.makedirs(plots_dir, exist_ok=True)

    n_train = len(history.train_loss)
    epochs_train = list(range(1, n_train + 1))
    # val runs every val_every epochs; reconstruct approximate epoch indices
    n_val = len(history.val_loss)
    if n_val > 0:
        step = max(1, n_train // n_val)
        epochs_val = [step * (i + 1) for i in range(n_val)]
    else:
        epochs_val = []

    # ── Total loss ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs_train, history.train_loss, label="train")
    if epochs_val:
        axes[0].plot(epochs_val, history.val_loss, "--", label="val")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].set_title("OPF Total Loss")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    # ── Component losses (gen_cost + kcl) ────────────────────────────────
    axes[1].plot(epochs_train, history.train_gen_cost, label="train gen_cost")
    axes[1].plot(epochs_train, history.train_kcl, label="train kcl_loss")
    if epochs_val:
        axes[1].plot(epochs_val, history.val_gen_cost, "--", label="val gen_cost")
        axes[1].plot(epochs_val, history.val_kcl, "--", label="val kcl_loss")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Component loss (unweighted)")
    axes[1].set_title("OPF Component Losses")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{plots_dir}/opf_loss.png", dpi=120)
    plt.close(fig)

    # ── Branch loss ───────────────────────────────────────────────────────
    if any(v > 0 for v in history.train_branch):
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(epochs_train, history.train_branch, label="train branch_loss")
        if epochs_val:
            ax2.plot(epochs_val, history.val_branch, "--", label="val branch_loss")
        ax2.set_yscale("log")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Branch thermal loss")
        ax2.set_title("Branch Thermal Constraint")
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(f"{plots_dir}/opf_branch.png", dpi=120)
        plt.close(fig2)


def plot_supervised_history(*, history, plots_dir: str) -> None:
    """Save supervised OPF training curves: total loss + v_mse + obj_gap."""
    os.makedirs(plots_dir, exist_ok=True)

    n_train = len(history.train_loss)
    epochs_train = list(range(1, n_train + 1))
    n_val = len(history.val_loss)
    step = max(1, n_train // n_val) if n_val > 0 else 1
    epochs_val = [step * (i + 1) for i in range(n_val)] if n_val > 0 else []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs_train, history.train_loss, label="train")
    if epochs_val:
        axes[0].plot(epochs_val, history.val_loss, "--", label="val")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total loss")
    axes[0].set_title("Supervised Total Loss"); axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(epochs_train, history.train_v_mse, label="train v_mse")
    axes[1].plot(epochs_train, history.train_pg_mse, label="train pg_mse")
    if epochs_val:
        axes[1].plot(epochs_val, history.val_v_mse, "--", label="val v_mse")
        axes[1].plot(epochs_val, history.val_pg_mse, "--", label="val pg_mse")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE vs labels")
    axes[1].set_title("Voltage & Dispatch MSE"); axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].plot(epochs_train, [g * 100 for g in history.train_obj_gap], label="train gap %")
    if epochs_val:
        axes[2].plot(epochs_val, [g * 100 for g in history.val_obj_gap], "--", label="val gap %")
    axes[2].axhline(0, color="k", linestyle=":", linewidth=0.8)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Objective gap (%)")
    axes[2].set_title("PV Objective Gap"); axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{plots_dir}/supervised_loss.png", dpi=120)
    plt.close(fig)


def plot_history(*, history: TrainHistory, pinn: bool, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)

    epochs = range(1, len(history.train_loss) + 1)

    if pinn:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history.train_loss, label="Train Physics Loss")
        plt.plot(epochs[: len(history.val_loss)], history.val_loss, label="Validation Physics Loss")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("PINN: Physics Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/physics_loss.png")
        plt.clf()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history.train_rmse, label="Train RMSE (phasor/all)")
    plt.plot(epochs[: len(history.val_rmse)], history.val_rmse, label="Val RMSE (phasor/all)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Supervised RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rmse_total.png")
    plt.clf()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(epochs, history.train_rmse_mag, label="Train |V|")
    ax[0].plot(epochs[: len(history.val_rmse_mag)], history.val_rmse_mag, label="Val |V|")
    ax[0].set_title("Magnitude RMSE")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(epochs, history.train_rmse_ang_deg, label="Train θ (deg)")
    ax[1].plot(epochs[: len(history.val_rmse_ang_deg)], history.val_rmse_ang_deg, label="Val θ (deg)")
    ax[1].set_title("Angle RMSE (degrees)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    fig.suptitle("Magnitude vs Angle RMSE")
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/rmse_components.png")
    plt.close(fig)
