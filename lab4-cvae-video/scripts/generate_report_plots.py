import io
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ASSETS = ROOT / "report" / "assets"


def find_event_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No event file found in {run_dir}")
    return candidates[-1]


def load_event(run_dir: Path) -> EventAccumulator:
    event_file = find_event_file(run_dir)
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0, "images": 0})
    acc.Reload()
    return acc


def read_scalar(acc: EventAccumulator, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    if tag not in acc.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = acc.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.float64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return steps, values


def smooth(y: np.ndarray, window: int = 31) -> np.ndarray:
    if y.size == 0 or window <= 1:
        return y
    window = min(window, y.size)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return y
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y_pad, kernel, mode="valid")


def trim_initial_large_values(x: np.ndarray, y: np.ndarray, pct: float = 95.0, max_trim_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Trim contiguous outliers at the beginning of a curve.

    This is used for training loss where the first few steps can be extremely large
    and make the rest of the curve unreadable.
    """
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return x, y

    threshold = np.nanpercentile(y, pct)
    max_trim = int(len(y) * max_trim_ratio)
    start = 0
    while start < len(y) - 1 and start < max_trim and y[start] > threshold:
        start += 1

    return x[start:], y[start:]


def best_run_by_val_psnr(prefix: str) -> Path:
    candidates = sorted([p for p in RUNS.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not candidates:
        raise FileNotFoundError(f"No run found for prefix: {prefix}")

    best_dir = None
    best_psnr = -np.inf
    for run_dir in candidates:
        try:
            acc = load_event(run_dir)
        except Exception:
            continue
        _, psnr = read_scalar(acc, "Metrics/val/psnr")
        if psnr.size == 0:
            continue
        score = float(np.nanmax(psnr))
        if score > best_psnr:
            best_psnr = score
            best_dir = run_dir

    if best_dir is None:
        raise RuntimeError(f"Found candidates but none has Metrics/val/psnr for prefix: {prefix}")
    return best_dir


def save_tfr_and_loss(run_dir: Path, out_path: Path) -> None:
    acc = load_event(run_dir)
    x_tfr, tfr = read_scalar(acc, "Metrics/train/tfr")
    x_loss, train_loss = read_scalar(acc, "Loss/train/total")

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    c1 = "tab:blue"
    c2 = "tab:red"

    # Convert TFR epoch coordinates to step scale
    x_tfr_scaled = x_tfr
    if tfr.size and x_loss.size > 1 and x_tfr.size > 1 and np.nanmax(x_tfr) > 0 and np.nanmax(x_loss) > 0:
        # Scale TFR epochs to match loss step range
        x_tfr_scaled = x_tfr / np.nanmax(x_tfr) * np.nanmax(x_loss)
    
    if tfr.size:
        ax1.plot(x_tfr_scaled, tfr, color=c1, linewidth=2.1, label="TFR")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Teacher Forcing Ratio", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    if train_loss.size:
        x_loss_trim, y_loss_trim = trim_initial_large_values(x_loss, train_loss, pct=95.0, max_trim_ratio=0.2)
        y_loss = smooth(y_loss_trim, window=51)
        ax2.plot(x_loss_trim, y_loss, color=c2, linewidth=1.8, label="Train Total Loss (smoothed)")
    ax2.set_ylabel("Train Total Loss", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.set_ylim(0.0, 0.01)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    plt.title("Teacher Forcing Ratio and Training Loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_tfr_vs_psnr(best_tf_dir: Path, out_path: Path) -> None:
    acc = load_event(best_tf_dir)
    x_tfr, tfr = read_scalar(acc, "Metrics/train/tfr")
    x_psnr, psnr = read_scalar(acc, "Metrics/val/psnr")

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    c1 = "tab:blue"
    c2 = "tab:orange"

    if tfr.size:
        ax1.plot(x_tfr, tfr, color=c1, linewidth=2.1, label="TFR")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Teacher Forcing Ratio", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    if psnr.size:
        ax2.plot(x_psnr, psnr, color=c2, linewidth=2.1, label="Val PSNR")
    ax2.set_ylabel("Validation PSNR (dB)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    plt.title("Teacher Forcing Ratio vs Validation PSNR")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_kl_compare(mono_dir: Path, cyc_dir: Path, fixed_dir: Path, out_path: Path) -> None:
    run_defs = [
        ("Monotonic KL", mono_dir, "tab:green"),
        ("Cyclical KL", cyc_dir, "tab:red"),
        ("Fixed KL (No Annealing)", fixed_dir, "tab:purple"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    tags = [
        ("Loss/train/total", "Train Total Loss", True),
        ("Loss/train/kl", "Train KL Loss", True),
        ("Metrics/val/psnr", "Validation PSNR", False),
    ]

    for ax, (tag, title, do_smooth) in zip(axes, tags):
        for label, run_dir, color in run_defs:
            acc = load_event(run_dir)
            x, y = read_scalar(acc, tag)
            if y.size == 0:
                continue
            if tag.startswith("Loss/train"):
                x, y = trim_initial_large_values(x, y)
            if do_smooth:
                y_plot = smooth(y, window=51)
            else:
                y_plot = y
            ax.plot(x, y_plot, label=label, color=color, linewidth=2.0)

        ax.set_title(title)
        ax.set_xlabel("Step" if tag.startswith("Loss/train") else "Epoch")
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Loss")
    axes[2].set_ylabel("PSNR (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_psnr_per_frame(best_tf_dir: Path, out_path: Path) -> Image.Image:
    acc = load_event(best_tf_dir)
    tag = "Metrics/val/psnr_per_frame_curve"
    if tag not in acc.Tags().get("images", []):
        raise RuntimeError(f"Image tag '{tag}' not found in {best_tf_dir}")

    images = acc.Images(tag)
    if not images:
        raise RuntimeError(f"No images under tag '{tag}' in {best_tf_dir}")

    # Use the last logged validation image (most recent)
    encoded = images[-1].encoded_image_string
    img = Image.open(io.BytesIO(encoded)).convert("RGB")
    img.save(out_path)
    return img


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    target_tf_dir = RUNS / "shuffle_tf_epoch_tfr0p7_sde10_d0p05_Apr25_141916"
    if not target_tf_dir.exists():
        raise FileNotFoundError(f"Target run not found: {target_tf_dir}")

    mono_dir = RUNS / "cosanl_mono_Apr19_044416"
    cyc_dir = RUNS / "cosanl_lr8e-4_Apr20_115816"
    fixed_dir = RUNS / "cosanl_fixed_Apr20_165136"

    # save_tfr_and_loss(target_tf_dir, ASSETS / "tfr_curve.png")
    # save_tfr_vs_psnr(target_tf_dir, ASSETS / "tfr_vs_psnr.png")
    save_kl_compare(mono_dir, cyc_dir, fixed_dir, ASSETS / "loss_curve_kl_compare.png")
    # save_psnr_per_frame(target_tf_dir, ASSETS / "psnr_per_frame.png")

    print(f"[OK] Target TF run: {target_tf_dir.name}")
    print("[OK] Saved report figures:")
    for name in [
        "tfr_curve.png",
        # "tfr_vs_psnr.png",
        # "loss_curve_kl_compare.png",
        # "psnr_per_frame.png",
    ]:
        print(f"  - report/assets/{name}")


if __name__ == "__main__":
    main()
