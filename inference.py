import os
import argparse
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import HierarchicalTemporalDecoder


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HierarchicalTemporalDecoder(
        label_dim=4, output_h=args.output_h, output_w=args.output_w,
        band_ch=args.band_ch, trend_base=args.trend_base,
        detail_base=args.detail_base,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    label = np.load(args.label_path).astype(np.float32)
    label_t = torch.from_numpy(label).unsqueeze(0).to(device)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    frames = []
    band_frames = {"static": [], "trend": [], "detail": []}

    with torch.no_grad():
        for t in range(args.num_frames):
            ts = torch.tensor([t], dtype=torch.float32, device=device)
            combined, static, trend, detail = model(label_t, ts, return_bands=True)
            frame = combined.squeeze().cpu().numpy()
            frames.append(frame)
            np.save(os.path.join(run_dir, f"pred_frame_{t:03d}.npy"), frame)
            band_frames["static"].append(static.squeeze().cpu().numpy())
            band_frames["trend"].append(trend.squeeze().cpu().numpy())
            band_frames["detail"].append(detail.squeeze().cpu().numpy())

    print(f"Saved {args.num_frames} frames to {run_dir}")

    if args.visualize:
        _visualize_grid(frames, run_dir)
        _visualize_filmstrip(frames, run_dir)
        _visualize_bands(band_frames, run_dir)


def _visualize_grid(frames, out_dir, n_show=25):
    """Save an NxN grid of evenly spaced frames."""
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n_show:
            ax.imshow(frames[indices[i]], cmap="viridis", aspect="auto")
            ax.set_title(f"t={indices[i]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Generated Time Series — Grid", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "grid.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Grid saved to {path}")


def _visualize_filmstrip(frames, out_dir, n_show=10):
    """Save a horizontal filmstrip of evenly spaced frames."""
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(1, n_show, figsize=(2.5 * n_show, 3))
    for i, ax in enumerate(axes):
        ax.imshow(frames[indices[i]], cmap="viridis", aspect="auto")
        ax.set_title(f"t={indices[i]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Generated Time Series — Filmstrip", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "filmstrip.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Filmstrip saved to {path}")


def _visualize_bands(band_frames, out_dir, n_show=8):
    """Show each temporal band at evenly-spaced timesteps."""
    indices = np.linspace(0, len(band_frames["static"]) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(3, n_show, figsize=(2.5 * n_show, 7.5))
    for row, name in enumerate(("static", "trend", "detail")):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(band_frames[name][idx], cmap="RdBu_r", aspect="auto")
            if row == 0:
                ax.set_title(f"t={idx}", fontsize=8)
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Hierarchical Band Decomposition", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "bands.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Band decomposition saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="best_model.pt")
    parser.add_argument("--label-path", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--output-h", type=int, default=300)
    parser.add_argument("--output-w", type=int, default=530)
    parser.add_argument("--band-ch", type=int, default=32)
    parser.add_argument("--trend-base", type=float, default=100000.0)
    parser.add_argument("--detail-base", type=float, default=100.0)
    parser.add_argument("--out-dir", type=str, default="predicted_frames")
    parser.add_argument("--visualize", action="store_true")
    generate(parser.parse_args())