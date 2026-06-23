import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import ImageSeriesDataset
from model import HierarchicalTemporalDecoder

DATA_DIR = os.path.expanduser(
    "~/new/mount/strain_vit_project/src/auto_regressive_transformer/data/samples/sample0"
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ImageSeriesDataset(DATA_DIR, max_frames=args.max_frames)
    print(f"Dataset size: {len(dataset)} frames")

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = HierarchicalTemporalDecoder(
        label_dim=4, output_h=args.output_h, output_w=args.output_w,
        band_ch=args.band_ch, trend_base=args.trend_base,
        detail_base=args.detail_base,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for labels, timesteps, images in train_loader:
            labels = labels.to(device)
            timesteps = timesteps.to(device)
            images = images.to(device)

            pred = model(labels, timesteps)
            loss = criterion(pred, images)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for labels, timesteps, images in val_loader:
                labels = labels.to(device)
                timesteps = timesteps.to(device)
                images = images.to(device)
                pred = model(labels, timesteps)
                val_loss += criterion(pred, images).item() * labels.size(0)
        val_loss /= val_size
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.ckpt)

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {args.ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--output-h", type=int, default=300)
    parser.add_argument("--output-w", type=int, default=530)
    parser.add_argument("--band-ch", type=int, default=32)
    parser.add_argument("--trend-base", type=float, default=100000.0)
    parser.add_argument("--detail-base", type=float, default=100.0)
    parser.add_argument("--ckpt", type=str, default="best_model.pt")
    train(parser.parse_args())
