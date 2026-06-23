# Hierarchical Multi-Rate FiLM Decoder

Generates a time series of deformation frames conditioned on a label vector using a **Hierarchical Multi-Rate FiLM Decoder** — a novel architecture that decomposes temporal generation into three frequency bands, each with its own FiLM-conditioned convolutional decoder.

## Architecture

```
                           ┌─ Static Band ──→ BandDecoder ──→ base field  ─┐
Label (4,) ──→ Label MLPs ─┤                                               │
                           ├─ Trend Band  ──→ BandDecoder ──→ slow trend  ─┼──→ σ(sum) ──→ Frame
Timestep t ──→ Sinusoidal ─┤  (low-freq)                                   │    (1, 300, 530)
               Encodings   └─ Detail Band ──→ BandDecoder ──→ fast detail ─┘
                              (high-freq)
```

### Temporal Band Decomposition

| Band | Time Encoding | Captures |
|---|---|---|
| **Static** | None (label only) | Time-invariant base deformation field |
| **Trend** | Low-freq sinusoidal (`base=100000`) | Slow, global loading evolution |
| **Detail** | High-freq sinusoidal (`base=100`) | Fast, localized transients |

- Each band has its own label MLP, time MLP, and `BandDecoder` (FiLM-conditioned ConvTranspose2d stack)
- Band outputs are summed in logit space, then passed through a shared sigmoid
- `return_bands=True` exposes individual band outputs for interpretability

## Data Format

Each sample directory contains:
- `label.npy` — shape `(4,)`, conditioning parameters
- `Test image series_001.npy` … `Test image series_500.npy` — shape `(300, 530)` each

## Usage

### Train
```bash
python3 train.py --epochs 200 --batch-size 16 --lr 1e-3
```

### Inference
```bash
python3 inference.py \
    --label-path ../auto_regressive_transformer/data/sample0/label.npy \
    --num-frames 500 \
    --visualize
```

### Score
```bash
python3 score/score.py --pred-dir predicted_frames
```

Compares predicted frames against ground-truth frame-by-frame and reports:

| Metric | Meaning | Good Range |
|---|---|---|
| **MSE** | Mean squared pixel error | < 0.001 |
| **MAE** | Mean absolute pixel error | < 0.01 |
| **PSNR** | Peak signal-to-noise ratio (dB) | > 30 dB |

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 200 | Training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--max-frames` | 500 | Number of frames to use from the dataset |
| `--output-h` | 300 | Frame height |
| `--output-w` | 530 | Frame width |
| `--band-ch` | 32 | Base channels per band decoder |
| `--trend-base` | 100000 | Sinusoidal base for low-freq trend band |
| `--detail-base` | 100 | Sinusoidal base for high-freq detail band |
| `--ckpt` | `best_model.pt` | Checkpoint path |
| `--visualize` | off | Save grid, filmstrip, and band decomposition PNGs |

## Files

| File | Purpose |
|---|---|
| `dataset.py` | `ImageSeriesDataset` — yields `(label, timestep, frame)` tuples |
| `model.py` | `HierarchicalTemporalDecoder`, `BandDecoder`, `FiLMBlock` |
| `train.py` | Training loop with AdamW, cosine LR, grad clipping, val split |
| `inference.py` | Generate frames + band decomposition visualization |
| `temporal_coherene_loss.py` | `TemporalCoherenceLoss` for sequential smoothness penalties |
