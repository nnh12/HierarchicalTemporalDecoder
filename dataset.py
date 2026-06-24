import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageSeriesDataset(Dataset):
    """Yields (label, timestep_index, frame) tuples — one per frame.

    Accepts a list of sample directories. Each sample dir must contain
    label.npy and a set of numbered frame .npy files.
    """

    def __init__(self, sample_dirs, max_frames_per_sample=None):
        self.labels = []
        self.image_files = []
        self.timesteps = []

        def _sort_key(path):
            m = re.search(r"_(\d+)\.npy$", path)
            return int(m.group(1)) if m else 0

        for sample_dir in sample_dirs:
            label = np.load(os.path.join(sample_dir, "label.npy")).astype(np.float32)
            all_npy = glob.glob(os.path.join(sample_dir, "*.npy"))
            frames = sorted(
                [f for f in all_npy if os.path.basename(f) != "label.npy"],
                key=_sort_key,
            )
            if max_frames_per_sample is not None:
                frames = frames[:max_frames_per_sample]

            for local_t, fpath in enumerate(frames):
                self.labels.append(label)
                self.image_files.append(fpath)
                self.timesteps.append(local_t)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx]).astype(np.float32)
        label = torch.from_numpy(self.labels[idx])
        timestep = torch.tensor(self.timesteps[idx], dtype=torch.float32)
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        return label, timestep, image
