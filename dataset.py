import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageSeriesDataset(Dataset):
    """Yields (label, timestep_index, frame) tuples — one per frame."""

    def __init__(self, sample_dir, max_frames=None):
        self.sample_dir = sample_dir
        self.label = np.load(os.path.join(sample_dir, "label.npy")).astype(np.float32)

        pattern = os.path.join(sample_dir, "Test image series*.npy")
        files = glob.glob(pattern)

        def _sort_key(path):
            m = re.search(r"_(\d+)\.npy$", path)
            return int(m.group(1)) if m else 0

        self.image_files = sorted(files, key=_sort_key)
        if max_frames is not None:
            self.image_files = self.image_files[:max_frames]

        self.num_frames = len(self.image_files)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx]).astype(np.float32)
        label = torch.from_numpy(self.label)
        timestep = torch.tensor(idx, dtype=torch.float32)
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        return label, timestep, image
