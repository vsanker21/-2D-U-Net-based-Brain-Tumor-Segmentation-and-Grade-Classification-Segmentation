"""Shared utilities for Project 1 methodological upgrade analyses."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from segmentation_model_loader import UNet2D, load_segmentation_model

BASE = Path(r"E:\Brain Tumor Segmentation")
G_BASE = Path(r"G:\Brain Tumor Segmentation")
DATA_DIR = BASE / "archive" / "3D Slices Sorted"
MASK_DIR = DATA_DIR / "masks"
SEG_MASK_DIR = DATA_DIR / "masks_brats2020_seg"  # multi-class {0,1,2,3} from H5 channels
OFFICIAL_MASK_DIR = DATA_DIR / "masks_brats2020_official"  # MICCAI seg aligned {0,1,2,3}
GRADE_CSV = BASE / "archive" / "BraTS2020_training_data" / "content" / "data" / "name_mapping.csv"
SEG_CKPT = BASE / "segmentation_model_t2_best.pth"
OUT_DIR = G_BASE / "project1_methodological_outputs"
NNUNET_RAW = G_BASE / "nnUNet_data" / "nnUNet_raw"
NNUNET_PREPROCESSED = G_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS = G_BASE / "nnUNet_results"

# In-memory mask cache (avoids writing ~9 MB per patient to disk)
_MEM_MASK_CACHE: dict[int, np.ndarray] = {}
_MEM_VOLUME_CACHE: dict[int, np.ndarray] = {}

RANDOM_STATE = 42
TARGET_SLICE = 240
# Classification spatial target (H, W, D) — reduces 3D CNN cost while preserving anatomy
CLASSIFY_SHAPE = (128, 128, 96)


def resize_volume(vol: np.ndarray, shape: tuple[int, int, int] = CLASSIFY_SHAPE) -> np.ndarray:
    if vol.shape == shape:
        return vol
    factors = (shape[0] / vol.shape[0], shape[1] / vol.shape[1], shape[2] / vol.shape[2])
    return zoom(vol, factors, order=1).astype(np.float32)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_grade_mapping() -> dict[str, str]:
    df = pd.read_csv(GRADE_CSV)
    return {row["BraTS_2020_subject_ID"]: row["Grade"] for _, row in df.iterrows()}


def list_patient_ids(grade_mapping: dict[str, str] | None = None) -> list[int]:
    ids: list[int] = []
    for f in sorted(DATA_DIR.glob("*_T2.nii.gz")):
        pid = int(f.name.split("_")[2])
        name = f"BraTS20_Training_{pid:03d}"
        if grade_mapping is None or name in grade_mapping:
            ids.append(pid)
    return sorted(set(ids))


def patient_labels(patient_ids: Sequence[int], grade_mapping: dict[str, str]) -> list[int]:
    return [1 if grade_mapping[f"BraTS20_Training_{pid:03d}"] == "HGG" else 0 for pid in patient_ids]


def master_train_test_split(
    patient_ids: Sequence[int],
    labels: Sequence[int],
    test_size: float = 0.2,
) -> tuple[list[int], list[int], list[int], list[int]]:
    return train_test_split(
        list(patient_ids),
        list(labels),
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=list(labels),
    )


def load_t2_volume(patient_id: int, use_cache: bool = True) -> np.ndarray:
    if use_cache and patient_id in _MEM_VOLUME_CACHE:
        return _MEM_VOLUME_CACHE[patient_id]
    path = DATA_DIR / f"BraTS20_Training_{patient_id:03d}_T2.nii.gz"
    vol = nib.load(path).get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    if use_cache:
        _MEM_VOLUME_CACHE[patient_id] = vol
    return vol


def preload_volumes(patient_ids: Iterable[int]) -> None:
    for pid in patient_ids:
        load_t2_volume(pid, use_cache=True)


def load_gt_mask_volume(patient_id: int, multiclass: bool = True, prefer_official: bool = False) -> np.ndarray:
    if multiclass:
        if prefer_official and OFFICIAL_MASK_DIR.exists():
            path = OFFICIAL_MASK_DIR / f"BraTS20_Training_{patient_id:03d}_seg.nii.gz"
            if path.exists():
                return np.squeeze(nib.load(path).get_fdata()).astype(np.uint8)
        if SEG_MASK_DIR.exists():
            path = SEG_MASK_DIR / f"BraTS20_Training_{patient_id:03d}_seg.nii.gz"
            if path.exists():
                return np.squeeze(nib.load(path).get_fdata()).astype(np.uint8)
    path = MASK_DIR / f"BraTS20_Training_{patient_id:03d}_mask.nii.gz"
    return np.squeeze(nib.load(path).get_fdata()).astype(np.uint8)


def resize_slice_2d(arr: np.ndarray, order: int = 1) -> np.ndarray:
    if arr.shape[0] == TARGET_SLICE and arr.shape[1] == TARGET_SLICE:
        return arr
    zy = TARGET_SLICE / arr.shape[0]
    zx = TARGET_SLICE / arr.shape[1]
    return zoom(arr, (zy, zx), order=order)


@torch.no_grad()
def predict_mask_volume_2d_unet(
    seg_model: nn.Module,
    volume: np.ndarray,
    device: torch.device,
    binary: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """Stack batched 2D U-Net slice predictions into a 3D label volume."""
    seg_model.eval()
    h, w, d = volume.shape
    out = np.zeros((h, w, d), dtype=np.uint8)
    for z0 in range(0, d, batch_size):
        z1 = min(z0 + batch_size, d)
        batch_slices = []
        for z in range(z0, z1):
            sl = resize_slice_2d(volume[:, :, z], order=1)
            sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
            batch_slices.append(sl)
        x = torch.from_numpy(np.stack(batch_slices)).float().unsqueeze(1).to(device)
        logits = seg_model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        for i, z in enumerate(range(z0, z1)):
            pred = preds[i]
            if pred.shape != (h, w):
                zy = h / pred.shape[0]
                zx = w / pred.shape[1]
                pred = zoom(pred, (zy, zx), order=0)
            out[:, :, z] = pred
    if binary:
        return (out > 0).astype(np.float32)
    return out.astype(np.float32)


def get_predicted_mask(patient_id: int, seg_model: nn.Module, device: torch.device) -> np.ndarray:
    if patient_id in _MEM_MASK_CACHE:
        return _MEM_MASK_CACHE[patient_id]
    vol = load_t2_volume(patient_id)
    pred = predict_mask_volume_2d_unet(seg_model, vol, device, binary=True)
    _MEM_MASK_CACHE[patient_id] = pred
    return pred


def cache_predicted_masks(
    patient_ids: Iterable[int],
    seg_model: nn.Module,
    device: torch.device,
    force: bool = False,
) -> None:
    """Warm in-memory predicted mask cache (no disk writes)."""
    if force:
        _MEM_MASK_CACHE.clear()
    ids = list(patient_ids)
    for pid in ids:
        get_predicted_mask(pid, seg_model, device)
        if len(_MEM_MASK_CACHE) % 50 == 0:
            print(f"  masks cached: {len(_MEM_MASK_CACHE)}/{len(ids)}", flush=True)


class TumorGradeClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5, in_channels: int = 1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights: Sequence[float], alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        weights = self.class_weights[target]
        return (weights * focal).mean()


@dataclass
class VolumeSample:
    image: torch.Tensor
    label: int
    patient_id: int


class MaskGuidedClassificationDataset(Dataset):
    def __init__(
        self,
        patient_ids: Sequence[int],
        grade_mapping: dict[str, str],
        seg_model: nn.Module | None = None,
        device: torch.device | None = None,
        mask_source: str = "predicted",
        augment: bool = False,
    ):
        self.patient_ids = list(patient_ids)
        self.grade_mapping = grade_mapping
        self.seg_model = seg_model
        self.device = device
        self.mask_source = mask_source
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        pid = self.patient_ids[idx]
        name = f"BraTS20_Training_{pid:03d}"
        label = 1 if self.grade_mapping[name] == "HGG" else 0
        vol = load_t2_volume(pid)
        if self.mask_source == "predicted":
            if self.seg_model is None or self.device is None:
                raise ValueError("seg_model and device required for predicted masks")
            mask = get_predicted_mask(pid, self.seg_model, self.device).astype(np.float32)
        elif self.mask_source == "ground_truth":
            mask = (load_gt_mask_volume(pid) > 0).astype(np.float32)
        elif self.mask_source == "random":
            rng = np.random.default_rng(RANDOM_STATE + pid * 10007)
            mask = rng.random(vol.shape, dtype=np.float32)
        else:
            raise ValueError(self.mask_source)

        vol = resize_volume(vol)
        mask = resize_volume(mask)

        if self.augment and random.random() < 0.5:
            if random.random() < 0.5:
                vol = np.flip(vol, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            if random.random() < 0.5:
                vol = np.flip(vol, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            scale = random.uniform(0.9, 1.1)
            vol = np.clip(vol * scale, 0.0, 1.0)

        x = torch.from_numpy(vol).unsqueeze(0)
        m = torch.from_numpy(mask).unsqueeze(0)
        return torch.cat([x, m], dim=0), torch.tensor(label, dtype=torch.long), pid


class T2OnlyClassificationDataset(Dataset):
    def __init__(self, patient_ids: Sequence[int], grade_mapping: dict[str, str], augment: bool = False):
        self.patient_ids = list(patient_ids)
        self.grade_mapping = grade_mapping
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        pid = self.patient_ids[idx]
        name = f"BraTS20_Training_{pid:03d}"
        label = 1 if self.grade_mapping[name] == "HGG" else 0
        vol = load_t2_volume(pid)
        vol = resize_volume(vol)
        if self.augment and random.random() < 0.5:
            if random.random() < 0.5:
                vol = np.flip(vol, axis=0).copy()
            if random.random() < 0.5:
                vol = np.flip(vol, axis=1).copy()
            scale = random.uniform(0.9, 1.1)
            vol = np.clip(vol * scale, 0.0, 1.0)
        x = torch.from_numpy(vol).unsqueeze(0)
        return x, torch.tensor(label, dtype=torch.long), pid


def class_weights_from_labels(labels: Sequence[int]) -> list[float]:
    counts = np.bincount(np.asarray(labels, dtype=int))
    total = len(labels)
    return [float(total / (len(counts) * counts[i])) for i in range(len(counts))]


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
