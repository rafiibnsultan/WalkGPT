import json
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IGNORE_INDEX = 255
MASK_EXTENSIONS = (".pt", ".pth", ".png", ".jpg", ".jpeg")


def _maybe_read(path_no_ext: str, exts: Iterable[str] = (".jpg", ".png", ".jpeg")) -> Optional[str]:
    """
    Return the first existing path derived from `path_no_ext` with the provided extensions.
    """
    for ext in exts:
        candidate = f"{path_no_ext}{ext}"
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_mask_path(mask_path_root: str) -> Optional[str]:
    """
    Return the concrete mask file path (if any) for the provided root path.
    """
    for ext in MASK_EXTENSIONS:
        candidate = f"{mask_path_root}{ext}"
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_mask(mask_path_root: str) -> Optional[torch.Tensor]:
    """
    Load a semantic mask stored as PyTorch tensor (.pt/.pth) or raster image (.png/.jpg).
    """
    tensor_candidates = [f"{mask_path_root}{ext}" for ext in MASK_EXTENSIONS if ext in (".pt", ".pth")]
    image_candidates = [f"{mask_path_root}{ext}" for ext in MASK_EXTENSIONS if ext not in (".pt", ".pth")]

    for path in tensor_candidates:
        if not os.path.isfile(path):
            continue
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("mask", "semantic_mask", "label", "labels"):
                if key in payload:
                    payload = payload[key]
                    break
        if isinstance(payload, np.ndarray):
            payload = torch.from_numpy(payload)
        if isinstance(payload, torch.Tensor):
            mask = payload
            break
    else:
        mask = None

    if mask is None:
        for path in image_candidates:
            if not os.path.isfile(path):
                continue
            mask_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                continue
            if mask_img.ndim == 3:
                mask_img = mask_img[..., 0]
            mask = torch.from_numpy(mask_img.astype(np.int32))
            break

    if mask is None:
        return None

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 2:
        raise ValueError(f"Mask tensor must be 2D. Received shape {tuple(mask.shape)}")
    return mask.to(torch.long)


class walkgptImageMaskDataset(Dataset):
    """
    Minimal dataset for semantic segmentation using walkgpt manifests.

    Each sample returns a tuple `(image, mask, metadata)` where:
      - `image`   is a float tensor (3, H, W) normalized to [0, 1]
      - `mask`    is a long tensor (H, W) with semantic class IDs (0..30, IGNORE_INDEX for void)
      - `metadata` contains book-keeping info (image_path, mask_path, session, index)

    Parameters
    ----------
    jsonl_path : str | Path
        Path to JSON lines manifest describing samples.
    resize_to : Optional[int]
        Resize images and masks to a square of this size if provided.
    normalize : bool
        Apply dataset-specific mean/std normalization after converting to [0, 1].
    drop_missing_masks : bool
        Skip samples without a valid mask. Otherwise returns an IGNORE_INDEX mask.
    drop_ignore_only : bool
        Skip samples whose mask (after optional resize) is entirely IGNORE_INDEX.
    transforms : Optional[Callable]
        Callable applied to the image tensor AFTER normalization.
    target_transforms : Optional[Callable]
        Callable applied to the mask tensor.
    """

    pixel_mean = torch.tensor([97.17, 105.73, 108.16]) / 255.0
    pixel_std = torch.tensor([53.05, 56.40, 61.93]) / 255.0

    def __init__(
        self,
        jsonl_path: str | Path,
        resize_to: Optional[int] = None,
        normalize: bool = True,
        drop_missing_masks: bool = True,
        drop_ignore_only: bool = False,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"Could not find manifest at {self.jsonl_path}")

        self.resize_to = resize_to
        self.normalize = normalize
        self.drop_missing_masks = drop_missing_masks
        self.drop_ignore_only = drop_ignore_only
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.ignore_index = ignore_index

        self.samples = self._read_manifest()
        self.indices = self._filter_indices()

    def _read_manifest(self) -> List[dict]:
        samples: List[dict] = []
        with self.jsonl_path.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not samples:
            raise RuntimeError(f"No samples found in manifest {self.jsonl_path}")
        return samples

    def _filter_indices(self) -> List[int]:
        indices: List[int] = []
        for idx, sample in enumerate(self.samples):
            session = sample.get("session", "").rstrip("/")
            index = sample.get("index", "")
            mask_root = os.path.join(session, "masks", index)

            mask = _load_mask(mask_root)
            if mask is None:
                if self.drop_missing_masks:
                    continue
                mask = torch.full((1, 1), self.ignore_index, dtype=torch.long)

            if self.resize_to is not None and mask is not None:
                mask = (
                    F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(self.resize_to, self.resize_to),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .to(torch.long)
                )

            if self.drop_ignore_only and mask is not None:
                if torch.all(mask == self.ignore_index):
                    continue

            indices.append(idx)

        if not indices:
            raise RuntimeError("No valid samples found after applying dataset filters.")
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        base_idx = self.indices[item]
        sample = self.samples[base_idx]

        session = sample.get("session", "").rstrip("/")
        index = sample.get("index", "")

        vf_root = os.path.join(session, "video_frames", index)
        img_path = _maybe_read(vf_root)
        if img_path is None:
            raise FileNotFoundError(f"Could not locate image for sample {session}/{index}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Failed to load image {img_path}")
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_root = os.path.join(session, "masks", index)
        mask_path = resolve_mask_path(mask_root)
        mask = _load_mask(mask_root)
        if mask is None:
            if self.drop_missing_masks:
                raise RuntimeError(f"Mask missing for sample {session}/{index}")
            mask = torch.full(
                (image_rgb.shape[0], image_rgb.shape[1]),
                self.ignore_index,
                dtype=torch.long,
            )
            original_mask_shape = mask.shape
        else:
            original_mask_shape = mask.shape

        image = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous().float() / 255.0

        if self.resize_to is not None:
            target_size = (self.resize_to, self.resize_to)
            image = F.interpolate(
                image.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask = (
                F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=target_size,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .to(torch.long)
            )

        if self.normalize:
            image = (image - self.pixel_mean.view(-1, 1, 1)) / self.pixel_std.view(-1, 1, 1)

        if self.transforms is not None:
            image = self.transforms(image)
        if self.target_transforms is not None:
            mask = self.target_transforms(mask)

        metadata = {
            "image_path": img_path,
            "mask_root": mask_root,
            "mask_path": mask_path or "",
            "session": session,
            "index": index,
            "mask_height": int(original_mask_shape[0]),
            "mask_width": int(original_mask_shape[1]),
        }

        return image, mask, metadata
