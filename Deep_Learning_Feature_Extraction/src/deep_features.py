from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import csv
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    models = None
    transforms = None


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ExtractionConfig:
    image_dir: Path
    output_dir: Path
    batch_size: int = 32
    num_workers: int = 2
    model_name: str = "resnet50"
    image_size: int = 224


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int = 224) -> None:
        if transforms is None:
            raise ImportError("torchvision is required for image transforms.")
        self.image_paths = self._collect_images(image_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _collect_images(image_dir: Path) -> List[Path]:
        paths = [
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        paths.sort()
        return paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Any, str]:
        image_path = self.image_paths[index]
        with Image.open(image_path).convert("RGB") as image:
            tensor = self.transform(image)
        return tensor, str(image_path)


def build_backbone(model_name: str):
    if models is None or nn is None:
        raise ImportError("torch and torchvision are required for model loading.")
    if model_name.lower() == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        return backbone

    if model_name.lower() == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        return backbone

    raise ValueError(f"Unsupported model_name: {model_name}")


def select_device():
    if torch is None:
        raise ImportError("torch is required for device selection.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def l2_normalize_rows(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(array, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return array / norms


def extract_deep_features(config: ExtractionConfig) -> Tuple[np.ndarray, Sequence[str]]:
    if torch is None or DataLoader is None:
        raise ImportError("torch and torchvision must be installed to extract deep features.")

    dataset = ImageFolderDataset(config.image_dir, image_size=config.image_size)
    if len(dataset) == 0:
        raise ValueError(f"No supported images found in: {config.image_dir}")

    use_cuda = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
    )

    device = select_device()
    model = build_backbone(config.model_name).to(device)
    model.eval()

    all_features: List[np.ndarray] = []
    all_paths: List[str] = []

    with torch.inference_mode():
        for images, paths in dataloader:
            images = images.to(device, non_blocking=use_cuda)
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
            else:
                outputs = model(images)

            outputs = outputs.float().detach().cpu().numpy()
            outputs = l2_normalize_rows(outputs.astype(np.float32))

            all_features.append(outputs)
            all_paths.extend(paths)

    features = np.vstack(all_features).astype(np.float32)
    return features, all_paths


def save_outputs(output_dir: Path, features: np.ndarray, image_paths: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "features.npy"
    metadata_path = output_dir / "metadata.csv"

    np.save(features_path, features)

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "image_path"])
        for index, image_path in enumerate(image_paths):
            writer.writerow([index, image_path])
