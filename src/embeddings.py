from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


HIST_BINS = 8
HISTOGRAM_EMBEDDING_SIZE = HIST_BINS * HIST_BINS * HIST_BINS
EmbeddingMethod = Literal["histogram", "cnn_resnet50", "swin_tiny", "clip"]
SUPPORTED_METHODS: tuple[EmbeddingMethod, ...] = (
    "histogram",
    "cnn_resnet50",
    "swin_tiny",
    "clip",
)


def get_embedding_size(method: EmbeddingMethod) -> int:
    if method == "histogram":
        return HISTOGRAM_EMBEDDING_SIZE
    if method == "clip":
        return 512
    if method in {"cnn_resnet50", "swin_tiny"}:
        model, _ = _load_timm_model(method)
        feature_dim = getattr(model, "num_features", None)
        if feature_dim is None:
            raise RuntimeError(f"Could not read feature dimension for model: {method}")
        return int(feature_dim)
    raise ValueError(f"Unsupported embedding method: {method}")


def _extract_histogram_embedding(image_path: Path) -> np.ndarray:
    """Create a simple color-histogram embedding for an image."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB").resize((128, 128))
        arr = np.asarray(rgb, dtype=np.float32) / 255.0

    hist, _ = np.histogramdd(
        arr.reshape(-1, 3),
        bins=(HIST_BINS, HIST_BINS, HIST_BINS),
        range=((0, 1), (0, 1), (0, 1)),
    )

    vec = hist.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


@lru_cache(maxsize=2)
def _load_timm_model(method: EmbeddingMethod):
    # Lazy import keeps the baseline path lightweight for users who only need histogram mode.
    import timm
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_name = "resnet50" if method == "cnn_resnet50" else "swin_tiny_patch4_window7_224"
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return model, transform


@lru_cache(maxsize=1)
def _load_clip_model():
    from transformers import CLIPProcessor, CLIPVisionModelWithProjection
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def _extract_clip_embedding(image_path: Path) -> np.ndarray:
    import torch
    model, processor, device = _load_clip_model()

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")

    inputs = processor(images=rgb, return_tensors="pt").to(device)
    autocast_device = "cuda" if device == "cuda" else "cpu"
    with torch.no_grad(), torch.autocast(device_type=autocast_device, enabled=device == "cuda", dtype=torch.float16):
        outputs = model(pixel_values=inputs["pixel_values"])
        vec = outputs.image_embeds

    arr = vec.squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr


def _extract_clip_embeddings_batch(image_paths: list[Path]) -> tuple[np.ndarray, list[Path]]:
    import torch
    model, processor, device = _load_clip_model()

    images = []
    valid_paths = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                images.append(img.convert("RGB"))
            valid_paths.append(image_path)
        except Exception as e:
            print(f"Warning: Skipping corrupted image {image_path}: {e}")

    if not images:
        return np.array([]), []

    inputs = processor(images=images, return_tensors="pt").to(device)
    autocast_device = "cuda" if device == "cuda" else "cpu"
    with torch.no_grad(), torch.autocast(device_type=autocast_device, enabled=device == "cuda", dtype=torch.float16):
        outputs = model(pixel_values=inputs["pixel_values"])
        vecs = outputs.image_embeds

    arrs = vecs.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(arrs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arrs /= norms
    return arrs, valid_paths


def _extract_deep_embedding(image_path: Path, method: EmbeddingMethod) -> np.ndarray:
    import torch

    model, transform = _load_timm_model(method)
    device = next(model.parameters()).device

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")

    input_tensor = transform(rgb).unsqueeze(0).to(device)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad(), torch.autocast(device_type=autocast_device, enabled=device.type == "cuda", dtype=torch.float16):
        vec = model(input_tensor)

    arr = vec.squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr


def _extract_deep_embeddings_batch(image_paths: list[Path], method: EmbeddingMethod) -> tuple[np.ndarray, list[Path]]:
    import torch

    model, transform = _load_timm_model(method)
    device = next(model.parameters()).device

    tensors = []
    valid_paths = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                rgb = img.convert("RGB")
            tensors.append(transform(rgb))
            valid_paths.append(image_path)
        except Exception as e:
            print(f"Warning: Skipping corrupted image {image_path}: {e}")

    if not tensors:
        return np.array([]), []

    input_tensor = torch.stack(tensors).to(device)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad(), torch.autocast(device_type=autocast_device, enabled=device.type == "cuda", dtype=torch.float16):
        vecs = model(input_tensor)

    arrs = vecs.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(arrs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arrs /= norms
    return arrs, valid_paths


def extract_embedding(image_path: Path, method: EmbeddingMethod = "histogram") -> np.ndarray:
    """Extract an embedding for an image using the selected method."""
    if method == "histogram":
        return _extract_histogram_embedding(image_path)
    if method == "clip":
        return _extract_clip_embedding(image_path)
    if method in {"cnn_resnet50", "swin_tiny"}:
        return _extract_deep_embedding(image_path, method)
    raise ValueError(f"Unsupported embedding method: {method}")


def extract_embeddings_batch(image_paths: list[Path], method: EmbeddingMethod = "histogram") -> tuple[np.ndarray, list[Path]]:
    """Extract embeddings for a batch of images using the selected method, skipping corrupted images."""
    if method == "histogram":
        valid_paths = []
        vecs = []
        for p in image_paths:
            try:
                vecs.append(_extract_histogram_embedding(p))
                valid_paths.append(p)
            except Exception as e:
                print(f"Warning: Skipping corrupted image {p}: {e}")
        if not vecs:
            return np.array([]), []
        return np.vstack(vecs), valid_paths
    if method == "clip":
        return _extract_clip_embeddings_batch(image_paths)
    if method in {"cnn_resnet50", "swin_tiny"}:
        return _extract_deep_embeddings_batch(image_paths, method)
    raise ValueError(f"Unsupported embedding method: {method}")
