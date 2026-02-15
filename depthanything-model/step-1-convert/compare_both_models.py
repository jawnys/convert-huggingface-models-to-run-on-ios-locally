#!/usr/bin/env python3

from pathlib import Path
from typing import Any, Dict, Tuple
import coremltools as ct
import numpy as np
import torch
from PIL import Image

from convert_model import (
    CHECKPOINT_PATH,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    DepthAnythingCoreMLWrapper,
    import_depth_anything_v2_class,
    normalize_state_dict,
    MODEL_CONFIG,
)

COREML_MODEL_PATH = Path("outputs/DepthAnythingV2Small.mlpackage")
TEST_IMAGE_PATH = Path("../shirt.png")
PT_PREVIEW_OUTPUT = Path("outputs/shirt-depth-pytorch.png")
COREML_PREVIEW_OUTPUT = Path("outputs/shirt-depth-coreml.png")


def load_pytorch_wrapper() -> DepthAnythingCoreMLWrapper:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    DepthAnythingV2 = import_depth_anything_v2_class()
    model = DepthAnythingV2(**MODEL_CONFIG)

    raw_checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = normalize_state_dict(raw_checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    wrapped = DepthAnythingCoreMLWrapper(model)
    wrapped.eval()
    return wrapped


def prepare_input_image(path: Path) -> Tuple[Image.Image, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Test image not found at: {path}")

    pil = Image.open(path).convert("RGB")
    pil = pil.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.BILINEAR)
    array = np.asarray(pil, dtype=np.float32)
    return pil, array


def pytorch_depth_map(
    model: DepthAnythingCoreMLWrapper, image_array: np.ndarray
) -> np.ndarray:
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        depth = model(tensor)

    return depth.squeeze().cpu().numpy().astype(np.float32)


def extract_coreml_depth(prediction: Dict[str, Any]) -> np.ndarray:
    for value in prediction.values():
        try:
            array = np.asarray(value, dtype=np.float32)
        except Exception:
            continue
        if array.size > 0:
            return array.squeeze()

    raise RuntimeError("Could not extract numeric depth output from Core ML prediction")


def coreml_depth_map(mlmodel: ct.models.MLModel, pil_image: Image.Image) -> np.ndarray:
    prediction = mlmodel.predict({"image": pil_image})
    return extract_coreml_depth(prediction).astype(np.float32)


def save_depth_preview(depth: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    min_value = float(depth.min())
    max_value = float(depth.max())
    normalized = (depth - min_value) / (max_value - min_value + 1e-8)

    preview = (normalized * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(output_path)


def print_sample_points(pt_depth: np.ndarray, coreml_depth: np.ndarray) -> None:
    h, w = pt_depth.shape
    points = {
        "center": (h // 2, w // 2),
        "top_center": (0, w // 2),
        "bottom_center": (h - 1, w // 2),
        "left_center": (h // 2, 0),
        "right_center": (h // 2, w - 1),
    }

    print("\nSample depth values (PyTorch vs Core ML):")
    for name, (row, col) in points.items():
        pt_value = float(pt_depth[row, col])
        coreml_value = float(coreml_depth[row, col])
        diff = abs(pt_value - coreml_value)
        print(
            f"  {name:12s} pt={pt_value:10.6f}  coreml={coreml_value:10.6f}  abs_diff={diff:.6f}"
        )


def compare_models() -> None:
    print("Start comparing PyTorch and Core ML depth outputs...")

    pil_image, image_array = prepare_input_image(TEST_IMAGE_PATH)

    pt_model = load_pytorch_wrapper()
    mlmodel = ct.models.MLModel(str(COREML_MODEL_PATH))

    pt_depth = pytorch_depth_map(pt_model, image_array)
    coreml_depth = coreml_depth_map(mlmodel, pil_image)

    if pt_depth.shape != coreml_depth.shape:
        raise ValueError(
            f"Output shape mismatch: PyTorch={pt_depth.shape}, CoreML={coreml_depth.shape}"
        )

    diff = np.abs(pt_depth - coreml_depth)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())

    flat_pt = pt_depth.reshape(-1)
    flat_coreml = coreml_depth.reshape(-1)
    corr = float(np.corrcoef(flat_pt, flat_coreml)[0, 1])

    print(f"Image: {TEST_IMAGE_PATH}")
    print(f"Output shape: {pt_depth.shape}")
    print(f"Max abs diff:  {max_abs_diff:.6f}")
    print(f"Mean abs diff: {mean_abs_diff:.6f}")
    print(f"Correlation:   {corr:.6f}")

    print_sample_points(pt_depth, coreml_depth)

    save_depth_preview(pt_depth, PT_PREVIEW_OUTPUT)
    save_depth_preview(coreml_depth, COREML_PREVIEW_OUTPUT)

    print(f"\nSaved preview: {PT_PREVIEW_OUTPUT}")
    print(f"Saved preview: {COREML_PREVIEW_OUTPUT}")
    print("Finished comparing models.")


if __name__ == "__main__":
    compare_models()
