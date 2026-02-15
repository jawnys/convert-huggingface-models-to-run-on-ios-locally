#!/usr/bin/env python3

import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Type
import coremltools as ct
import torch
from coremltools.models import MLModel

CHECKPOINT_PATH = Path("inputs/depth_anything_v2_vits.pth")
OUTPUT_MODEL_PATH = Path("outputs/DepthAnythingV2Small.mlpackage")
INPUT_HEIGHT = 518
INPUT_WIDTH = 518

MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_torchvision_compose_available() -> None:
    """
    Depth-Anything-V2 imports torchvision.transforms.Compose at module load time.
    Some Python builds (e.g. asdf without liblzma) fail importing torchvision.
    For conversion we only need Compose behavior, so install a tiny shim if needed.
    """
    try:
        from torchvision.transforms import Compose  # noqa: F401

        return
    except Exception:
        pass

    class Compose:  # type: ignore[no-redef]
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, value):
            for transform in self.transforms:
                value = transform(value)
            return value

    torchvision_module = types.ModuleType("torchvision")
    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_module.Compose = Compose
    torchvision_module.transforms = transforms_module

    # Replace any partially-loaded torchvision modules with a minimal shim.
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module
    print(
        "Warning: using torchvision Compose shim (full torchvision import unavailable)."
    )


def candidate_depth_anything_repo_paths() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    env_repo = os.environ.get("DEPTH_ANYTHING_V2_REPO", "").strip()

    candidates = [
        script_dir / "inputs/Depth-Anything-V2",
        script_dir / "inputs/Depth-Anything-V2-main",
    ]

    if env_repo:
        candidates.insert(0, Path(env_repo).expanduser())

    return candidates


def import_depth_anything_v2_class() -> Type[torch.nn.Module]:
    """
    Import DepthAnythingV2 either from installed package or local cloned repo.
    """
    ensure_torchvision_compose_available()

    try:
        from depth_anything_v2.dpt import DepthAnythingV2

        return DepthAnythingV2
    except Exception as first_error:
        for repo_dir in candidate_depth_anything_repo_paths():
            if repo_dir.is_dir():
                sys.path.insert(0, str(repo_dir.resolve()))
                try:
                    from depth_anything_v2.dpt import DepthAnythingV2

                    return DepthAnythingV2
                except Exception:
                    continue

        raise ImportError(
            "Could not import depth_anything_v2. Fix with:\n"
            "  cd inputs\n"
            "  git clone https://github.com/DepthAnything/Depth-Anything-V2.git\n"
            "Then re-run ./convert_model.py\n"
            "(Alternatively set DEPTH_ANYTHING_V2_REPO=/absolute/path/to/Depth-Anything-V2)"
        ) from first_error


def normalize_state_dict(raw_checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Handle common checkpoint wrappers and prefixes.
    """
    candidate = raw_checkpoint

    if isinstance(candidate, dict):
        if "state_dict" in candidate and isinstance(candidate["state_dict"], dict):
            candidate = candidate["state_dict"]
        elif "model" in candidate and isinstance(candidate["model"], dict):
            candidate = candidate["model"]

    if not isinstance(candidate, dict):
        raise ValueError("Checkpoint is not a state_dict-like dictionary")

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        normalized_key = key[7:] if key.startswith("module.") else key
        state_dict[normalized_key] = value

    if not state_dict:
        raise ValueError("No tensor weights found in checkpoint")

    return state_dict


class DepthAnythingCoreMLWrapper(torch.nn.Module):
    """
    Wraps DepthAnythingV2 to include image preprocessing inside the model graph.
    Input: float32 RGB tensor [0, 255] with shape (1, 3, 518, 518)
    Output: depth tensor shape (1, 1, 518, 518)
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image / 255.0
        x = (x - self.mean) / self.std

        depth = self.model(x)

        if depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.ndim == 3:
            depth = depth.unsqueeze(1)

        return depth


def load_depth_anything_model() -> torch.nn.Module:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    DepthAnythingV2 = import_depth_anything_v2_class()
    model = DepthAnythingV2(**MODEL_CONFIG)

    raw_checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = normalize_state_dict(raw_checkpoint)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing keys ({len(missing_keys)}):")
        for key in missing_keys[:10]:
            print(f"  - {key}")
    if unexpected_keys:
        print(f"Warning: unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys[:10]:
            print(f"  - {key}")

    model.eval()
    return model


def convert_model() -> None:
    print("Start converting Depth Anything v2 (vits) checkpoint to Core ML...")
    torch.set_grad_enabled(False)

    model = load_depth_anything_model()
    wrapped = DepthAnythingCoreMLWrapper(model)
    wrapped.eval()

    sample = torch.randint(
        0,
        256,
        (1, 3, INPUT_HEIGHT, INPUT_WIDTH),
        dtype=torch.float32,
    )

    traced = torch.jit.trace(wrapped, sample)

    image_input = ct.ImageType(
        name="image",
        shape=sample.shape,
        scale=1.0,
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB,
    )

    coreml_model = ct.convert(
        traced,
        inputs=[image_input],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram",
    )

    if coreml_model is None or not isinstance(coreml_model, MLModel):
        raise ValueError("Core ML conversion failed")

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(str(OUTPUT_MODEL_PATH))

    print(f"Saved Core ML model to: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    convert_model()
