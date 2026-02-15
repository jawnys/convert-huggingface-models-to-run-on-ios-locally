#!/usr/bin/env python3

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from transformers import ViTForImageClassification, ViTImageProcessor
import os

NSFW_THRESHOLD = 0.5
PYTORCH_MODEL = "input-model"
COREML_MODEL = "output-model/NsfwDetector.mlpackage"
TEST_IMAGES_DIR = "../test-images"

def collect_test_images(root_dir: str, extensions=None):
    """
    Recursively finds all images in a given directory.
    """
    if extensions is None:
        extensions = {"jpg", "jpeg", "png", "bmp", "gif", "avif", "heic"}
    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.split('.')[-1].lower() in extensions:
                found.append(os.path.join(dirpath, fn))
    return sorted(found)


def compare_models():
    """
    Compare the PyTorch model outputs with the Core ML model outputs on a set of images.
    """
    print("🏎️ Start comparing models...")

    # Load HF model + processor
    model = ViTForImageClassification.from_pretrained(
        PYTORCH_MODEL, local_files_only=True
    )
    model.eval()
    processor = ViTImageProcessor.from_pretrained(PYTORCH_MODEL, local_files_only=True)

    # Debug: print processor params and labels
    print("processor.rescale_factor:", getattr(processor, "rescale_factor", None))
    print("processor.image_mean:", getattr(processor, "image_mean", None))
    print("processor.image_std:", getattr(processor, "image_std", None))
    print("model.labels(id2label):", getattr(model.config, "id2label", None))

    # Load Core ML model (on mac host)
    mlmodel = ct.models.MLModel(COREML_MODEL)

    def pt_probs(pil: Image.Image) -> np.ndarray:
        inputs = processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            return F.softmax(logits, dim=-1).cpu().numpy()[0]

    def coreml_probs(pil: Image.Image) -> np.ndarray:
        # MLModel.predict may accept PIL.Image for ImageType input named "image".
        try:
            out = mlmodel.predict({"image": pil})
        except Exception:
            arr = np.array(pil).astype(np.float32)
            out = mlmodel.predict({"image": arr})

        # Handle several possible output formats from Core ML:
        # - array-like probabilities (common)
        # - a dict mapping label->score
        # - named outputs where one is the probability vector
        # Prefer array-like outputs first, then dict-of-labels converted using model.config.id2label
        array_like = None
        label_map = None
        for v in out.values():
            if isinstance(v, dict):
                label_map = v
                break
            if hasattr(v, "__len__"):
                # prefer the first array-like
                if array_like is None:
                    array_like = v

        if array_like is not None:
            return np.asarray(array_like).ravel()

        if label_map is not None:
            # Convert label->score dict into ordered vector matching model.config.id2label
            id2label = getattr(model.config, "id2label", None)
            if id2label:
                # Build ordered list by id (0..N-1)
                n = len(id2label)
                ordered = [label_map.get(id2label.get(i), 0.0) for i in range(n)]
                return np.asarray(ordered, dtype=np.float32)
            else:
                # Fallback: deterministic order
                keys = sorted(label_map.keys())
                return np.asarray([label_map[k] for k in keys], dtype=np.float32)

        raise RuntimeError(
            "Couldn't extract numeric outputs from Core ML predict() result"
        )

    # Force resize to 224x224 for the quick test to ensure both models receive
    # the same fixed-size input irrespective of the processor's settings.
    target_size = (224, 224)
    results = []

    for p in collect_test_images(TEST_IMAGES_DIR):
        if not os.path.exists(p):
            print(f"Skipping missing test image: {p}")
            continue
        pil = Image.open(p).convert("RGB")
        # Always resize test images to the target size for consistent comparisons
        pil = pil.resize(target_size)

        p_pt = pt_probs(pil)
        p_core = coreml_probs(pil)

        # Helper: decide which emoji to show based on NSFW probability. We assume
        # class 1 is the NSFW class if the model has two classes; otherwise the
        # highest-index class is used as NSFW by default.
        def prob_with_emoji(probs: np.ndarray) -> str:
            try:
                # Determine NSFW score index: prefer label named 'nsfw' or use index 1
                id2label = getattr(model.config, "id2label", None)
                nsfw_idx = None
                if id2label:
                    for i, lbl in id2label.items():
                        if isinstance(lbl, str) and lbl.lower() == "nsfw":
                            nsfw_idx = int(i)
                            break
                if nsfw_idx is None:
                    # Fallback: if there are 2 classes, use index 1, else use last index
                    nsfw_idx = 1 if probs.shape[0] == 2 else (probs.shape[0] - 1)

                nsfw_score = float(probs[nsfw_idx])
            except Exception:
                nsfw_score = float(probs[-1])

            emoji = "🔞" if nsfw_score >= NSFW_THRESHOLD else "✅"
            return f"{probs} {emoji}"
        

        # normalize shape if necessary
        p_core = p_core[: p_pt.shape[0]]
        d = np.abs(p_pt - p_core)
        print(
            f"{p}: max_abs_diff={d.max():.6f}, pt_top={p_pt.argmax()}, core_top={p_core.argmax()}"
        )
        print("pt_probs:", prob_with_emoji(p_pt), "\ncore_probs:", prob_with_emoji(p_core), "\n")
        results.append(float(d.max()))

    if results:
        print(
            "Summary: max_diff_over_images=",
            max(results),
            ", mean_diff=",
            float(np.mean(results)),
        )

    print("🏁 Finished comparing models")


if __name__ == "__main__":
    compare_models()
