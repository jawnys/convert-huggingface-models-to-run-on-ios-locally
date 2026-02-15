#!/usr/bin/env python3

import os
import torch
import coremltools as ct
from coremltools.models import MLModel
from transformers import ViTForImageClassification, ViTImageProcessor
from typing import cast


def convert_model():
    """
    Converts the Falconsai/nsfw_image_detection model to CoreML model.
    """
    print("🏎️ Start converting model...")

    input_model_dir = "inputs"
    output_model_dir = "outputs"

    os.makedirs(output_model_dir, exist_ok=True)

    # Load HF model + processor
    model = cast(
        ViTForImageClassification,
        ViTForImageClassification.from_pretrained(
            input_model_dir, local_files_only=True
        ),
    )
    model.eval()
    processor = cast(
        ViTImageProcessor,
        ViTImageProcessor.from_pretrained(input_model_dir, local_files_only=True),
    )

    # Simple wrapper: accepts (1,3,224,224) float tensor in [0,255], outputs probs
    class Wrapper(torch.nn.Module):
        def __init__(self, model, processor):
            super().__init__()
            self.model = model
            self.mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
            self.std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
            self.rescale = processor.rescale_factor

        def forward(self, x):
            x = x * self.rescale
            x = (x - self.mean) / self.std
            logits = self.model(pixel_values=x).logits
            return torch.nn.functional.softmax(logits, dim=-1)

    wrapped = Wrapper(model, processor)
    wrapped.eval()

    # Sample input for tracing
    sample = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.float32)

    # Trace and convert
    traced = torch.jit.trace(wrapped, sample)

    input_desc = ct.ImageType(
        name="image",
        shape=(1, 3, 224, 224),
        scale=1.0,
        bias=[0, 0, 0],
        color_layout=ct.colorlayout.RGB,
    )

    coreml_model = ct.convert(
        traced,
        inputs=[input_desc],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram",
    )

    if (coreml_model is None) or (not isinstance(coreml_model, MLModel)):
        raise ValueError("CoreML conversion failed")

    out_path = os.path.join(output_model_dir, "NsfwDetector.mlpackage")
    coreml_model.save(out_path)

    print("🏁 Saved CoreML model to: ", out_path)


if __name__ == "__main__":
    convert_model()
