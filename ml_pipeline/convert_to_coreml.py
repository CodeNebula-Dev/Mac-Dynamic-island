#!/usr/bin/env python3
"""
Model Conversion Utility: MobileFaceNet / ArcFace -> Apple Silicon CoreML (.mlpackage)
Optimized for Apple Neural Engine (ANE) on M1, M2, M3, M4 chips.
"""

import sys
import os
import argparse
import numpy as np

def generate_mock_torch_model():
    """Generates a MobileFaceNet PyTorch model architecture for face feature embedding."""
    try:
        import torch
        import torch.nn as nn

        class ConvBlock(nn.Module):
            def __init__(self, in_c, out_c, kernel=(3, 3), stride=1, padding=1, groups=1):
                super().__init__()
                self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
                self.bn = nn.BatchNorm2d(out_c)
                self.prelu = nn.PReLU()

            def forward(self, x):
                return self.prelu(self.bn(self.conv(x)))

        class MobileFaceNetBackbone(nn.Module):
            def __init__(self, embedding_size=512):
                super().__init__()
                self.features = nn.Sequential(
                    ConvBlock(3, 64, kernel=3, stride=2, padding=1),
                    ConvBlock(64, 64, kernel=3, stride=1, padding=1, groups=64),
                    ConvBlock(64, 128, kernel=3, stride=2, padding=1),
                    ConvBlock(128, 128, kernel=3, stride=1, padding=1, groups=128),
                    ConvBlock(128, 256, kernel=3, stride=2, padding=1),
                    ConvBlock(256, 256, kernel=3, stride=1, padding=1, groups=256),
                    ConvBlock(256, 512, kernel=3, stride=2, padding=1),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.linear = nn.Linear(512, embedding_size, bias=False)
                self.bn = nn.BatchNorm1d(embedding_size)

            def forward(self, x):
                feat = self.features(x)
                feat = torch.flatten(feat, 1)
                feat = self.bn(self.linear(feat))
                # L2 normalize embeddings for cosine similarity
                norm = torch.norm(feat, p=2, dim=1, keepdim=True)
                return feat / (norm + 1e-10)

        return MobileFaceNetBackbone()
    except ImportError:
        print("[!] PyTorch is not installed. Please run: pip install -r requirements.txt")
        return None

def convert_to_coreml(output_path: str = "MobileFaceNet_ANE.mlpackage"):
    """Converts PyTorch Face Recognition model to Apple CoreML with ANE compute unit target."""
    try:
        import torch
        import coremltools as ct
    except ImportError:
        print("[!] coremltools or torch not found. Install with: pip install torch coremltools")
        return

    print("[*] Initializing MobileFaceNet architecture...")
    model = generate_mock_torch_model()
    if model is None:
        return

    model.eval()
    example_input = torch.rand(1, 3, 112, 112)
    traced_model = torch.jit.trace(model, example_input)

    print("[*] Converting to CoreML (.mlpackage) with Image input configuration...")
    image_input = ct.ImageType(
        name="input_image",
        shape=(1, 3, 112, 112),
        scale=1.0 / 127.5,
        bias=[-1.0, -1.0, -1.0],
        color_layout=ct.colorlayout.RGB
    )

    coreml_model = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[ct.TensorType(name="face_embedding")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL, # Leverages Apple Neural Engine + GPU
        minimum_deployment_target=ct.target.macOS13
    )

    # Attach model metadata for macOS inspection
    coreml_model.author = "CodeNebula-Dev"
    coreml_model.short_description = "MobileFaceNet Face Recognition Feature Extractor optimized for Apple Neural Engine."
    coreml_model.version = "1.0.0"

    print(f"[*] Saving CoreML package to {output_path}...")
    coreml_model.save(output_path)
    print(f"[✓] Conversion complete! Deploy '{output_path}' directly into MacDynamicIsland Swift app.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Face Recognition model to Apple Silicon CoreML")
    parser.add_argument("--output", type=str, default="MobileFaceNet_ANE.mlpackage", help="Output .mlpackage filename")
    args = parser.parse_args()
    convert_to_coreml(args.output)
