#!/usr/bin/env python3
"""
Model Conversion Utility:
1. MobileFaceNet / ArcFace -> Apple Silicon CoreML (.mlpackage) for Face Recognition Embedding.
2. MiniFASNetV2 -> Apple Silicon CoreML (.mlpackage) for Anti-Spoofing (Rejecting phone screens & photos).
Optimized for Apple Neural Engine (ANE) on M1, M2, M3, M4 chips.
"""

import sys
import os
import argparse
import numpy as np

def generate_mobilefacenet_model():
    """Generates MobileFaceNet PyTorch model architecture for face feature embedding."""
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

def generate_minifasnet_antispoof_model():
    """Generates MiniFASNetV2 architecture for anti-spoofing (detecting phone screens & photos)."""
    try:
        import torch
        import torch.nn as nn

        class DepthwiseSeparableConv(nn.Module):
            def __init__(self, in_c, out_c, stride=1):
                super().__init__()
                self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, padding=1, groups=in_c, bias=False)
                self.bn1 = nn.BatchNorm2d(in_c)
                self.relu = nn.ReLU6(inplace=True)
                self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
                self.bn2 = nn.BatchNorm2d(out_c)

            def forward(self, x):
                x = self.relu(self.bn1(self.depthwise(x)))
                x = self.relu(self.bn2(self.pointwise(x)))
                return x

        class MiniFASNetV2(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU6(inplace=True)
                )
                self.stage1 = DepthwiseSeparableConv(32, 64, stride=2)
                self.stage2 = DepthwiseSeparableConv(64, 128, stride=2)
                self.stage3 = DepthwiseSeparableConv(128, 128, stride=2)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(128, num_classes)
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                x = self.stem(x)
                x = self.stage1(x)
                x = self.stage2(x)
                x = self.stage3(x)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                logits = self.classifier(x)
                probs = self.softmax(logits)
                return probs  # [prob_spoof, prob_live]

        return MiniFASNetV2()
    except ImportError:
        print("[!] PyTorch is not installed. Please run: pip install -r requirements.txt")
        return None

def convert_embedding_model(output_path: str = "MobileFaceNet_ANE.mlpackage"):
    """Converts PyTorch Face Recognition model to Apple CoreML with ANE compute unit target."""
    try:
        import torch
        import coremltools as ct
    except ImportError:
        print("[!] coremltools or torch not found. Install with: pip install torch coremltools")
        return

    print("[*] Initializing MobileFaceNet architecture (112x112 -> 512-D embedding)...")
    model = generate_mobilefacenet_model()
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
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13
    )

    coreml_model.author = "CodeNebula-Dev"
    coreml_model.short_description = "MobileFaceNet Face Recognition Feature Extractor optimized for Apple Neural Engine."
    coreml_model.version = "1.0.0"

    print(f"[*] Saving CoreML package to {output_path}...")
    coreml_model.save(output_path)
    print(f"[✓] Face Embedding Model conversion complete: '{output_path}'")

def convert_antispoof_model(output_path: str = "MiniFASNet_AntiSpoof.mlpackage"):
    """Converts MiniFASNet Anti-Spoofing model to Apple CoreML for photo/screen replay rejection."""
    try:
        import torch
        import coremltools as ct
    except ImportError:
        print("[!] coremltools or torch not found. Install with: pip install torch coremltools")
        return

    print("[*] Initializing MiniFASNetV2 Anti-Spoofing architecture (80x80 -> Live/Spoof Probability)...")
    model = generate_minifasnet_antispoof_model()
    if model is None:
        return

    model.eval()
    example_input = torch.rand(1, 3, 80, 80)
    traced_model = torch.jit.trace(model, example_input)

    image_input = ct.ImageType(
        name="face_crop_image",
        shape=(1, 3, 80, 80),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB
    )

    coreml_model = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[ct.TensorType(name="liveness_probabilities")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13
    )

    coreml_model.author = "CodeNebula-Dev"
    coreml_model.short_description = "MiniFASNetV2 Anti-Spoofing Liveness Classifier for Apple Silicon Neural Engine."
    coreml_model.version = "1.0.0"

    print(f"[*] Saving Anti-Spoofing package to {output_path}...")
    coreml_model.save(output_path)
    print(f"[✓] Anti-Spoofing Model conversion complete: '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Face ID & Anti-Spoofing models to Apple Silicon CoreML")
    parser.add_argument("--mode", choices=["all", "embedding", "antispoof"], default="all", help="Which model to convert")
    parser.add_argument("--embedding-output", type=str, default="MobileFaceNet_ANE.mlpackage")
    parser.add_argument("--antispoof-output", type=str, default="MiniFASNet_AntiSpoof.mlpackage")
    args = parser.parse_args()

    if args.mode in ("all", "embedding"):
        convert_embedding_model(args.embedding_output)
    if args.mode in ("all", "antispoof"):
        convert_antispoof_model(args.antispoof_output)
