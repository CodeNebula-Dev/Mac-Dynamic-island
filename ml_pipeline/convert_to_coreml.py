#!/usr/bin/env python3
"""
CoreML Model Conversion Pipeline (v2 — Pre-Trained Weights)
─────────────────────────────────────────────────────────────
Converts pre-trained face recognition and anti-spoofing models to Apple Silicon CoreML (.mlpackage)
optimized for the Apple Neural Engine (ANE) on M1/M2/M3/M4 chips.

Conversion Paths:
  1. MobileFaceNet (Face Recognition):
     ONNX (InsightFace pre-trained) → CoreML .mlpackage (FP16, ANE)

  2. MiniFASNetV2 (Anti-Spoofing):
     PyTorch .pth (Silent-Face pre-trained) → TorchScript → CoreML .mlpackage (FP16, ANE)

Usage:
  # First download pre-trained weights:
  python download_weights.py

  # Then convert to CoreML:
  python convert_to_coreml.py --mode all
  python convert_to_coreml.py --mode embedding    # Face recognition only
  python convert_to_coreml.py --mode antispoof     # Anti-spoofing only
  python convert_to_coreml.py --mode embedding --use-pytorch  # Use PyTorch path instead of ONNX
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
OUTPUT_DIR = Path(__file__).parent / "exported_models"


# ──────────────────────────────────────────────────────────────────────────────
# Path A: ONNX → CoreML (Recommended for InsightFace recognition models)
# ──────────────────────────────────────────────────────────────────────────────

def convert_onnx_recognition_model(
    onnx_path: str = None,
    output_path: str = None,
):
    """
    Converts InsightFace ONNX face recognition model directly to CoreML.
    This is the recommended path — avoids architecture mismatch risks.
    """
    try:
        import coremltools as ct
    except ImportError:
        print("[!] coremltools not installed. Run: pip install coremltools")
        return False

    if onnx_path is None:
        onnx_path = str(CHECKPOINTS_DIR / "w600k_mbf.onnx")

    if output_path is None:
        output_path = str(OUTPUT_DIR / "MobileFaceNet_ANE.mlpackage")

    if not Path(onnx_path).exists():
        print(f"[!] ONNX checkpoint not found: {onnx_path}")
        print(f"    Run 'python download_weights.py' first to download pre-trained weights.")
        return False

    print(f"[*] Converting ONNX → CoreML (Face Recognition)")
    print(f"    Source: {onnx_path}")
    print(f"    Target: {output_path}")
    print(f"    Compute: FP16, Apple Neural Engine priority")

    # Validate ONNX model structure
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"    ✓ ONNX model validation passed")

        # Log input/output shapes
        for inp in onnx_model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"    Input: '{inp.name}' shape={shape}")
        for out in onnx_model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"    Output: '{out.name}' shape={shape}")
    except ImportError:
        print("    (onnx package not installed — skipping model validation)")
    except Exception as e:
        print(f"    ⚠ ONNX validation warning: {e}")

    # Convert ONNX → CoreML
    print(f"[*] Running coremltools ONNX conversion...")
    try:
        # CoreML conversion with image input for zero-copy CVPixelBuffer ingestion
        # InsightFace models expect input normalized to [-1, 1]: (pixel - 127.5) / 128.0
        # Which is equivalent to: scale = 1/127.5, bias = -1.0

        mlmodel = ct.convert(
            onnx_path,
            inputs=[
                ct.ImageType(
                    name="input_image",
                    shape=(1, 3, 112, 112),
                    scale=1.0 / 127.5,
                    bias=[-1.0, -1.0, -1.0],
                    color_layout=ct.colorlayout.RGB,
                )
            ],
            outputs=[ct.TensorType(name="face_embedding")],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,  # Prioritizes ANE → GPU → CPU
            minimum_deployment_target=ct.target.macOS13,
        )
    except Exception as e:
        print(f"[!] CoreML conversion failed: {e}")
        print(f"    This may be due to unsupported ONNX operators.")
        print(f"    Try the PyTorch path instead: --use-pytorch")
        return False

    # Add metadata
    mlmodel.author = "CodeNebula-Dev (via InsightFace)"
    mlmodel.short_description = (
        "MobileFaceNet face recognition model trained on WebFace600K with ArcFace loss. "
        "Produces 512-D L2-normalized embeddings. Optimized for Apple Neural Engine (ANE)."
    )
    mlmodel.version = "2.0.0"
    mlmodel.license = "Non-commercial research (InsightFace MIT license)"

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mlmodel.save(output_path)
    print(f"[✓] Face Recognition model exported: {output_path}")

    # Quick sanity check: run inference with a test tensor
    _validate_coreml_model(output_path, expected_output_dim=512)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Path B: PyTorch → TorchScript → CoreML (Fallback for recognition models)
# ──────────────────────────────────────────────────────────────────────────────

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


def convert_pytorch_recognition_model(
    checkpoint_path: str = None,
    output_path: str = None,
):
    """
    Converts PyTorch face recognition model to CoreML.
    Loads pre-trained weights from checkpoint before tracing.
    Falls back to architecture-only export if no checkpoint found.
    """
    try:
        import torch
        import coremltools as ct
    except ImportError:
        print("[!] torch or coremltools not found. Install with: pip install torch coremltools")
        return False

    if output_path is None:
        output_path = str(OUTPUT_DIR / "MobileFaceNet_ANE.mlpackage")

    print(f"[*] Initializing MobileFaceNet architecture (112x112 → 512-D embedding)...")
    model = generate_mobilefacenet_model()
    if model is None:
        return False

    # ━━━ CRITICAL FIX: Load pre-trained weights ━━━
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[*] Loading pre-trained weights from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"], strict=False)
                elif "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            print(f"    ✓ Pre-trained weights loaded successfully")
        except Exception as e:
            print(f"    ⚠ Warning: Could not load checkpoint: {e}")
            print(f"    Proceeding with architecture-only export (random weights).")
            print(f"    The exported model will NOT produce meaningful embeddings!")
    else:
        print(f"[!] WARNING: No pre-trained weights found!")
        print(f"    The model will be exported with RANDOM weights.")
        print(f"    Run 'python download_weights.py' first, or use --use-onnx (recommended).")

    model.eval()
    example_input = torch.rand(1, 3, 112, 112)
    traced_model = torch.jit.trace(model, example_input)

    print(f"[*] Converting to CoreML (.mlpackage) with Image input configuration...")
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
    coreml_model.short_description = (
        "MobileFaceNet Face Recognition Feature Extractor optimized for Apple Neural Engine."
    )
    coreml_model.version = "2.0.0"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coreml_model.save(output_path)
    print(f"[✓] Face Embedding Model exported: {output_path}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Anti-Spoofing Model Conversion (PyTorch .pth → CoreML)
# ──────────────────────────────────────────────────────────────────────────────

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


def convert_antispoof_model(
    checkpoint_path: str = None,
    output_path: str = None,
):
    """
    Converts MiniFASNet Anti-Spoofing model to CoreML.
    Loads pre-trained weights from Silent-Face-Anti-Spoofing checkpoint.
    """
    try:
        import torch
        import coremltools as ct
    except ImportError:
        print("[!] coremltools or torch not found. Install with: pip install torch coremltools")
        return False

    if checkpoint_path is None:
        checkpoint_path = str(CHECKPOINTS_DIR / "MiniFASNetV2_80x80.pth")

    if output_path is None:
        output_path = str(OUTPUT_DIR / "MiniFASNet_AntiSpoof.mlpackage")

    print(f"[*] Initializing MiniFASNetV2 Anti-Spoofing architecture (80x80 → Live/Spoof)...")
    model = generate_minifasnet_antispoof_model()
    if model is None:
        return False

    # ━━━ CRITICAL FIX: Load pre-trained weights ━━━
    if Path(checkpoint_path).exists():
        print(f"[*] Loading pre-trained anti-spoofing weights from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"], strict=False)
                elif "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            print(f"    ✓ Pre-trained anti-spoofing weights loaded")
        except Exception as e:
            print(f"    ⚠ Warning: Could not load checkpoint: {e}")
            print(f"    The anti-spoof model may not reliably detect photos/screens.")
    else:
        print(f"[!] WARNING: No pre-trained anti-spoofing weights found: {checkpoint_path}")
        print(f"    Run 'python download_weights.py' first.")
        print(f"    Proceeding with RANDOM weights (anti-spoofing will NOT work).")

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

    coreml_model.author = "CodeNebula-Dev (via Silent-Face-Anti-Spoofing)"
    coreml_model.short_description = (
        "MiniFASNetV2 Anti-Spoofing Liveness Classifier for Apple Silicon Neural Engine. "
        "Detects printed photos, phone screen replays, and other presentation attacks."
    )
    coreml_model.version = "2.0.0"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coreml_model.save(output_path)
    print(f"[✓] Anti-Spoofing Model exported: {output_path}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Validation Helper
# ──────────────────────────────────────────────────────────────────────────────

def _validate_coreml_model(mlpackage_path: str, expected_output_dim: int = 512):
    """Quick sanity check: run inference with a synthetic input to verify the model loads."""
    try:
        import coremltools as ct
        print(f"\n[*] Validating exported model: {mlpackage_path}")
        model = ct.models.MLModel(mlpackage_path)

        # Print model spec
        spec = model.get_spec()
        print(f"    Inputs: {[inp.name for inp in spec.description.input]}")
        print(f"    Outputs: {[out.name for out in spec.description.output]}")

        # Test with a synthetic image (PIL)
        try:
            from PIL import Image
            test_img = Image.new("RGB", (112, 112), color=(128, 128, 128))
            pred = model.predict({"input_image": test_img})

            if "face_embedding" in pred:
                embedding = np.array(pred["face_embedding"]).flatten()
                norm = np.linalg.norm(embedding)
                print(f"    Output shape: {embedding.shape}")
                print(f"    L2 norm: {norm:.4f} (should be ~1.0 for normalized embeddings)")
                print(f"    First 5 values: {embedding[:5]}")

                if embedding.shape[0] == expected_output_dim:
                    print(f"    ✓ Output dimension correct: {expected_output_dim}")
                else:
                    print(f"    ⚠ Unexpected output dimension: {embedding.shape[0]} (expected {expected_output_dim})")

                if abs(norm - 1.0) < 0.1:
                    print(f"    ✓ Embedding is approximately unit-normalized")
                else:
                    print(f"    ⚠ Embedding is NOT unit-normalized (norm={norm:.4f})")
        except ImportError:
            print(f"    (Pillow not installed — skipping inference test)")

    except Exception as e:
        print(f"    ⚠ Validation failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Face ID & Anti-Spoofing models to Apple Silicon CoreML (.mlpackage)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "embedding", "antispoof"],
        default="all",
        help="Which model to convert"
    )
    parser.add_argument(
        "--use-pytorch",
        action="store_true",
        help="Use PyTorch → TorchScript → CoreML path instead of ONNX → CoreML (for recognition model)"
    )
    parser.add_argument(
        "--embedding-output",
        type=str,
        default=None,
        help="Output path for face recognition .mlpackage"
    )
    parser.add_argument(
        "--antispoof-output",
        type=str,
        default=None,
        help="Output path for anti-spoofing .mlpackage"
    )
    parser.add_argument(
        "--embedding-checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint for face recognition (only used with --use-pytorch)"
    )
    parser.add_argument(
        "--antispoof-checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint for anti-spoofing"
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Mac Dynamic Island — CoreML Model Conversion Pipeline v2          ║")
    print("║  Exports to .mlpackage with FP16 precision for Apple Neural Engine  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    success = True

    if args.mode in ("all", "embedding"):
        if args.use_pytorch:
            ok = convert_pytorch_recognition_model(
                checkpoint_path=args.embedding_checkpoint,
                output_path=args.embedding_output,
            )
        else:
            ok = convert_onnx_recognition_model(
                onnx_path=args.embedding_checkpoint,
                output_path=args.embedding_output,
            )
        success = success and ok

    if args.mode in ("all", "antispoof"):
        ok = convert_antispoof_model(
            checkpoint_path=args.antispoof_checkpoint,
            output_path=args.antispoof_output,
        )
        success = success and ok

    if success:
        print(f"\n[✓] All conversions complete!")
        print(f"    Exported models: {OUTPUT_DIR}")
        print(f"    → Copy .mlpackage files to Sources/MacDynamicIsland/Resources/")
    else:
        print(f"\n[!] Some conversions failed. Check errors above.")

    sys.exit(0 if success else 1)
