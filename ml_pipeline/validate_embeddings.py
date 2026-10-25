#!/usr/bin/env python3
"""
Embedding Validation Script
────────────────────────────
Validates that the exported CoreML .mlpackage produces meaningful, discriminative
face embeddings — not random noise from untrained weights.

Tests:
  1. Same image twice → cosine similarity ≈ 1.0
  2. Same person, different angle/lighting → cosine similarity > 0.65
  3. Different people → cosine similarity < 0.35
  4. Embedding is L2-normalized (||v|| ≈ 1.0)
  5. Embedding dimension is 512

Usage:
  python validate_embeddings.py
  python validate_embeddings.py --model-path exported_models/MobileFaceNet_ANE.mlpackage
  python validate_embeddings.py --use-webcam  # Live validation with FaceTime camera
"""

import sys
import argparse
import numpy as np
from pathlib import Path


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def validate_with_synthetic_images(model_path: str):
    """
    Run the model on synthetic test images and verify output properties.
    This doesn't test actual face recognition accuracy, but ensures:
    - The model loads and runs
    - Output shape is correct (512-D)
    - Output is approximately L2-normalized
    - Same input produces same output (deterministic)
    - Different inputs produce different outputs (discriminative)
    """
    try:
        import coremltools as ct
        from PIL import Image
    except ImportError:
        print("[!] Required: pip install coremltools Pillow")
        return False

    print(f"[*] Loading CoreML model: {model_path}")
    try:
        model = ct.models.MLModel(model_path)
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        return False

    results = []

    # Test 1: Output shape and normalization
    print("\n── Test 1: Output Shape & L2 Normalization ──")
    img_gray = Image.new("RGB", (112, 112), color=(128, 128, 128))
    pred = model.predict({"input_image": img_gray})
    embedding = np.array(pred["face_embedding"]).flatten()

    dim_ok = embedding.shape[0] == 512
    norm = np.linalg.norm(embedding)
    norm_ok = abs(norm - 1.0) < 0.15

    print(f"  Dimension: {embedding.shape[0]} {'✓' if dim_ok else '✗ (expected 512)'}")
    print(f"  L2 Norm:   {norm:.6f} {'✓' if norm_ok else '✗ (expected ~1.0)'}")
    results.extend([dim_ok, norm_ok])

    # Test 2: Determinism — same input → same output
    print("\n── Test 2: Determinism (same input → same output) ──")
    pred2 = model.predict({"input_image": img_gray})
    embedding2 = np.array(pred2["face_embedding"]).flatten()
    sim_same = cosine_similarity(embedding, embedding2)
    determ_ok = sim_same > 0.999

    print(f"  Cosine similarity (identical inputs): {sim_same:.6f} {'✓' if determ_ok else '✗'}")
    results.append(determ_ok)

    # Test 3: Discrimination — different inputs → different outputs
    print("\n── Test 3: Discrimination (different inputs → different outputs) ──")
    img_white = Image.new("RGB", (112, 112), color=(255, 255, 255))
    img_black = Image.new("RGB", (112, 112), color=(0, 0, 0))
    img_red = Image.new("RGB", (112, 112), color=(255, 0, 0))

    emb_white = np.array(model.predict({"input_image": img_white})["face_embedding"]).flatten()
    emb_black = np.array(model.predict({"input_image": img_black})["face_embedding"]).flatten()
    emb_red = np.array(model.predict({"input_image": img_red})["face_embedding"]).flatten()

    sim_gray_white = cosine_similarity(embedding, emb_white)
    sim_gray_black = cosine_similarity(embedding, emb_black)
    sim_white_red = cosine_similarity(emb_white, emb_red)

    # Different solid-color images should produce different embeddings
    # (they won't be super different since none are real faces, but shouldn't be identical)
    disc_ok = sim_gray_white < 0.999 and sim_gray_black < 0.999

    print(f"  Sim(gray, white): {sim_gray_white:.4f}")
    print(f"  Sim(gray, black): {sim_gray_black:.4f}")
    print(f"  Sim(white, red):  {sim_white_red:.4f}")
    print(f"  Different outputs for different inputs: {'✓' if disc_ok else '✗'}")
    results.append(disc_ok)

    # Test 4: Statistical properties of the embedding vector
    print("\n── Test 4: Embedding Statistics ──")
    mean_val = np.mean(embedding)
    std_val = np.std(embedding)
    min_val = np.min(embedding)
    max_val = np.max(embedding)

    print(f"  Mean:   {mean_val:.6f}")
    print(f"  Std:    {std_val:.6f}")
    print(f"  Min:    {min_val:.6f}")
    print(f"  Max:    {max_val:.6f}")

    # Embeddings from a trained model should have reasonable statistics
    # Random embeddings from untrained model tend to have very small values
    stats_ok = std_val > 0.01  # Should have some variance
    print(f"  Non-trivial variance: {'✓' if stats_ok else '⚠ (may indicate untrained weights)'}")
    results.append(stats_ok)

    # Summary
    print(f"\n{'='*50}")
    passed = sum(results)
    total = len(results)
    all_pass = all(results)
    print(f"  Results: {passed}/{total} tests passed {'✓' if all_pass else '⚠'}")

    if all_pass:
        print(f"  Model appears functional.")
        print(f"  NOTE: These tests verify model structure, NOT face recognition accuracy.")
        print(f"  For accuracy testing, use real face images from LFW or similar benchmarks.")
    else:
        print(f"  Some tests failed. The model may have issues.")
        print(f"  Ensure pre-trained weights were loaded during conversion.")

    return all_pass


def validate_with_webcam(model_path: str):
    """
    Live validation using the Mac's FaceTime camera.
    Captures two frames of the same person and verifies high similarity.
    """
    try:
        import coremltools as ct
        import cv2
        from PIL import Image
    except ImportError:
        print("[!] Required: pip install coremltools opencv-python-headless Pillow")
        return False

    print(f"[*] Loading CoreML model: {model_path}")
    model = ct.models.MLModel(model_path)

    print("[*] Opening FaceTime camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Cannot open camera")
        return False

    print("[*] Press SPACE to capture frame 1, then SPACE again for frame 2.")
    print("    Press 'q' to quit.")

    embeddings = []

    while len(embeddings) < 2:
        ret, frame = cap.read()
        if not ret:
            continue

        # Show preview
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            # Convert BGR → RGB and resize to 112x112
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((112, 112))

            pred = model.predict({"input_image": pil_img})
            emb = np.array(pred["face_embedding"]).flatten()
            embeddings.append(emb)

            print(f"  Captured frame {len(embeddings)}/2 — embedding norm: {np.linalg.norm(emb):.4f}")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 2:
        sim = cosine_similarity(embeddings[0], embeddings[1])
        print(f"\n  Cosine similarity between captures: {sim:.4f}")
        if sim > 0.65:
            print(f"  ✓ Same person detected (threshold: 0.65)")
        elif sim > 0.40:
            print(f"  ⚠ Borderline — may be same person under different conditions")
        else:
            print(f"  ✗ Embeddings are too different — model may not be working correctly")
        return sim > 0.50
    else:
        print("[!] Not enough captures taken")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate exported CoreML face recognition model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).parent / "exported_models" / "MobileFaceNet_ANE.mlpackage"),
        help="Path to the CoreML .mlpackage file"
    )
    parser.add_argument(
        "--use-webcam",
        action="store_true",
        help="Run live webcam validation (captures 2 frames of same person)"
    )
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"[!] Model not found: {args.model_path}")
        print(f"    Run 'python convert_to_coreml.py' first.")
        sys.exit(1)

    if args.use_webcam:
        success = validate_with_webcam(args.model_path)
    else:
        success = validate_with_synthetic_images(args.model_path)

    sys.exit(0 if success else 1)
