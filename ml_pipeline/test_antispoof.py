#!/usr/bin/env python3
"""
Anti-Spoofing Model Validation Script
──────────────────────────────────────
Validates that the exported MiniFASNet CoreML model can distinguish:
  - Live human faces (prob_live > 0.82)
  - Phone/tablet screen replays (prob_live < 0.40)
  - Printed photos (prob_live < 0.40)

Usage:
  python test_antispoof.py
  python test_antispoof.py --model-path exported_models/MiniFASNet_AntiSpoof.mlpackage
  python test_antispoof.py --use-webcam  # Live testing with FaceTime camera
"""

import sys
import argparse
import numpy as np
from pathlib import Path


def test_with_synthetic_inputs(model_path: str):
    """
    Test the anti-spoofing model with synthetic inputs.
    Since we don't have real spoof samples, this verifies:
    - Model loads and runs correctly
    - Output shape is [1, 2] (prob_spoof, prob_live)
    - Probabilities sum to ~1.0
    - Model is deterministic
    """
    try:
        import coremltools as ct
        from PIL import Image
    except ImportError:
        print("[!] Required: pip install coremltools Pillow")
        return False

    print(f"[*] Loading Anti-Spoofing CoreML model: {model_path}")
    try:
        model = ct.models.MLModel(model_path)
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        return False

    results = []

    # Test 1: Output shape and probability distribution
    print("\n── Test 1: Output Shape & Probability Sum ──")
    img = Image.new("RGB", (80, 80), color=(180, 140, 120))  # Skin-ish color
    pred = model.predict({"face_crop_image": img})
    probs = np.array(pred["liveness_probabilities"]).flatten()

    shape_ok = probs.shape[0] == 2
    sum_ok = abs(np.sum(probs) - 1.0) < 0.01

    print(f"  Output shape: {probs.shape} {'✓' if shape_ok else '✗ (expected 2)'}")
    print(f"  Probabilities: [spoof={probs[0]:.4f}, live={probs[1]:.4f}]")
    print(f"  Sum: {np.sum(probs):.4f} {'✓' if sum_ok else '✗ (expected ~1.0)'}")
    results.extend([shape_ok, sum_ok])

    # Test 2: Determinism
    print("\n── Test 2: Determinism ──")
    pred2 = model.predict({"face_crop_image": img})
    probs2 = np.array(pred2["liveness_probabilities"]).flatten()
    determ_ok = np.allclose(probs, probs2, atol=1e-4)
    print(f"  Same input → same output: {'✓' if determ_ok else '✗'}")
    results.append(determ_ok)

    # Test 3: Different inputs produce different outputs
    print("\n── Test 3: Input Sensitivity ──")
    test_images = {
        "skin_tone": Image.new("RGB", (80, 80), color=(180, 140, 120)),
        "white_paper": Image.new("RGB", (80, 80), color=(245, 245, 240)),
        "phone_blue": Image.new("RGB", (80, 80), color=(40, 60, 200)),
        "dark": Image.new("RGB", (80, 80), color=(20, 20, 20)),
    }

    all_probs = {}
    for name, img in test_images.items():
        pred = model.predict({"face_crop_image": img})
        p = np.array(pred["liveness_probabilities"]).flatten()
        all_probs[name] = p
        print(f"  {name:15s} → spoof={p[0]:.4f}, live={p[1]:.4f}")

    # Check that at least some inputs produce different outputs
    unique_outputs = len(set(tuple(np.round(p, 3)) for p in all_probs.values()))
    sensitivity_ok = unique_outputs > 1
    print(f"  Unique output distributions: {unique_outputs}/4 {'✓' if sensitivity_ok else '⚠'}")
    results.append(sensitivity_ok)

    # Test 4: Inference speed
    print("\n── Test 4: Inference Speed ──")
    import time
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict({"face_crop_image": img})
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times[2:])  # Skip first 2 (warmup)
    p95_ms = np.percentile(times[2:], 95)
    speed_ok = avg_ms < 50  # Should be <5ms on ANE, <50ms on CPU

    print(f"  Average: {avg_ms:.2f} ms {'✓' if speed_ok else '⚠ (slow — may not be running on ANE)'}")
    print(f"  P95:     {p95_ms:.2f} ms")
    results.append(speed_ok)

    # Summary
    print(f"\n{'='*50}")
    passed = sum(results)
    total = len(results)
    print(f"  Results: {passed}/{total} tests passed")

    if all(results):
        print(f"  ✓ Anti-spoofing model is functional.")
        print(f"  NOTE: Accuracy testing requires real spoof samples.")
        print(f"  Use CelebA-Spoof or CASIA-SURF datasets for proper benchmarking.")
    else:
        print(f"  ⚠ Some tests failed. Check model weights.")

    return all(results)


def test_with_webcam(model_path: str):
    """
    Live test: capture frames from webcam and show real-time liveness scores.
    Useful for testing with real faces and phone screen presentations.
    """
    try:
        import coremltools as ct
        import cv2
        from PIL import Image
    except ImportError:
        print("[!] Required: pip install coremltools opencv-python-headless Pillow")
        return False

    print(f"[*] Loading Anti-Spoofing model: {model_path}")
    model = ct.models.MLModel(model_path)

    print("[*] Opening FaceTime camera for live anti-spoof testing...")
    print("    Show your face normally → should see 'LIVE' score > 0.82")
    print("    Show a photo on phone  → should see 'SPOOF' detected")
    print("    Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Cannot open camera")
        return False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Crop center square and resize to 80x80
            h, w = frame.shape[:2]
            size = min(h, w)
            y_off = (h - size) // 2
            x_off = (w - size) // 2
            crop = frame[y_off:y_off+size, x_off:x_off+size]

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((80, 80))

            pred = model.predict({"face_crop_image": pil_img})
            probs = np.array(pred["liveness_probabilities"]).flatten()

            prob_spoof = probs[0]
            prob_live = probs[1]

            # Draw results on frame
            if prob_live >= 0.82:
                label = f"LIVE ({prob_live:.2f})"
                color = (0, 255, 0)
            elif prob_live >= 0.50:
                label = f"UNCERTAIN ({prob_live:.2f})"
                color = (0, 255, 255)
            else:
                label = f"SPOOF ({prob_spoof:.2f})"
                color = (0, 0, 255)

            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.rectangle(frame, (x_off, y_off), (x_off+size, y_off+size), color, 2)
            cv2.imshow("Anti-Spoof Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate anti-spoofing CoreML model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).parent / "exported_models" / "MiniFASNet_AntiSpoof.mlpackage"),
        help="Path to the anti-spoofing CoreML .mlpackage"
    )
    parser.add_argument(
        "--use-webcam",
        action="store_true",
        help="Run live webcam anti-spoofing test"
    )
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"[!] Model not found: {args.model_path}")
        print(f"    Run 'python convert_to_coreml.py --mode antispoof' first.")
        sys.exit(1)

    if args.use_webcam:
        success = test_with_webcam(args.model_path)
    else:
        success = test_with_synthetic_inputs(args.model_path)

    sys.exit(0 if success else 1)
