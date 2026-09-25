#!/usr/bin/env python3
"""
Face Recognition Cosine Similarity & Threshold Evaluation Script
"""

import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two 512-d normalized face vectors."""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))

def evaluate_threshold(sim: float, threshold: float = 0.65) -> bool:
    """Standard Face ID threshold test."""
    return sim >= threshold

if __name__ == "__main__":
    print("[*] Running simulated biometric face embedding distance benchmark...")
    np.random.seed(42)

    # 1. Simulate enrolled user embedding (512-dim vector from Apple Neural Engine)
    user_enrolled = np.random.randn(512)
    user_enrolled /= np.linalg.norm(user_enrolled)

    # 2. Simulate live camera capture of the SAME user (slight perturbation in lighting/angle)
    perturbation = np.random.randn(512)
    perturbation /= np.linalg.norm(perturbation)
    user_live_match = user_enrolled + 0.35 * perturbation
    user_live_match /= np.linalg.norm(user_live_match)

    # 3. Simulate live camera capture of an IMPOSTOR (uncorrelated face vector)
    impostor = np.random.randn(512)
    impostor /= np.linalg.norm(impostor)

    sim_genuine = cosine_similarity(user_enrolled, user_live_match)
    sim_impostor = cosine_similarity(user_enrolled, impostor)

    print(f"[*] Genuine Match Similarity: {sim_genuine:.4f} (Threshold: >= 0.65) -> Auth: {'ACCEPTED' if evaluate_threshold(sim_genuine) else 'REJECTED'}")
    print(f"[*] Impostor Match Similarity: {sim_impostor:.4f} (Threshold: >= 0.65) -> Auth: {'ACCEPTED' if evaluate_threshold(sim_impostor) else 'REJECTED'}")
    print("[✓] Threshold benchmark successfully demonstrated.")
