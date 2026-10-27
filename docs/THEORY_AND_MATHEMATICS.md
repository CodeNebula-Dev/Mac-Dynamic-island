# Mac Dynamic Island — Face ID: Mathematical Foundations, Theory & System Dynamics

> **Target Platform:** Apple Silicon MacBooks (M1/M2/M3/M4)  
> **Camera Modality:** Monocular 1080p/720p FaceTime HD Camera (Zero Infrared/Depth Hardware)  
> **Inference Engine:** Apple Neural Engine (ANE) via CoreML FP16 (Unified Memory Architecture)  
> **UI Dynamics:** Inspired by **Hitodama**, **Bloom**, and **NotchNook**

---

## 1. Problem Formulation & Physical Hardware Realities

Standard MacBooks do **not** possess a TrueDepth infrared camera or VCSEL dot projector. Consequently, biometric security must rely entirely on:
1. Canonical **2D facial alignment** invariant to scale, translation, and in-plane rotation.
2. Deep **metric representation learning** on a unit hypersphere $\mathbb{S}^{511}$ using ArcFace margin loss.
3. Multi-angle **5-pose centroid synthesis** to construct a noise-resistant enrolled template.
4. Multi-tier **Presentation Attack Detection (PAD)** combining Central Difference Convolutions (CDC), Fourier frequency analysis, and Eye Aspect Ratio (EAR) blink temporal dynamics.
5. High-fidelity **damped spring physics** with cursor proximity fields for fluid hover responsiveness.

---

## 2. Umeyama 2D Affine Similarity Transform

To eliminate facial pose variability before neural embedding, 5 detected landmarks $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^5$ are mapped to canonical coordinates $\mathbf{Y} = \{\mathbf{y}_i\}_{i=1}^5$ via a similarity transform:

$$\min_{c, \mathbf{R}, \mathbf{t}} \frac{1}{N} \sum_{i=1}^N \|\mathbf{y}_i - (c \mathbf{R} \mathbf{x}_i + \mathbf{t})\|_2^2$$

### Centroid & Covariance Matrix
$$\bm{\mu}_x = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i, \quad \bm{\mu}_y = \frac{1}{N} \sum_{i=1}^N \mathbf{y}_i$$
$$\sigma_x^2 = \frac{1}{N} \sum_{i=1}^N \|\mathbf{x}_i - \bm{\mu}_x\|_2^2$$
$$\bm{\Sigma}_{xy} = \frac{1}{N} \sum_{i=1}^N (\mathbf{y}_i - \bm{\mu}_y)(\mathbf{x}_i - \bm{\mu}_x)^\top$$

### SVD Decomposition
$$\bm{\Sigma}_{xy} = \mathbf{U} \mathbf{D} \mathbf{V}^\top, \quad \mathbf{D} = \operatorname{diag}(d_1, d_2)$$

### Reflection Prevention ($\det(\mathbf{R}) = +1$)
$$\mathbf{S} = \begin{cases} \mathbf{I}_2 & \text{if } \det(\mathbf{U}) \det(\mathbf{V}) = 1 \\ \operatorname{diag}(1, -1) & \text{if } \det(\mathbf{U}) \det(\mathbf{V}) = -1 \end{cases}$$

### Optimal Parameters
$$\mathbf{R}^* = \mathbf{U} \mathbf{S} \mathbf{V}^\top$$
$$c^* = \frac{1}{\sigma_x^2} \operatorname{tr}(\mathbf{D} \mathbf{S})$$
$$\mathbf{t}^* = \bm{\mu}_y - c^* \mathbf{R}^* \bm{\mu}_x$$

### Canonical $112 \times 112$ Landmarks (InsightFace WebFace600K Standard)
- **Left Eye Center:** `(38.2946, 51.6963)`
- **Right Eye Center:** `(73.5318, 51.5014)`
- **Nose Tip:** `(56.0252, 71.7366)`
- **Left Mouth Corner:** `(41.5493, 92.3655)`
- **Right Mouth Corner:** `(70.7299, 92.2041)`

---

## 3. Metric Embedding & ArcFace Loss

The aligned $112 \times 112$ crop is fed to MobileFaceNet, outputting 512-dimensional vector $\mathbf{z}$:

$$\mathbf{u} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2} \in \mathbb{S}^{511}$$

### ArcFace Loss Function (CVPR 2019)
$$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos \theta_j}}$$
- $s = 64.0$ (hypersphere radius)
- $m = 0.50$ radians ($28.6^\circ$ additive angular margin)

### Matching Formula (Cosine Similarity)
$$\operatorname{Sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v} = \sum_{k=1}^{512} u_k v_k = \cos \theta_{\mathbf{u}\mathbf{v}}$$

- **Authorized:** $\operatorname{Sim}(\mathbf{u}, \mathbf{v}) \ge 0.68$ ($\theta \le 47.16^\circ$)
- **Rejected:** $\operatorname{Sim}(\mathbf{u}, \mathbf{v}) < 0.68$

---

## 4. 5-Angle Biometric Centroid Synthesis

Enrollment captures 5 canonical angles:
1. **Frontal:** $\text{yaw} = 0^\circ, \text{pitch} = 0^\circ$
2. **Left:** $\text{yaw} = +15^\circ$ (Apple Vision convention: positive = user turns left)
3. **Right:** $\text{yaw} = -15^\circ$
4. **Up:** $\text{pitch} = +10^\circ$
5. **Down:** $\text{pitch} = -10^\circ$

### Centroid Formulation
$$\mathbf{c} = \frac{1}{5} \sum_{k=1}^5 \mathbf{f}_k, \quad \mathbf{e} = \frac{\mathbf{c}}{\|\mathbf{c}\|_2}$$

The template $\mathbf{e}$ is serialized and saved to the **macOS Keychain** protected by `kSecAttrAccessibleWhenUnlockedThisDeviceOnly`.

---

## 5. Anti-Spoofing & Liveness Detection

### Central Difference Convolution (CDC, CVPR 2020)
$$y(p_0) = \theta \sum_{p_n \in \mathcal{R}} w(p_n)(x(p_0 + p_n) - x(p_0)) + (1 - \theta)\sum_{p_n \in \mathcal{R}} w(p_n) x(p_0 + p_n)$$
Extracts screen pixel grids, paper halftones, and moiré patterns from single $80 \times 80$ face crops.

### Eye Aspect Ratio (EAR) Blink Detection
$$\operatorname{EAR} = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2 \|p_1 - p_4\|_2}$$
Requires $\Delta \operatorname{EAR} \ge 0.08$ drop and rebound over $100\text{--}350\text{ ms}$ to defeat static photo attacks.

---

## 6. Dynamic Island Physics: Hitodama & Bloom Mechanics

The hover expansion follows a second-order damped harmonic oscillator:

$$m \ddot{x}(t) + c \dot{x}(t) + k (x(t) - x_{\text{target}}) = 0$$

### SwiftUI Parameter Conversion
$$\omega_n = \frac{2\pi}{T_0}, \quad k = m \left(\frac{2\pi}{T_0}\right)^2, \quad c = \frac{4\pi \zeta m}{T_0}$$

- **Response $T_0$:** $0.28\text{ s}$
- **Damping fraction $\zeta$:** $0.78$
- **Debounce:** $120\text{ ms}$

### Cursor Proximity Potential Field
$$d(\mathbf{p}, \mathcal{B}) = \sqrt{\max(x_{\min} - x_p, 0, x_p - x_{\max})^2 + \max(y_{\min} - y_p, 0, y_p - y_{\max})^2}$$
$$P(d) = \max\left(0, \ 1.0 - \frac{d(\mathbf{p}, \mathcal{B})}{60.0}\right)$$

When $P(d) > 0.05$, the notch illuminates with an ambient edge glow before mouse entry.
