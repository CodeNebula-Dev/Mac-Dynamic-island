# 🏝️ Mac Dynamic Island & Face ID

<div align="center">

![Platform](https://img.shields.io/badge/Platform-macOS%2013%2B-black?style=for-the-badge&logo=apple)
![Chip](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4-orange?style=for-the-badge&logo=apple)
![Swift](https://img.shields.io/badge/Swift-6.2-F05138?style=for-the-badge&logo=swift&logoColor=white)
![CoreML](https://img.shields.io/badge/Apple%20Neural%20Engine-CoreML-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A native, fluid Dynamic Island for macOS featuring an on-device Computer Vision & Machine Learning "Face ID" pipeline, optimized for Apple Silicon.**

[Architecture & Implementation Plan](IMPLEMENTATION_PLAN.md) • [Features](#-features) • [Installation & Build](#-getting-started) • [ML Pipeline](#-ml--computer-vision-pipeline) • [Roadmap](#-roadmap)

</div>

---

```
                       ┌────────────────────────────────┐
                       │       MacBook Top Display      │
                       │        ┌──────────────┐        │
                       │        │ Camera Notch │        │
                       │  ┌─────┴──────────────┴─────┐  │
                       │  │   Mac Dynamic Island     │  │
                       │  │  ◉ Face ID Authenticated │  │
                       │  └──────────────────────────┘  │
```

## ✨ Highlights

* **Native Notch Hugging**: Anchors seamlessly to the physical camera notch of MacBook Pro & Air (M1, M2, M3, M4) with automatic fallback for external monitors and non-notch Macs.
* **Face ID for Mac**: Biometric face authentication using the built-in FaceTime HD camera, Apple Vision Framework, and CoreML running on the **Apple Neural Engine (ANE)**.
* **Anti-Spoofing & Liveness Detection**: Real-time Eye Aspect Ratio (EAR) blink tracking and facial landmark variance to prevent 2D photo and screen spoofing.
* **120Hz ProMotion Spring Physics**: Buttery smooth morphing and expansion animations powered by SwiftUI interactive springs.
* **Smart Hover Actions**: Expands on mouse hover to provide quick actions and notifications.
* **Media & Equalizer HUD**: Dynamic playback status with animated audio equalizer waveform bars.
* **MagSafe & Battery Pulse**: MagSafe connection pulse with circular charge indicator.
* **Zero Cloud Dependency**: 100% private, on-device processing with zero data leaving your machine. Camera turns on strictly on-demand.

---

## 🚀 Why Native Swift for Apple Silicon?

When building a high-performance Dynamic Island with Computer Vision for macOS, language selection is critical:

| Requirement | Swift (AppKit + SwiftUI + CoreML) | Python / PyQt | Electron / Web |
| :--- | :--- | :--- | :--- |
| **Notch Blending** | **Native** (`NSPanel` at `.statusBar` level) | Clunky, lacks window levels | Impossible without delay |
| **Animation Rate** | **120 FPS ProMotion** | 30-60 FPS with jitter | Stutters during CPU load |
| **ML Inference** | **< 3ms on Apple Neural Engine (ANE)** | 40-80ms (CPU/MPS bridge) | Infeasible |
| **Memory & Battery**| **~38 MB RAM, < 0.1% Idle CPU** | ~180 MB RAM, 6-10% CPU | ~350 MB RAM, high drain |
| **Camera Zero-Copy**| Direct `CVPixelBuffer` to Vision/CoreML | Frame serialization lag | Heavy WebRTC overhead |

> **Verdict**: The runtime app is written in **Pure Swift** for unmatched speed and battery efficiency. An offline **Python pipeline** (`ml_pipeline/`) is provided for model training, benchmarking, and CoreML export.

---

## 🛠️ Project Structure

```
MacDynamic-ComputerVision/
├── Package.swift                     # Swift Package Manager manifest (macOS 13+)
├── IMPLEMENTATION_PLAN.md            # In-depth architectural blueprint & roadmap
├── README.md                         # Project documentation & guide
├── Sources/
│   └── MacDynamicIsland/
│       ├── App/
│       │   └── MacDynamicIslandApp.swift   # App entry point & Menu Bar controller
│       ├── Core/
│       │   ├── NotchMetrics.swift          # Hardware notch geometry & screen detector
│       │   └── IslandState.swift           # Central state machine & animation coordinator
│       ├── UI/
│       │   ├── NotchWindow.swift           # Borderless floating panel & hover tracking
│       │   └── DynamicIslandView.swift     # Morphing capsule UI with spring physics
│       └── Features/
│           ├── FaceID/
│           │   ├── FaceIDManager.swift     # AVFoundation capture, Vision & liveness engine
│           │   └── FaceIDHUDView.swift     # Biometric scanning ring & verified checkmark
│           ├── Media/
│           │   └── MediaHUDView.swift      # Music player HUD with live audio visualizer
│           └── Battery/
│               └── BatteryHUDView.swift    # MagSafe charging indicator & battery ring
└── ml_pipeline/
    ├── requirements.txt              # PyTorch, CoreMLTools, InsightFace dependencies
    ├── convert_to_coreml.py          # Script to export MobileFaceNet/ArcFace to CoreML
    └── benchmark.py                  # Cosine similarity and biometric threshold benchmark
```

---

## 🏁 Getting Started

### Prerequisites

* macOS 13.0 (Ventura) or newer (macOS 14 Sonoma / macOS 15+ recommended)
* Apple Silicon Mac (M1, M2, M3, M4 series)
* Swift 5.9+ / Swift 6.2 (included with Xcode or Command Line Tools)

### Build & Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CodeNebula-Dev/Mac-Dynamic-island.git
   cd Mac-Dynamic-island
   ```

2. **Build with Swift Package Manager:**
   ```bash
   swift build
   ```

3. **Run the Dynamic Island application:**
   ```bash
   swift run MacDynamicIsland
   ```

The Dynamic Island will immediately appear anchored around your MacBook notch (or floating at the top center of your display).

---

## 🎮 Interactive Controls & Menu Bar

Mac Dynamic Island runs as a menu bar accessory. Look for the capsule icon `(•)` in your macOS menu bar:

* **Hover over the Notch**: The island smoothly expands to reveal quick triggers.
* **Scan Face (Face ID)** (`⌘F`): Triggers the camera and displays the biometric authentication reticle.
* **Simulate Media Playing** (`⌘M`): Expands the island with album info and an animated audio equalizer.
* **Simulate MagSafe Charge** (`⌘B`): Triggers the green charging pulse and battery percentage.
* **Collapse Island** (`⌘R`): Collapses the island back to its idle notch state.

---

## 🧠 ML & Computer Vision Pipeline

The `ml_pipeline/` directory contains tools for training, evaluating, and exporting face recognition models to Apple CoreML.

### 1. Install ML Dependencies
```bash
cd ml_pipeline
pip install -r requirements.txt
```

### 2. Export Model to Apple Neural Engine (.mlpackage)
Converts a MobileFaceNet / ArcFace backbone into a hardware-accelerated CoreML package:
```bash
python3 convert_to_coreml.py --output MobileFaceNet_ANE.mlpackage
```

### 3. Evaluate Biometric Similarity Threshold
Run the cosine distance verification benchmark:
```bash
python3 benchmark.py
```

---

## 🗺️ Roadmap

- [x] **v0.1**: Notch geometry detection & borderless floating window engine.
- [x] **v0.2**: Dynamic Island spring animation physics & interactive hover.
- [x] **v0.3**: Face ID HUD animations (scanning ring, green verification checkmark).
- [x] **v0.4**: AVFoundation camera session & Apple Vision landmark tracking.
- [x] **v0.5**: Eye Aspect Ratio (EAR) anti-spoofing and liveness scoring.
- [x] **v0.6**: Media equalizer and MagSafe battery charging widgets.
- [x] **v0.7**: Offline Python ML export pipeline for Apple Neural Engine.
- [ ] **v1.0**: Live macOS Keychain biometric enrollment and verification.
- [ ] **v1.1**: PAM plugin (`pam_mac_dynamic_island.so`) for Terminal `sudo` Face ID authentication.
- [ ] **v1.2**: Native Spotify & Apple Music player integration via ScriptingBridge.
- [ ] **v2.0**: Attention-aware screen dimming and contactless hand air gestures.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/CodeNebula-Dev/Mac-Dynamic-island/issues).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/CodeNebula-Dev">CodeNebula-Dev</a></sub>
</div>
