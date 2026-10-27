import SwiftUI

/// Dynamic Island Multi-Angle Enrollment UI
/// Renders a guided 5-angle capture experience with real-time head-pose telemetry,
/// progress ring, directional indicators, and completion animations.
public struct EnrollmentHUDView: View {
    @ObservedObject var manager = EnrollmentManager.shared
    @State private var pulseScale: CGFloat = 1.0

    public init() {}

    public var body: some View {
        HStack(spacing: 14) {
            // Left: Pose Reticle & Directional Guide
            reticleView

            // Middle: Instruction & Head Pose Telemetry
            instructionView

            Spacer(minLength: 8)

            // Right: 5-Angle Progress Dots & Controls
            progressView
        }
        .padding(.horizontal, 10)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Reticle View

    private var reticleView: some View {
        ZStack {
            // Background track ring
            Circle()
                .stroke(Color.white.opacity(0.12), lineWidth: 3.5)
                .frame(width: 44, height: 44)

            // Dynamic progress ring
            Circle()
                .trim(from: 0.0, to: CGFloat(manager.holdProgress))
                .stroke(
                    manager.isAngleMatched ? Color.green : Color.cyan,
                    style: StrokeStyle(lineWidth: 3.5, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .frame(width: 44, height: 44)
                .animation(.linear(duration: 0.1), value: manager.holdProgress)

            // Center Icon / Directional Arrow
            switch manager.state {
            case .capturing(let angle, _):
                directionalIcon(for: angle)
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(manager.isAngleMatched ? .green : .cyan)
                    .scaleEffect(manager.isAngleMatched ? 1.15 : 1.0)
                    .animation(.easeInOut(duration: 0.2), value: manager.isAngleMatched)

            case .processingCentroid:
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: .cyan))
                    .scaleEffect(0.8)

            case .completed:
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 26, weight: .bold))
                    .foregroundColor(.green)

            case .failed:
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 26, weight: .bold))
                    .foregroundColor(.red)

            default:
                Image(systemName: "faceid")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundColor(.cyan)
            }
        }
    }

    private func directionalIcon(for angle: EnrollmentManager.EnrollmentAngle) -> some View {
        Group {
            switch angle {
            case .frontal:
                Image(systemName: "face.smiling")
            case .left:
                Image(systemName: "arrow.left")
            case .right:
                Image(systemName: "arrow.right")
            case .up:
                Image(systemName: "arrow.up")
            case .down:
                Image(systemName: "arrow.down")
            }
        }
    }

    // MARK: - Instruction & Telemetry

    private var instructionView: some View {
        VStack(alignment: .leading, spacing: 3) {
            switch manager.state {
            case .capturing(let angle, _):
                HStack(spacing: 6) {
                    Text("Angle \(angle.rawValue + 1)/5: \(angle.shortTitle)")
                        .font(.system(size: 14, weight: .bold, design: .rounded))
                        .foregroundColor(.white)

                    if manager.isAngleMatched {
                        Text("HOLD STEADY")
                            .font(.system(size: 10, weight: .heavy, design: .rounded))
                            .foregroundColor(.black)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(Color.green)
                            .cornerRadius(4)
                    }
                }

                Text(angle.guidanceInstruction)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(manager.isAngleMatched ? .green : .white.opacity(0.8))

                // Telemetry line
                HStack(spacing: 8) {
                    Text(String(format: "Yaw: %+.1f°", manager.currentYaw))
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundColor(abs(manager.currentYaw - angle.targetYawDeg) <= angle.yawTolerance ? .green : .gray)

                    Text(String(format: "Pitch: %+.1f°", manager.currentPitch))
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundColor(abs(manager.currentPitch - angle.targetPitchDeg) <= angle.pitchTolerance ? .green : .gray)
                }

            case .processingCentroid:
                Text("Computing Biometric Centroid")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                Text("Generating 512-D L2 template...")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.cyan)

            case .completed(let userName):
                Text("Enrollment Complete!")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundColor(.green)
                Text("Face ID template saved for \(userName)")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white.opacity(0.85))

            case .failed(let reason):
                Text("Enrollment Failed")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundColor(.red)
                Text(reason)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.gray)

            default:
                Text("Enrollment Ready")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
            }
        }
    }

    // MARK: - Progress Dots & Controls

    private var progressView: some View {
        VStack(spacing: 8) {
            // 5 Angle indicators
            HStack(spacing: 5) {
                ForEach(EnrollmentManager.EnrollmentAngle.allCases) { angle in
                    ZStack {
                        if angle.rawValue < manager.currentAngleIndex {
                            // Completed angle
                            Circle()
                                .fill(Color.green)
                                .frame(width: 8, height: 8)
                        } else if angle.rawValue == manager.currentAngleIndex {
                            // Active angle
                            Circle()
                                .fill(manager.isAngleMatched ? Color.green : Color.cyan)
                                .frame(width: 10, height: 10)
                                .overlay(
                                    Circle()
                                        .stroke(Color.white.opacity(0.6), lineWidth: 1.5)
                                )
                        } else {
                            // Future angle
                            Circle()
                                .fill(Color.white.opacity(0.2))
                                .frame(width: 8, height: 8)
                        }
                    }
                }
            }

            // Cancel / Dismiss button
            Button(action: {
                manager.cancelEnrollment()
                Task { @MainActor in
                    IslandState.shared.collapseToIdle()
                }
            }) {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(.white.opacity(0.7))
                    .frame(width: 20, height: 20)
                    .background(Color.white.opacity(0.12))
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
        }
    }
}
