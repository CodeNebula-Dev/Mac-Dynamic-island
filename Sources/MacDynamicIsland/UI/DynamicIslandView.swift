import SwiftUI

@MainActor
public struct DynamicIslandView: View {
    @ObservedObject var state: IslandState
    let metrics: NotchMetrics

    public init(state: IslandState, metrics: NotchMetrics = .current()) {
        self.state = state
        self.metrics = metrics
    }

    public var body: some View {
        let targetWidth = state.targetWidth(for: metrics)
        let targetHeight = state.targetHeight(for: metrics)

        ZStack {
            // Background Capsule with deep OLED black & subtle border
            RoundedRectangle(cornerRadius: state.cornerRadius, style: .continuous)
                .fill(Color.black)
                .overlay(
                    RoundedRectangle(cornerRadius: state.cornerRadius, style: .continuous)
                        .stroke(
                            LinearGradient(
                                colors: [
                                    Color.white.opacity(state.isHovered ? 0.25 : 0.10),
                                    Color.white.opacity(0.04)
                                ],
                                startPoint: .top,
                                endPoint: .bottom
                            ),
                            lineWidth: 1
                        )
                )
                .shadow(
                    color: Color.black.opacity(state.currentMode == .idle ? 0.3 : 0.6),
                    radius: state.currentMode == .idle ? 6 : 14,
                    x: 0,
                    y: state.currentMode == .idle ? 2 : 6
                )

            // Content Container
            Group {
                switch state.currentMode {
                case .idle:
                    idleView
                case .hover:
                    hoverQuickActionsView
                case .welcome:
                    welcomeView
                case .faceID(let status):
                    FaceIDHUDView(status: status)
                case .media(let title, let artist, let isPlaying):
                    MediaHUDView(title: title, artist: artist, isPlaying: isPlaying)
                case .battery(let percentage, let isCharging):
                    BatteryHUDView(percentage: percentage, isCharging: isCharging)
                }
            }
            .transition(.opacity.combined(with: .scale(scale: 0.95)))
        }
        .frame(width: targetWidth, height: targetHeight)
        .animation(.spring(response: 0.38, dampingFraction: 0.72, blendDuration: 0), value: state.currentMode)
        .animation(.spring(response: 0.38, dampingFraction: 0.72, blendDuration: 0), value: state.isHovered)
        .onHover { isHovered in
            state.setHovered(isHovered)
        }
    }

    // MARK: - Subviews

    private var welcomeView: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.green.opacity(0.2))
                    .frame(width: 32, height: 32)
                Circle()
                    .fill(Color.green)
                    .frame(width: 10, height: 10)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("Dynamic Island Active")
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                Text("Hover notch or click menu bar")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.gray)
            }

            Spacer()

            HStack(spacing: 4) {
                Image(systemName: "faceid")
                    .font(.system(size: 11, weight: .semibold))
                Text("READY")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
            }
            .foregroundColor(.cyan)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(Color.cyan.opacity(0.15))
            .cornerRadius(6)
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var idleView: some View {
        HStack(spacing: 8) {
            // Subtle idle camera dot indicator or minimal status
            if !metrics.hasNotch {
                Circle()
                    .fill(Color.green.opacity(0.7))
                    .frame(width: 5, height: 5)
                Text("Island")
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .foregroundColor(.white.opacity(0.6))
            }
        }
    }

    private var hoverQuickActionsView: some View {
        HStack(spacing: 12) {
            // Face ID Quick Trigger
            Button(action: {
                state.triggerFaceIDDemo()
            }) {
                HStack(spacing: 4) {
                    Image(systemName: "faceid")
                        .font(.system(size: 12, weight: .semibold))
                    Text("Face ID")
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                }
                .foregroundColor(.cyan)
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(Color.cyan.opacity(0.15))
                .cornerRadius(8)
            }
            .buttonStyle(.plain)

            // Media Quick Trigger
            Button(action: {
                state.triggerMediaDemo()
            }) {
                HStack(spacing: 4) {
                    Image(systemName: "music.note")
                        .font(.system(size: 12, weight: .semibold))
                    Text("Media")
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                }
                .foregroundColor(.purple)
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(Color.purple.opacity(0.15))
                .cornerRadius(8)
            }
            .buttonStyle(.plain)

            // Battery Quick Trigger
            Button(action: {
                state.triggerBatteryPulse()
            }) {
                HStack(spacing: 4) {
                    Image(systemName: "bolt.fill")
                        .font(.system(size: 12, weight: .semibold))
                    Text("Power")
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                }
                .foregroundColor(.green)
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(Color.green.opacity(0.15))
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 14)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
