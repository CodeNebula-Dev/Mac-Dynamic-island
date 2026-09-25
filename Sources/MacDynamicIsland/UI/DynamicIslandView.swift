import SwiftUI

@MainActor
public struct DynamicIslandView: View {
    @ObservedObject var state: IslandState

    public init(state: IslandState) {
        self.state = state
    }

    public var body: some View {
        let metrics = NotchMetrics.current()
        let islandWidth = state.currentWidth(for: metrics)
        let islandHeight = state.currentHeight(for: metrics)

        VStack(spacing: 0) {
            ZStack(alignment: .top) {
                // Background Notch Shape matching hardware bezel curvature
                NotchShape(
                    topCornerRadius: state.topCornerRadius(),
                    bottomCornerRadius: state.bottomCornerRadius()
                )
                .fill(Color.black)
                .overlay(
                    NotchShape(
                        topCornerRadius: state.topCornerRadius(),
                        bottomCornerRadius: state.bottomCornerRadius()
                    )
                    .stroke(
                        Color.white.opacity(state.isExpanded ? 0.14 : 0.0),
                        lineWidth: 0.5
                    )
                )
                .shadow(
                    color: Color.black.opacity(state.isExpanded ? 0.65 : 0.0),
                    radius: state.isExpanded ? 18 : 0,
                    x: 0,
                    y: state.isExpanded ? 8 : 0
                )

                // Visible content area placed strictly below the physical notch
                if state.isExpanded {
                    VStack(spacing: 0) {
                        // Top spacer matching physical notch depth
                        Color.clear.frame(height: metrics.notchHeight)

                        // Expanded interactive view
                        Group {
                            switch state.currentMode {
                            case .idle:
                                EmptyView()
                            case .hover:
                                hoverView
                            case .faceID(let status):
                                FaceIDHUDView(status: status)
                            case .battery(let percentage, let isCharging, let isPluggedIn):
                                BatteryHUDView(percentage: percentage, isCharging: isCharging, isPluggedIn: isPluggedIn)
                            case .media(let title, let artist, let isPlaying):
                                MediaHUDView(title: title, artist: artist, isPlaying: isPlaying)
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 6)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                    .transition(
                        .asymmetric(
                            insertion: .opacity.combined(with: .move(edge: .top)),
                            removal: .opacity
                        )
                    )
                }
            }
            .frame(width: islandWidth, height: islandHeight)
            .animation(.interactiveSpring(response: 0.38, dampingFraction: 0.8, blendDuration: 0), value: islandWidth)
            .animation(.interactiveSpring(response: 0.38, dampingFraction: 0.8, blendDuration: 0), value: islandHeight)
            .animation(.interactiveSpring(response: 0.38, dampingFraction: 0.8, blendDuration: 0), value: state.currentMode)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    // MARK: - Hover Quick Actions

    private var hoverView: some View {
        HStack(spacing: 12) {
            Button(action: { state.triggerFaceID() }) {
                HStack(spacing: 6) {
                    Image(systemName: "faceid")
                        .font(.system(size: 15, weight: .semibold))
                    Text("Face ID")
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                }
                .foregroundColor(.cyan)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color.cyan.opacity(0.16))
                .cornerRadius(12)
            }
            .buttonStyle(.plain)

            Button(action: { state.triggerBattery() }) {
                HStack(spacing: 6) {
                    Image(systemName: "battery.100")
                        .font(.system(size: 15, weight: .semibold))
                    Text("Battery")
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                }
                .foregroundColor(.green)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color.green.opacity(0.16))
                .cornerRadius(12)
            }
            .buttonStyle(.plain)

            Button(action: { state.triggerMedia() }) {
                HStack(spacing: 6) {
                    Image(systemName: "music.note")
                        .font(.system(size: 15, weight: .semibold))
                    Text("Media")
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                }
                .foregroundColor(.purple)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color.purple.opacity(0.16))
                .cornerRadius(12)
            }
            .buttonStyle(.plain)
        }
    }
}
