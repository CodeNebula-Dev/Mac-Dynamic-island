import SwiftUI

@MainActor
public struct DynamicIslandView: View {
    @ObservedObject var state: IslandState
    @State private var edgeGlowPhase: Double = 0.0

    public init(state: IslandState) {
        self.state = state
    }

    // Fine-tuned spring matching NotchNook/Bloom physics
    private var morphSpring: Animation {
        .spring(response: 0.28, dampingFraction: 0.78, blendDuration: 0.15)
    }

    public var body: some View {
        let metrics = NotchMetrics.current()
        let islandWidth = state.currentWidth(for: metrics)
        let islandHeight = state.currentHeight(for: metrics)

        VStack(spacing: 0) {
            ZStack(alignment: .top) {
                // Layer 0: Ambient edge glow (visible on hover/expand or cursor proximity — hitodama-inspired)
                let showGlow = state.isExpanded || state.cursorProximity > 0.05
                if showGlow {
                    RoundedRectangle(cornerRadius: state.bottomCornerRadius() + 2, style: .continuous)
                        .fill(
                            .linearGradient(
                                colors: edgeGlowColors,
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .blur(radius: 12)
                        .opacity(state.isExpanded ? 0.35 : Double(state.cursorProximity) * 0.22)
                        .scaleEffect(state.isExpanded ? 1.03 : 1.0 + CGFloat(state.cursorProximity) * 0.015)
                        .animation(.easeInOut(duration: 2.5).repeatForever(autoreverses: true), value: edgeGlowPhase)
                }

                // Layer 1: Background shape with glassmorphism
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
                        LinearGradient(
                            colors: state.isExpanded
                                ? [Color.white.opacity(0.22), Color.white.opacity(0.06)]
                                : [Color.clear],
                            startPoint: .top,
                            endPoint: .bottom
                        ),
                        lineWidth: 0.75
                    )
                )
                .shadow(
                    color: Color.black.opacity(state.isExpanded ? 0.55 : 0.0),
                    radius: state.isExpanded ? 22 : 0,
                    x: 0,
                    y: state.isExpanded ? 10 : 0
                )

                // Layer 2: Content area below physical notch
                if state.isExpanded {
                    VStack(spacing: 0) {
                        Color.clear.frame(height: metrics.notchHeight)

                        Group {
                            switch state.currentMode {
                            case .idle:
                                EmptyView()
                            case .hover:
                                hoverView
                            case .faceID(let status):
                                FaceIDHUDView(status: status)
                            case .enrollment:
                                EnrollmentHUDView()
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
                            insertion: .opacity
                                .combined(with: .scale(scale: 0.92, anchor: .top))
                                .combined(with: .offset(y: -4)),
                            removal: .opacity
                                .combined(with: .scale(scale: 0.96, anchor: .top))
                        )
                    )
                }
            }
            .frame(width: islandWidth, height: islandHeight)
            .animation(morphSpring, value: islandWidth)
            .animation(morphSpring, value: islandHeight)
            .animation(morphSpring, value: state.currentMode)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .onAppear {
            edgeGlowPhase = 1.0
        }
    }

    // MARK: - Edge Glow Colors (context-aware)

    private var edgeGlowColors: [Color] {
        switch state.currentMode {
        case .faceID(.success):
            return [Color.green.opacity(0.6), Color.mint.opacity(0.3)]
        case .faceID(.failed):
            return [Color.red.opacity(0.5), Color.orange.opacity(0.3)]
        case .faceID(.notEnrolled):
            return [Color.orange.opacity(0.4), Color.yellow.opacity(0.2)]
        case .enrollment:
            return [Color.cyan.opacity(0.5), Color.blue.opacity(0.3)]
        case .battery:
            return [Color.green.opacity(0.4), Color.teal.opacity(0.25)]
        case .media:
            return [Color.purple.opacity(0.5), Color.pink.opacity(0.3)]
        default:
            return [Color.cyan.opacity(0.3), Color.blue.opacity(0.2)]
        }
    }

    // MARK: - Hover Quick Actions (Bloom-inspired pill buttons with glow)

    private var hoverView: some View {
        HStack(spacing: 10) {
            hoverButton(
                icon: "faceid",
                label: "Face ID",
                color: .cyan,
                action: { state.triggerFaceID() }
            )

            hoverButton(
                icon: "battery.100",
                label: "Battery",
                color: .green,
                action: { state.triggerBattery() }
            )

            hoverButton(
                icon: "music.note",
                label: "Media",
                color: .purple,
                action: { state.triggerMedia() }
            )
        }
    }

    private func hoverButton(icon: String, label: String, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 14, weight: .semibold))
                Text(label)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
            }
            .foregroundColor(color)
            .padding(.horizontal, 14)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 11, style: .continuous)
                    .fill(color.opacity(0.14))
                    .overlay(
                        RoundedRectangle(cornerRadius: 11, style: .continuous)
                            .stroke(color.opacity(0.18), lineWidth: 0.5)
                    )
            )
            .contentShape(RoundedRectangle(cornerRadius: 11, style: .continuous))
        }
        .buttonStyle(IslandButtonStyle())
    }
}

// MARK: - Premium Button Style with Scale Effect

struct IslandButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.92 : 1.0)
            .opacity(configuration.isPressed ? 0.8 : 1.0)
            .animation(.spring(response: 0.2, dampingFraction: 0.7), value: configuration.isPressed)
    }
}
