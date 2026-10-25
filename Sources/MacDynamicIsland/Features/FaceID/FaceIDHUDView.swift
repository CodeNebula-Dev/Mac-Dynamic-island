import SwiftUI

public struct FaceIDHUDView: View {
    public let status: FaceIDStatus
    @State private var rotation: Double = 0.0

    public init(status: FaceIDStatus) {
        self.status = status
    }

    public var body: some View {
        HStack(spacing: 14) {
            // Icon
            iconView

            // Status label
            statusLabels

            Spacer(minLength: 4)

            // Contextual Action Button
            actionButton
        }
        .padding(.horizontal, 8)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Icon

    private var iconView: some View {
        ZStack {
            switch status {
            case .ready, .scanning:
                Circle()
                    .stroke(
                        AngularGradient(
                            gradient: Gradient(colors: [.cyan, .blue, .purple, .cyan]),
                            center: .center
                        ),
                        lineWidth: 3.0
                    )
                    .frame(width: 36, height: 36)
                    .rotationEffect(.degrees(rotation))
                    .onAppear {
                        withAnimation(
                            .linear(duration: 1.5).repeatForever(autoreverses: false)
                        ) {
                            rotation = 360
                        }
                    }

                Image(systemName: "faceid")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundColor(.cyan)

            case .notEnrolled:
                Image(systemName: "faceid")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundColor(.orange)

            case .success:
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundColor(.green)
                    .transition(.scale.combined(with: .opacity))

            case .failed(let reason):
                let isSpoof = reason.contains("Screen") || reason.contains("Photo") || reason.contains("micro-motion")
                Image(systemName: isSpoof ? "exclamationmark.triangle.fill" : "xmark.circle.fill")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundColor(isSpoof ? .orange : .red)
                    .transition(.scale.combined(with: .opacity))
            }
        }
    }

    // MARK: - Labels

    private var statusLabels: some View {
        VStack(alignment: .leading, spacing: 3) {
            switch status {
            case .ready:
                Text("Face ID")
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)
                Text("Looking for enrolled face...")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.gray)

            case .scanning:
                Text("Scanning Face...")
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)
                Text("Evaluating liveness & 512-D embedding")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.cyan.opacity(0.9))

            case .notEnrolled:
                Text("Face ID Not Enrolled")
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundColor(.orange)
                Text("5-angle capture required for biometric match")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.gray)

            case .success(let name, let similarity):
                HStack(spacing: 6) {
                    Text("Verified")
                        .font(.system(size: 15, weight: .bold, design: .rounded))
                        .foregroundColor(.green)

                    Text(String(format: "%.0f%% Match", similarity * 100))
                        .font(.system(size: 11, weight: .heavy, design: .rounded))
                        .foregroundColor(.black)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.green)
                        .cornerRadius(6)
                }
                Text("Welcome back, \(name)")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white.opacity(0.9))

            case .failed(let reason):
                Text(reason.contains("Detected") ? "Spoof Detected" : "Face ID Failed")
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundColor(reason.contains("Detected") ? .orange : .red)
                Text(reason)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.gray)
                    .lineLimit(1)
            }
        }
    }

    // MARK: - Action Button

    @ViewBuilder
    private var actionButton: some View {
        switch status {
        case .notEnrolled:
            Button(action: {
                IslandState.shared.triggerEnrollment()
            }) {
                HStack(spacing: 5) {
                    Text("Enroll")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .bold))
                }
                .foregroundColor(.black)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.orange)
                .cornerRadius(12)
            }
            .buttonStyle(.plain)

        case .failed:
            Button(action: {
                FaceIDManager.shared.startAuthentication()
            }) {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 11, weight: .bold))
                    Text("Retry")
                        .font(.system(size: 12, weight: .bold, design: .rounded))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Color.white.opacity(0.18))
                .cornerRadius(10)
            }
            .buttonStyle(.plain)

        default:
            EmptyView()
        }
    }
}
