import SwiftUI

public struct FaceIDHUDView: View {
    public let status: FaceIDStatus
    @State private var rotation: Double = 0.0

    public init(status: FaceIDStatus) {
        self.status = status
    }

    public var body: some View {
        HStack(spacing: 14) {
            // Icon with glowing effect
            ZStack {
                switch status {
                case .ready, .scanning:
                    Circle()
                        .stroke(
                            AngularGradient(
                                gradient: Gradient(colors: [.cyan, .blue, .purple, .cyan]),
                                center: .center
                            ),
                            lineWidth: 2.5
                        )
                        .frame(width: 34, height: 34)
                        .rotationEffect(.degrees(rotation))
                        .onAppear {
                            withAnimation(
                                .linear(duration: 1.5).repeatForever(autoreverses: false)
                            ) {
                                rotation = 360
                            }
                        }

                    Image(systemName: "faceid")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.cyan)

                case .success:
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundColor(.green)
                        .transition(.scale.combined(with: .opacity))

                case .failed:
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundColor(.red)
                        .transition(.scale.combined(with: .opacity))
                }
            }

            // Status label & subtitle
            VStack(alignment: .leading, spacing: 2) {
                switch status {
                case .ready:
                    Text("Face ID")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)
                    Text("Looking for Face...")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.gray)

                case .scanning:
                    Text("Authenticating...")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)
                    Text("Apple Neural Engine Active")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.cyan.opacity(0.8))

                case .success(let name):
                    Text("Face ID Verified")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                        .foregroundColor(.green)
                    Text("Welcome back, \(name)")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.white.opacity(0.85))

                case .failed(let reason):
                    Text("Face ID Failed")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                        .foregroundColor(.red)
                    Text(reason)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.gray)
                }
            }

            Spacer()

            // Mac Apple Silicon Badge
            HStack(spacing: 4) {
                Image(systemName: "cpu")
                    .font(.system(size: 10))
                Text("ANE")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
            }
            .foregroundColor(.white.opacity(0.6))
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(Color.white.opacity(0.12))
            .cornerRadius(6)
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
