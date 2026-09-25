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

                case .notEnrolled:
                    Image(systemName: "faceid")
                        .font(.system(size: 26, weight: .semibold))
                        .foregroundColor(.orange)

                case .success:
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 26, weight: .semibold))
                        .foregroundColor(.green)
                        .transition(.scale.combined(with: .opacity))

                case .failed:
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 26, weight: .semibold))
                        .foregroundColor(.red)
                        .transition(.scale.combined(with: .opacity))
                }
            }

            // Status label
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
                    Text("Scanning...")
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)
                    Text("Hold still")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.cyan.opacity(0.9))

                case .notEnrolled:
                    Text("Face ID Not Enrolled")
                        .font(.system(size: 15, weight: .bold, design: .rounded))
                        .foregroundColor(.orange)
                    Text("Enrollment required in Settings")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.gray)

                case .success(let name):
                    Text("Verified")
                        .font(.system(size: 15, weight: .bold, design: .rounded))
                        .foregroundColor(.green)
                    Text("Welcome, \(name)")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.white.opacity(0.9))

                case .failed(let reason):
                    Text("Face ID Failed")
                        .font(.system(size: 15, weight: .bold, design: .rounded))
                        .foregroundColor(.red)
                    Text(reason)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.gray)
                }
            }

            Spacer()
        }
        .padding(.horizontal, 8)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
