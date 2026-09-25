import SwiftUI

public struct BatteryHUDView: View {
    public let percentage: Int
    public let isCharging: Bool

    public init(percentage: Int, isCharging: Bool) {
        self.percentage = percentage
        self.isCharging = isCharging
    }

    public var body: some View {
        HStack(spacing: 12) {
            // MagSafe Charging Ring or Battery Icon
            ZStack {
                Circle()
                    .stroke(Color.green.opacity(0.3), lineWidth: 3)
                    .frame(width: 32, height: 32)

                Circle()
                    .trim(from: 0.0, to: CGFloat(percentage) / 100.0)
                    .stroke(
                        Color.green,
                        style: StrokeStyle(lineWidth: 3, lineCap: .round)
                    )
                    .rotationEffect(.degrees(-90))
                    .frame(width: 32, height: 32)

                if isCharging {
                    Image(systemName: "bolt.fill")
                        .foregroundColor(.green)
                        .font(.system(size: 13, weight: .bold))
                } else {
                    Text("\(percentage)")
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                }
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(isCharging ? "MagSafe Connected" : "Battery Status")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)

                Text("\(percentage)% Charged")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.gray)
            }

            Spacer()

            // Green status glow pill
            HStack(spacing: 4) {
                Circle()
                    .fill(Color.green)
                    .frame(width: 6, height: 6)
                Text("HEALTHY")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
            }
            .foregroundColor(.green)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(Color.green.opacity(0.15))
            .cornerRadius(6)
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
