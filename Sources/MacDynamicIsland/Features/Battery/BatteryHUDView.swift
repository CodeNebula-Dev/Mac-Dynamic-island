import SwiftUI

public struct BatteryHUDView: View {
    public let percentage: Int
    public let isCharging: Bool
    public let isPluggedIn: Bool

    public init(percentage: Int, isCharging: Bool, isPluggedIn: Bool = false) {
        self.percentage = percentage
        self.isCharging = isCharging
        self.isPluggedIn = isPluggedIn
    }

    private var batteryColor: Color {
        if percentage < 0 { return .gray }
        if isCharging || isPluggedIn { return .green }
        if percentage <= 20 { return .red }
        if percentage <= 40 { return .yellow }
        return .green
    }

    private var statusTitle: String {
        if isCharging {
            return "Charging"
        } else if isPluggedIn {
            return "Power Adapter"
        } else {
            return "On Battery"
        }
    }

    public var body: some View {
        HStack(spacing: 14) {
            // Circular charge indicator
            ZStack {
                Circle()
                    .stroke(batteryColor.opacity(0.22), lineWidth: 3.5)
                    .frame(width: 36, height: 36)

                if percentage >= 0 {
                    Circle()
                        .trim(from: 0.0, to: CGFloat(min(max(percentage, 0), 100)) / 100.0)
                        .stroke(
                            batteryColor,
                            style: StrokeStyle(lineWidth: 3.5, lineCap: .round)
                        )
                        .rotationEffect(.degrees(-90))
                        .frame(width: 36, height: 36)
                }

                if isCharging {
                    Image(systemName: "bolt.fill")
                        .foregroundColor(batteryColor)
                        .font(.system(size: 15, weight: .bold))
                } else if isPluggedIn {
                    Image(systemName: "powerplug.fill")
                        .foregroundColor(batteryColor)
                        .font(.system(size: 14, weight: .bold))
                } else {
                    Text(percentage >= 0 ? "\(percentage)" : "?")
                        .font(.system(size: 12, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                }
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(statusTitle)
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)

                Text(percentage >= 0 ? "\(percentage)% Remaining" : "Status Unknown")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.gray)
            }

            Spacer()
        }
        .padding(.horizontal, 8)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
