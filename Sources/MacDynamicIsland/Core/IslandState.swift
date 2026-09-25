import SwiftUI
import Combine

public enum FaceIDStatus: Equatable {
    case ready
    case scanning
    case success(userName: String)
    case failed(reason: String)
}

public enum IslandMode: Equatable {
    case idle
    case hover
    case welcome
    case faceID(FaceIDStatus)
    case media(title: String, artist: String, isPlaying: Bool)
    case battery(percentage: Int, isCharging: Bool)
}

@MainActor
public final class IslandState: ObservableObject {
    public static let shared = IslandState()

    @Published public var currentMode: IslandMode = .idle
    @Published public var isHovered: Bool = false
    @Published public var customWidth: CGFloat? = nil
    @Published public var customHeight: CGFloat? = nil

    private var autoCollapseTimer: AnyCancellable?

    public init() {}

    /// Target width based on mode and metrics
    public func targetWidth(for metrics: NotchMetrics) -> CGFloat {
        if let custom = customWidth { return custom }

        switch currentMode {
        case .idle:
            return metrics.notchWidth
        case .hover:
            return max(metrics.notchWidth + 90, 260)
        case .welcome:
            return 310
        case .faceID:
            return 320
        case .media:
            return 340
        case .battery:
            return 280
        }
    }

    /// Target height based on mode and metrics
    public func targetHeight(for metrics: NotchMetrics) -> CGFloat {
        if let custom = customHeight { return custom }

        switch currentMode {
        case .idle:
            return metrics.notchHeight
        case .hover:
            return 52
        case .welcome:
            return 56
        case .faceID:
            return 64
        case .media:
            return 62
        case .battery:
            return 56
        }
    }

    public var cornerRadius: CGFloat {
        switch currentMode {
        case .idle:
            return 14
        default:
            return 24
        }
    }

    public func setHovered(_ hovered: Bool) {
        guard isHovered != hovered else { return }
        isHovered = hovered

        if hovered {
            if currentMode == .idle {
                setMode(.hover)
            }
        } else {
            if currentMode == .hover {
                setMode(.idle)
            }
        }
    }

    public func setMode(_ mode: IslandMode, autoCollapseAfter seconds: Double? = nil) {
        autoCollapseTimer?.cancel()
        currentMode = mode

        if let seconds = seconds {
            autoCollapseTimer = Just(())
                .delay(for: .seconds(seconds), scheduler: RunLoop.main)
                .sink { [weak self] _ in
                    self?.collapseToIdle()
                }
        }
    }

    public func collapseToIdle() {
        if isHovered {
            currentMode = .hover
        } else {
            currentMode = .idle
        }
    }

    // Convenience triggers
    public func triggerWelcomePulse() {
        setMode(.welcome, autoCollapseAfter: 4.0)
    }

    public func triggerFaceIDDemo() {
        setMode(.faceID(.scanning))
        
        // Simulate real-time biometric scanning sequence
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
            guard let self = self else { return }
            self.setMode(.faceID(.success(userName: "Devansh")), autoCollapseAfter: 3.0)
        }
    }

    public func triggerBatteryPulse(percentage: Int = 94, isCharging: Bool = true) {
        setMode(.battery(percentage: percentage, isCharging: isCharging), autoCollapseAfter: 4.0)
    }

    public func triggerMediaDemo() {
        setMode(.media(title: "Starboy", artist: "The Weeknd", isPlaying: true), autoCollapseAfter: 6.0)
    }
}
