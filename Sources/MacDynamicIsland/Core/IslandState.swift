import SwiftUI
import Combine
import IOKit.ps

public enum FaceIDStatus: Equatable {
    case ready
    case scanning
    case notEnrolled
    case success(userName: String, similarity: Float = 0.95)
    case failed(reason: String)
}

public enum IslandMode: Equatable {
    case idle
    case hover
    case faceID(FaceIDStatus)
    case enrollment
    case media(title: String, artist: String, isPlaying: Bool)
    case battery(percentage: Int, isCharging: Bool, isPluggedIn: Bool)
}

@MainActor
public final class IslandState: ObservableObject {
    public static let shared = IslandState()

    @Published public var currentMode: IslandMode = .idle
    @Published public var isHovered: Bool = false

    private var autoCollapseTimer: AnyCancellable?
    private var hoverExitTask: Task<Void, Never>?
    private var cancellables = Set<AnyCancellable>()

    public init() {
        observeSubsystems()
    }

    // MARK: - Subsystem Observation

    private func observeSubsystems() {
        // Observe FaceIDManager authState updates
        FaceIDManager.shared.$authState
            .receive(on: RunLoop.main)
            .sink { [weak self] state in
                guard let self = self else { return }

                switch state {
                case .detecting, .verifying:
                    if case .faceID = self.currentMode {
                        self.setMode(.faceID(.scanning))
                    }
                case .authorized(let user, let sim):
                    self.setMode(.faceID(.success(userName: user, similarity: sim)), autoCollapseAfter: 3.0)
                case .unauthorized(let reason):
                    if case .faceID = self.currentMode {
                        if reason.contains("not enrolled") {
                            self.setMode(.faceID(.notEnrolled), autoCollapseAfter: 6.0)
                        } else {
                            self.setMode(.faceID(.failed(reason: reason)), autoCollapseAfter: 3.5)
                        }
                    }
                case .idle:
                    break
                }
            }
            .store(in: &cancellables)

        // Observe EnrollmentManager state completion
        EnrollmentManager.shared.$state
            .receive(on: RunLoop.main)
            .sink { [weak self] state in
                guard let self = self else { return }
                switch state {
                case .completed:
                    self.autoCollapseTimer?.cancel()
                    self.autoCollapseTimer = Just(())
                        .delay(for: .seconds(3.0), scheduler: RunLoop.main)
                        .sink { [weak self] _ in
                            self?.collapseToIdle()
                        }
                default:
                    break
                }
            }
            .store(in: &cancellables)
    }

    // MARK: - Dimensions

    public var isExpanded: Bool {
        currentMode != .idle
    }

    public func currentWidth(for metrics: NotchMetrics) -> CGFloat {
        switch currentMode {
        case .idle:
            return metrics.idleShapeWidth
        case .hover:
            return max(metrics.notchWidth + 190, 390)
        case .faceID:
            return max(metrics.notchWidth + 220, 430)
        case .enrollment:
            return max(metrics.notchWidth + 250, 460)
        case .media:
            return max(metrics.notchWidth + 210, 420)
        case .battery:
            return max(metrics.notchWidth + 180, 380)
        }
    }

    public func currentHeight(for metrics: NotchMetrics) -> CGFloat {
        let contentHeight: CGFloat
        switch currentMode {
        case .idle:
            return metrics.idleShapeHeight
        case .hover:
            contentHeight = 58
        case .faceID:
            contentHeight = 68
        case .enrollment:
            contentHeight = 74
        case .media:
            contentHeight = 62
        case .battery:
            contentHeight = 58
        }

        return metrics.notchHeight + contentHeight
    }

    public func topCornerRadius() -> CGFloat {
        isExpanded ? 14 : 6
    }

    public func bottomCornerRadius() -> CGFloat {
        isExpanded ? 22 : 14
    }

    // MARK: - Hit-Testing Bounds (AppKit coordinate space: bottom-left = 0,0)

    public func interactiveRect(in bounds: CGRect, metrics: NotchMetrics) -> CGRect {
        let curW = currentWidth(for: metrics)
        let curH = currentHeight(for: metrics)

        let padX: CGFloat = isExpanded ? 6 : 28
        let padY: CGFloat = isExpanded ? 6 : 18

        let width = curW + padX * 2
        let height = curH + padY

        let x = (bounds.width - width) / 2.0
        let y = bounds.height - height

        return CGRect(x: x, y: y, width: width, height: height)
    }

    // MARK: - Hover & Mode Transitions

    public func setHovered(_ hovered: Bool) {
        hoverExitTask?.cancel()

        if hovered {
            isHovered = true
            if currentMode == .idle {
                withAnimation(.interactiveSpring(response: 0.38, dampingFraction: 0.8, blendDuration: 0)) {
                    self.currentMode = .hover
                }
            }
        } else {
            // Debounce collapse to eliminate cursor flicker / gaps between controls
            hoverExitTask = Task { @MainActor in
                try? await Task.sleep(nanoseconds: 220_000_000) // 220ms
                guard !Task.isCancelled else { return }
                self.isHovered = false
                if self.currentMode == .hover {
                    withAnimation(.interactiveSpring(response: 0.38, dampingFraction: 0.8, blendDuration: 0)) {
                        self.currentMode = .idle
                    }
                }
            }
        }
    }

    public func setMode(_ mode: IslandMode, autoCollapseAfter seconds: Double? = nil) {
        autoCollapseTimer?.cancel()
        withAnimation(.interactiveSpring(response: 0.38, dampingFraction: 0.8, blendDuration: 0)) {
            self.currentMode = mode
        }

        if let seconds = seconds {
            autoCollapseTimer = Just(())
                .delay(for: .seconds(seconds), scheduler: RunLoop.main)
                .sink { [weak self] _ in
                    self?.collapseToIdle()
                }
        }
    }

    public func collapseToIdle() {
        withAnimation(.interactiveSpring(response: 0.38, dampingFraction: 0.8, blendDuration: 0)) {
            if self.isHovered {
                self.currentMode = .hover
            } else {
                self.currentMode = .idle
            }
        }
    }

    // MARK: - Actions

    public func triggerFaceID() {
        FaceIDManager.shared.reloadEnrollmentStatus()
        if FaceIDManager.shared.isEnrolled {
            setMode(.faceID(.scanning), autoCollapseAfter: 8.0)
            FaceIDManager.shared.startAuthentication()
        } else {
            setMode(.faceID(.notEnrolled), autoCollapseAfter: 8.0)
        }
    }

    public func triggerEnrollment(userName: String = "User") {
        setMode(.enrollment)
        EnrollmentManager.shared.startEnrollment(userName: userName)
    }

    public func triggerBattery() {
        let (pct, charging, plugged) = Self.readSystemBattery()
        setMode(.battery(percentage: pct, isCharging: charging, isPluggedIn: plugged), autoCollapseAfter: 5.0)
    }

    public func triggerMedia() {
        setMode(.media(title: "Dynamic Island", artist: "macOS Audio", isPlaying: true), autoCollapseAfter: 5.0)
    }

    // MARK: - Battery Reading

    public static func readSystemBattery() -> (percentage: Int, isCharging: Bool, isPluggedIn: Bool) {
        guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [Any],
              let firstSource = sources.first,
              let info = IOPSGetPowerSourceDescription(snapshot, firstSource as CFTypeRef)?.takeUnretainedValue() as? [String: Any] else {
            return (percentage: -1, isCharging: false, isPluggedIn: false)
        }

        let current = info[kIOPSCurrentCapacityKey] as? Int ?? -1
        let maxCap = info[kIOPSMaxCapacityKey] as? Int ?? 100

        let percentage: Int
        if maxCap > 0 && current >= 0 {
            percentage = Int(round(Double(current) / Double(maxCap) * 100.0))
        } else {
            percentage = current
        }

        let isCharging = (info[kIOPSIsChargingKey] as? Bool) ?? false
        let powerState = info["Power Source State"] as? String ?? ""
        let isPluggedIn = isCharging || powerState == "AC Power"

        return (percentage: percentage, isCharging: isCharging, isPluggedIn: isPluggedIn)
    }
}
