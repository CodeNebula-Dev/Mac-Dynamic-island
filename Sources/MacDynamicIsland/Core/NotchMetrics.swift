import AppKit
import Foundation

/// Geometry detector using auxiliaryTopLeftArea / auxiliaryTopRightArea
/// for pixel-perfect alignment with Apple Silicon MacBook camera notches and external displays.
public struct NotchMetrics {
    public let hasNotch: Bool

    // Physical notch dimensions
    public let notchWidth: CGFloat    // e.g. 179pt on MacBook Pro 14"
    public let notchHeight: CGFloat   // e.g. 32pt (safeAreaInsets.top)
    public let notchOriginX: CGFloat  // e.g. 646pt (left edge of notch)
    public let screenFrame: CGRect

    // Fixed canvas window dimensions
    public static let windowWidth: CGFloat = 560
    public static let windowHeight: CGFloat = 200

    public static func current(for screen: NSScreen? = NSScreen.main) -> NotchMetrics {
        guard let screen = screen else {
            return NotchMetrics(
                hasNotch: false,
                notchWidth: 200,
                notchHeight: 32,
                notchOriginX: (1440 - 200) / 2.0,
                screenFrame: CGRect(x: 0, y: 0, width: 1440, height: 900)
            )
        }

        let frame = screen.frame
        let safeInsets = screen.safeAreaInsets
        let hasHardwareNotch = safeInsets.top > 0

        if hasHardwareNotch,
           let leftEar = screen.auxiliaryTopLeftArea,
           let rightEar = screen.auxiliaryTopRightArea {
            let computedWidth = frame.width - leftEar.width - rightEar.width
            let computedX = frame.origin.x + leftEar.width

            return NotchMetrics(
                hasNotch: true,
                notchWidth: computedWidth,
                notchHeight: safeInsets.top,
                notchOriginX: computedX,
                screenFrame: frame
            )
        } else {
            let defaultWidth: CGFloat = 210
            let defaultHeight: CGFloat = 32
            let computedX = frame.origin.x + (frame.width - defaultWidth) / 2.0

            return NotchMetrics(
                hasNotch: false,
                notchWidth: defaultWidth,
                notchHeight: defaultHeight,
                notchOriginX: computedX,
                screenFrame: frame
            )
        }
    }

    /// Center X of the notch in screen coordinates
    public var notchCenterX: CGFloat {
        notchOriginX + notchWidth / 2.0
    }

    /// Stationary NSPanel window frame anchored at screen top
    public var windowFrame: CGRect {
        let topOfScreen = screenFrame.origin.y + screenFrame.height
        let x = notchCenterX - Self.windowWidth / 2.0
        let y = topOfScreen - Self.windowHeight
        return CGRect(
            x: x,
            y: y,
            width: Self.windowWidth,
            height: Self.windowHeight
        )
    }

    /// Idle notch shape width (physical width + top flare allowances)
    public var idleShapeWidth: CGFloat {
        notchWidth + 12
    }

    /// Idle notch shape height (physical height)
    public var idleShapeHeight: CGFloat {
        notchHeight
    }
}
