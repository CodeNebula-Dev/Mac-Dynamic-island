import AppKit
import Foundation

/// Geometry and notch layout detector for Apple Silicon MacBooks and external displays
public struct NotchMetrics {
    public let hasNotch: Bool
    public let notchWidth: CGFloat
    public let notchHeight: CGFloat
    public let screenFrame: CGRect
    public let topMargin: CGFloat

    public static func current(for screen: NSScreen? = NSScreen.main) -> NotchMetrics {
        guard let screen = screen else {
            return NotchMetrics(
                hasNotch: false,
                notchWidth: 160,
                notchHeight: 32,
                screenFrame: CGRect(x: 0, y: 0, width: 1440, height: 900),
                topMargin: 0
            )
        }

        let frame = screen.frame
        let safeInsets = screen.safeAreaInsets
        let hasHardwareNotch = safeInsets.top > 0

        // Apple Silicon notch dimensions (typically ~160-180pt wide, ~32-34pt tall)
        let width: CGFloat
        let height: CGFloat

        if hasHardwareNotch {
            // Screen has physical notch
            width = 170.0
            height = safeInsets.top
        } else {
            // Fallback virtual pill dimensions for non-notch displays or external monitors
            width = 150.0
            height = 28.0
        }

        return NotchMetrics(
            hasNotch: hasHardwareNotch,
            notchWidth: width,
            notchHeight: height,
            screenFrame: frame,
            topMargin: hasHardwareNotch ? 0 : 8
        )
    }

    /// Calculates the origin and size for the Dynamic Island window
    public func islandFrame(width: CGFloat, height: CGFloat) -> CGRect {
        let x = screenFrame.origin.x + (screenFrame.width - width) / 2.0
        let y = screenFrame.origin.y + screenFrame.height - height - topMargin
        return CGRect(x: x, y: y, width: width, height: height)
    }
}
