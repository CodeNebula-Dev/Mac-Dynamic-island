import AppKit
import SwiftUI
import Combine

/// Borderless, non-activating floating panel positioned precisely over the MacBook notch
@MainActor
public final class NotchWindow: NSPanel {
    private let islandState: IslandState
    private var cancellables = Set<AnyCancellable>()
    private var trackingArea: NSTrackingArea?

    public init(state: IslandState) {
        self.islandState = state

        super.init(
            contentRect: .zero,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        configureWindow()
        setupContentView()
        observeState()
        updatePosition()
    }

    private func configureWindow() {
        self.isFloatingPanel = true
        self.level = .statusBar
        self.backgroundColor = .clear
        self.isOpaque = false
        self.hasShadow = false
        self.ignoresMouseEvents = false
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        self.isMovableByWindowBackground = false
    }

    private func setupContentView() {
        let rootView = DynamicIslandView(state: islandState, metrics: NotchMetrics.current())
        let hostingView = NSHostingView(rootView: rootView)
        hostingView.wantsLayer = true
        hostingView.layer?.backgroundColor = NSColor.clear.cgColor
        self.contentView = hostingView
    }

    private func observeState() {
        // Observe mode and hover changes to reposition and resize the window frame smoothly
        Publishers.CombineLatest(islandState.$currentMode, islandState.$isHovered)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.updatePosition(animated: true)
            }
            .store(in: &cancellables)

        // Observe screen configuration changes (e.g. connecting external monitor)
        NotificationCenter.default.publisher(for: NSApplication.didChangeScreenParametersNotification)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.updatePosition(animated: false)
            }
            .store(in: &cancellables)
    }

    public func updatePosition(animated: Bool = false) {
        let metrics = NotchMetrics.current(for: NSScreen.main)
        let targetWidth = islandState.targetWidth(for: metrics)
        let targetHeight = islandState.targetHeight(for: metrics)
        let newFrame = metrics.islandFrame(width: targetWidth, height: targetHeight)

        if animated {
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.35
                context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                self.animator().setFrame(newFrame, display: true)
            }
        } else {
            self.setFrame(newFrame, display: true)
        }

        updateTrackingArea(for: newFrame)
    }

    private func updateTrackingArea(for frame: CGRect) {
        guard let view = self.contentView else { return }
        if let existing = trackingArea {
            view.removeTrackingArea(existing)
        }

        let area = NSTrackingArea(
            rect: view.bounds,
            options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        view.addTrackingArea(area)
        self.trackingArea = area
    }

    public override func mouseEntered(with event: NSEvent) {
        super.mouseEntered(with: event)
        islandState.setHovered(true)
    }

    public override func mouseExited(with event: NSEvent) {
        super.mouseExited(with: event)
        islandState.setHovered(false)
    }
}
