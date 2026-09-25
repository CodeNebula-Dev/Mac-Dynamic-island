import AppKit
import SwiftUI
import Combine

/// Borderless, non-activating floating panel positioned at the physical camera notch.
/// Maintains a fixed stationary frame to eliminate any AppKit window-resizing stutter or flickering.
/// Hit-testing passes through events outside the dynamic island region to preserve full desktop interactivity.
@MainActor
public final class NotchWindow: NSPanel {
    private let islandState: IslandState
    private var cancellables = Set<AnyCancellable>()

    public init(state: IslandState) {
        self.islandState = state

        let metrics = NotchMetrics.current(for: NSScreen.main)
        super.init(
            contentRect: metrics.windowFrame,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        configureWindow()
        setupContentView()
        observeScreenChanges()
    }

    // MARK: - Window Configuration

    private func configureWindow() {
        self.isFloatingPanel = true
        self.level = .mainMenu + 3
        self.backgroundColor = .clear
        self.isOpaque = false
        self.hasShadow = false
        self.ignoresMouseEvents = false
        self.collectionBehavior = [
            .canJoinAllSpaces,
            .fullScreenAuxiliary,
            .stationary,
            .ignoresCycle
        ]
        self.isMovableByWindowBackground = false
    }

    private func setupContentView() {
        let rootView = DynamicIslandView(state: islandState)
        let hostingView = IslandHostingView(rootView: rootView, state: islandState)
        hostingView.wantsLayer = true
        hostingView.layer?.backgroundColor = NSColor.clear.cgColor
        self.contentView = hostingView
    }

    // MARK: - Screen Changes

    private func observeScreenChanges() {
        NotificationCenter.default.publisher(for: NSApplication.didChangeScreenParametersNotification)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.repositionWindow()
            }
            .store(in: &cancellables)
    }

    public func repositionWindow() {
        let metrics = NotchMetrics.current(for: NSScreen.main)
        self.setFrame(metrics.windowFrame, display: true)
    }

    public override var canBecomeKey: Bool { false }
    public override var canBecomeMain: Bool { false }
}

// MARK: - Interactive Island Hosting View

/// Custom NSHostingView that dynamically adjusts its interactive hit-testing zone
/// and tracks mouse entry / exit cleanly without window resizing.
private final class IslandHostingView<Content: View>: NSHostingView<Content> {
    private weak var islandState: IslandState?

    init(rootView: Content, state: IslandState) {
        self.islandState = state
        super.init(rootView: rootView)
    }

    @MainActor required public init(rootView: Content) {
        super.init(rootView: rootView)
    }

    @MainActor required dynamic init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas {
            removeTrackingArea(area)
        }
        let trackingArea = NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .mouseMoved, .activeAlways, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(trackingArea)
    }

    override func hitTest(_ point: NSPoint) -> NSView? {
        guard let state = islandState else { return super.hitTest(point) }
        let metrics = NotchMetrics.current()
        let activeRect = state.interactiveRect(in: bounds, metrics: metrics)

        if activeRect.contains(point) {
            return super.hitTest(point)
        } else {
            // Pass through events to the desktop / menu bar behind
            return nil
        }
    }

    override func mouseMoved(with event: NSEvent) {
        super.mouseMoved(with: event)
        guard let state = islandState else { return }
        let point = convert(event.locationInWindow, from: nil)
        let metrics = NotchMetrics.current()
        let activeRect = state.interactiveRect(in: bounds, metrics: metrics)

        if activeRect.contains(point) {
            if !state.isHovered && state.currentMode == .idle {
                state.setHovered(true)
            }
        } else {
            if state.isHovered && state.currentMode == .hover {
                state.setHovered(false)
            }
        }
    }

    override func mouseExited(with event: NSEvent) {
        super.mouseExited(with: event)
        islandState?.setHovered(false)
    }
}
