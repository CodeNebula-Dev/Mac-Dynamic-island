import AppKit
import SwiftUI

@MainActor
@main
public final class MacDynamicIslandApp: NSObject, NSApplicationDelegate {
    private var notchWindow: NotchWindow?
    private var statusItem: NSStatusItem?
    private let islandState = IslandState.shared
    private let faceIDManager = FaceIDManager.shared

    public static func main() {
        let app = NSApplication.shared
        let delegate = MacDynamicIslandApp()
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
    }

    public func applicationDidFinishLaunching(_ notification: Notification) {
        setupStatusBar()
        setupNotchWindow()
    }

    private func setupNotchWindow() {
        let window = NotchWindow(state: islandState)
        window.makeKeyAndOrderFront(nil)
        self.notchWindow = window
    }

    private func setupStatusBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        if let button = statusItem?.button {
            button.image = NSImage(
                systemSymbolName: "capsule.portrait.fill",
                accessibilityDescription: "Dynamic Island"
            )
        }

        let menu = NSMenu()

        // Face ID Action
        let faceIDItem = NSMenuItem(
            title: "Scan Face (Face ID)",
            action: #selector(triggerFaceID),
            keyEquivalent: "f"
        )
        faceIDItem.target = self
        menu.addItem(faceIDItem)

        // Media Action
        let mediaItem = NSMenuItem(
            title: "Simulate Media Playing",
            action: #selector(triggerMedia),
            keyEquivalent: "m"
        )
        mediaItem.target = self
        menu.addItem(mediaItem)

        // Battery Action
        let batteryItem = NSMenuItem(
            title: "Simulate MagSafe Charge",
            action: #selector(triggerBattery),
            keyEquivalent: "b"
        )
        batteryItem.target = self
        menu.addItem(batteryItem)

        menu.addItem(NSMenuItem.separator())

        // Reset
        let resetItem = NSMenuItem(
            title: "Collapse Island",
            action: #selector(collapseIsland),
            keyEquivalent: "r"
        )
        resetItem.target = self
        menu.addItem(resetItem)

        menu.addItem(NSMenuItem.separator())

        // Quit
        let quitItem = NSMenuItem(
            title: "Quit Mac Dynamic Island",
            action: #selector(quitApp),
            keyEquivalent: "q"
        )
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem?.menu = menu
    }

    // MARK: - Actions

    @objc private func triggerFaceID() {
        islandState.triggerFaceIDDemo()
        faceIDManager.startAuthentication()
    }

    @objc private func triggerMedia() {
        islandState.triggerMediaDemo()
    }

    @objc private func triggerBattery() {
        islandState.triggerBatteryPulse()
    }

    @objc private func collapseIsland() {
        islandState.collapseToIdle()
    }

    @objc private func quitApp() {
        NSApplication.shared.terminate(nil)
    }
}
