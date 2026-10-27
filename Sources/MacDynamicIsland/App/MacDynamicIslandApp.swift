import AppKit
import SwiftUI

@MainActor
@main
public final class MacDynamicIslandApp: NSObject, NSApplicationDelegate {
    private var notchWindow: NotchWindow?
    private var statusItem: NSStatusItem?
    private let islandState = IslandState.shared

    private static var sharedDelegate: MacDynamicIslandApp?

    public static func main() {
        if CommandLine.arguments.contains("--test") || CommandLine.arguments.contains("-t") {
            let exitCode = SelfTestRunner.run()
            exit(exitCode)
        }

        let app = NSApplication.shared
        let delegate = MacDynamicIslandApp()
        sharedDelegate = delegate
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
    }

    public func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return false
    }

    public func applicationDidFinishLaunching(_ notification: Notification) {
        setupStatusBar()
        setupNotchWindow()

        // Immediate activation: expand island smoothly so user visibly sees it activate!
        let args = CommandLine.arguments
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
            guard let self = self else { return }
            if args.contains("--enroll") || args.contains("-e") {
                self.islandState.triggerEnrollment()
            } else if args.contains("--faceid") || args.contains("-f") {
                self.islandState.triggerFaceID()
            } else if args.contains("--battery") || args.contains("-b") {
                self.islandState.triggerBattery()
            } else if args.contains("--media") || args.contains("-m") {
                self.islandState.triggerMedia()
            } else {
                // Default startup: animate into expanded hover mode for 4s so it visibly pops out!
                self.islandState.setMode(.hover, autoCollapseAfter: 4.0)
            }
        }
    }

    private func setupNotchWindow() {
        let window = NotchWindow(state: islandState)
        window.orderFrontRegardless()
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

        let faceIDItem = NSMenuItem(title: "Face ID", action: #selector(triggerFaceID), keyEquivalent: "f")
        faceIDItem.target = self
        menu.addItem(faceIDItem)

        let enrollItem = NSMenuItem(title: "Enroll Face ID...", action: #selector(triggerEnrollment), keyEquivalent: "e")
        enrollItem.target = self
        menu.addItem(enrollItem)

        let batteryItem = NSMenuItem(title: "Battery Status", action: #selector(triggerBattery), keyEquivalent: "b")
        batteryItem.target = self
        menu.addItem(batteryItem)

        let mediaItem = NSMenuItem(title: "Media Info", action: #selector(triggerMedia), keyEquivalent: "m")
        mediaItem.target = self
        menu.addItem(mediaItem)

        menu.addItem(NSMenuItem.separator())

        let resetItem = NSMenuItem(title: "Collapse", action: #selector(collapseIsland), keyEquivalent: "r")
        resetItem.target = self
        menu.addItem(resetItem)

        menu.addItem(NSMenuItem.separator())

        let quitItem = NSMenuItem(title: "Quit", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem?.menu = menu
    }

    // MARK: - Actions

    @objc private func triggerFaceID() {
        islandState.triggerFaceID()
    }

    @objc private func triggerEnrollment() {
        islandState.triggerEnrollment()
    }

    @objc private func triggerBattery() {
        islandState.triggerBattery()
    }

    @objc private func triggerMedia() {
        islandState.triggerMedia()
    }

    @objc private func collapseIsland() {
        islandState.collapseToIdle()
    }

    @objc private func quitApp() {
        NSApplication.shared.terminate(nil)
    }
}
