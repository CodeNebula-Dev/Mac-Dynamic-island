import Foundation
import CoreGraphics

/// Diagnostic self-test suite for Mac Dynamic Island & Face ID subsystems
@MainActor
public struct SelfTestRunner {
    public static func run() -> Int32 {
        print("""
        \u{001B}[1;36m╔══════════════════════════════════════════════════════════════════╗
        ║      Mac Dynamic Island & Face ID — Self-Diagnostic Suite        ║
        ╚══════════════════════════════════════════════════════════════════╝\u{001B}[0m
        """)

        var passed = 0
        var failed = 0

        func report(name: String, ok: Bool, details: String = "") {
            if ok {
                passed += 1
                print("  \u{001B}[32m✓ [PASS]\u{001B}[0m \(name) \(details)")
            } else {
                failed += 1
                print("  \u{001B}[31m✗ [FAIL]\u{001B}[0m \(name) \(details)")
            }
        }

        // Test 1: Hardware Notch Detection & Screen Geometry
        print("\n\u{001B}[1m1. Hardware Notch & Display Metrics\u{001B}[0m")
        let metrics = NotchMetrics.current()
        let hasNotch = metrics.hasNotch
        let width = Int(metrics.screenFrame.width)
        let height = Int(metrics.screenFrame.height)
        let details = "Screen: \(width)x\(height) | Notch: \(Int(metrics.notchWidth))x\(Int(metrics.notchHeight)) | Physical Notch: \(hasNotch)"
        report(name: "NotchMetrics.current()", ok: width > 0 && height > 0, details: details)

        // Test 2: System Battery Subsystem
        print("\n\u{001B}[1m2. System Power & Battery HUD\u{001B}[0m")
        let (batteryPct, charging, plugged) = IslandState.readSystemBattery()
        let battDetails = "Capacity: \(batteryPct)% | Charging: \(charging) | AC Power: \(plugged)"
        report(name: "IslandState.readSystemBattery()", ok: batteryPct >= -1 && batteryPct <= 100, details: battDetails)

        // Test 3: Cosine Similarity Vector Mathematics
        print("\n\u{001B}[1m3. 512-D Embedding Vector Mathematics\u{001B}[0m")
        let unitV = [Float](repeating: 1.0 / sqrt(512.0), count: 512)
        let identicalSim = FaceEmbeddingModel.cosineSimilarity(u: unitV, v: unitV)
        report(
            name: "Cosine Similarity (Identical Vectors)",
            ok: abs(identicalSim - 1.0) < 1e-4,
            details: String(format: "Expected 1.000, got %.4f", identicalSim)
        )

        var u = [Float](repeating: 0.0, count: 512)
        var v = [Float](repeating: 0.0, count: 512)
        u[0] = 1.0
        v[1] = 1.0
        let orthoSim = FaceEmbeddingModel.cosineSimilarity(u: u, v: v)
        report(
            name: "Cosine Similarity (Orthogonal Vectors)",
            ok: abs(orthoSim) < 1e-4,
            details: String(format: "Expected 0.000, got %.4f", orthoSim)
        )

        // Test 4: 5-Angle Centroid Generation
        print("\n\u{001B}[1m4. Multi-Angle Centroid Generation\u{001B}[0m")
        var syntheticAngles: [[Float]] = []
        for i in 0..<5 {
            var sample = unitV
            sample[i] += 0.04
            let norm = sqrt(sample.map { $0 * $0 }.reduce(0, +))
            syntheticAngles.append(sample.map { $0 / norm })
        }

        if let centroid = FaceEmbeddingModel.computeCentroid(embeddings: syntheticAngles) {
            let norm = sqrt(centroid.map { $0 * $0 }.reduce(0, +))
            let isNormalized = abs(norm - 1.0) < 1e-4
            var allHigh = true
            for (_, angleEmbed) in syntheticAngles.enumerated() {
                let sim = FaceEmbeddingModel.cosineSimilarity(u: centroid, v: angleEmbed)
                if sim < 0.98 { allHigh = false }
            }
            report(
                name: "Centroid L2 Normalization & Discrimination",
                ok: centroid.count == 512 && isNormalized && allHigh,
                details: String(format: "Centroid ||c||_2 = %.4f (5 angles)", norm)
            )
        } else {
            report(name: "Centroid Computation", ok: false, details: "Returned nil")
        }

        // Test 5: Umeyama 2D Face Alignment Transform
        print("\n\u{001B}[1m5. Umeyama 2D Face Alignment Transform\u{001B}[0m")
        let srcPoints = [
            CGPoint(x: 10, y: 10),
            CGPoint(x: 50, y: 10),
            CGPoint(x: 30, y: 30),
            CGPoint(x: 15, y: 50),
            CGPoint(x: 45, y: 50)
        ]
        let tgtPoints = [
            CGPoint(x: 110, y: 210),
            CGPoint(x: 150, y: 210),
            CGPoint(x: 130, y: 230),
            CGPoint(x: 115, y: 250),
            CGPoint(x: 145, y: 250)
        ]
        if let transform = FaceAligner.shared.computeSimilarityTransform(sourcePoints: srcPoints, targetPoints: tgtPoints) {
            let txOk = abs(transform.tx - 100.0) < 1e-3
            let tyOk = abs(transform.ty - 200.0) < 1e-3
            let scaleOk = abs(transform.a - 1.0) < 1e-3
            report(
                name: "Similarity Transform (dx=100, dy=200, scale=1.0)",
                ok: txOk && tyOk && scaleOk,
                details: String(format: "tx=%.1f, ty=%.1f, a=%.2f", transform.tx, transform.ty, transform.a)
            )
        } else {
            report(name: "Similarity Transform", ok: false, details: "Returned nil")
        }

        // Test 6: Biometric Keychain Vault Storage
        print("\n\u{001B}[1m6. Biometric Keychain Vault\u{001B}[0m")
        let testEmbedding = [Float](repeating: 0.05, count: 512)
        let testTemplate = EnrolledFaceTemplate(
            userName: "SelfTestUser",
            embedding: testEmbedding,
            angleEmbeddings: [testEmbedding]
        )
        let saved = BiometricKeychain.shared.store(template: testTemplate)
        let retrieved = BiometricKeychain.shared.retrieveTemplate()
        let match = retrieved?.userName == "SelfTestUser" && retrieved?.embedding.count == 512
        let deleted = BiometricKeychain.shared.deleteTemplate()

        report(
            name: "Keychain Store / Retrieve / Delete Cycle",
            ok: saved && match && deleted,
            details: "Secured with kSecAttrAccessibleWhenUnlockedThisDeviceOnly"
        )

        // Test 7: Enrollment Angles Specification
        print("\n\u{001B}[1m7. 5-Angle Head Pose Requirements\u{001B}[0m")
        let angles = EnrollmentManager.EnrollmentAngle.allCases
        let anglesOk = (angles.count == 5) &&
            (EnrollmentManager.EnrollmentAngle.frontal.targetYawDeg == 0.0) &&
            (EnrollmentManager.EnrollmentAngle.left.targetYawDeg == 15.0) &&
            (EnrollmentManager.EnrollmentAngle.right.targetYawDeg == -15.0) &&
            (EnrollmentManager.EnrollmentAngle.up.targetPitchDeg == 10.0) &&
            (EnrollmentManager.EnrollmentAngle.down.targetPitchDeg == -10.0)
        report(
            name: "5-Angle Yaw & Pitch Targets",
            ok: anglesOk,
            details: "Frontal (0°), Left (+15° Vision yaw), Right (-15° Vision yaw), Up (+10°), Down (-10°)"
        )

        // Summary
        print("""
        \n\u{001B}[1mDiagnostic Summary:\u{001B}[0m
          Passed: \u{001B}[32m\(passed)\u{001B}[0m / \(passed + failed) tests
          Failed: \u{001B}[\(failed == 0 ? "32" : "31")m\(failed)\u{001B}[0m
        """)

        if failed == 0 {
            print("\u{001B}[1;32m✓ All system engines, math, and storage components validated successfully!\u{001B}[0m\n")
            return 0
        } else {
            print("\u{001B}[1;31m✗ Diagnostics reported \(failed) failure(s).\u{001B}[0m\n")
            return 1
        }
    }
}
