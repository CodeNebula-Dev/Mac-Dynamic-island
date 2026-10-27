import Foundation
import AVFoundation
import Vision
import Combine
import CoreVideo

/// 5-Angle Guided Face Enrollment State Machine
/// Captures user's face across 5 canonical angles (Frontal, Left, Right, Up, Down),
/// extracts 512-D embeddings, computes the normalized centroid template,
/// and securely stores the biometric reference into macOS Keychain.
public final class EnrollmentManager: ObservableObject {
    public static let shared = EnrollmentManager()

    public enum EnrollmentAngle: Int, CaseIterable, Identifiable {
        case frontal = 0
        case left = 1
        case right = 2
        case up = 3
        case down = 4

        public var id: Int { rawValue }

        public var title: String {
            switch self {
            case .frontal: return "Look Straight Ahead"
            case .left: return "Turn Head Left"
            case .right: return "Turn Head Right"
            case .up: return "Tilt Head Up"
            case .down: return "Tilt Head Down"
            }
        }

        public var shortTitle: String {
            switch self {
            case .frontal: return "Front"
            case .left: return "Left"
            case .right: return "Right"
            case .up: return "Up"
            case .down: return "Down"
            }
        }

        public var guidanceInstruction: String {
            switch self {
            case .frontal: return "Center your face in the frame"
            case .left: return "Slowly turn your head to the left"
            case .right: return "Slowly turn your head to the right"
            case .up: return "Tilt your chin slightly upward"
            case .down: return "Tilt your chin slightly downward"
            }
        }

        /// Target yaw in degrees.
        /// Apple Vision yaw convention: **positive = user turns right (from camera POV: face looks left)**
        /// So when the user turns their head LEFT, Vision reports a POSITIVE yaw.
        /// When the user turns their head RIGHT, Vision reports NEGATIVE yaw.
        public var targetYawDeg: Float {
            switch self {
            case .frontal: return 0.0
            case .left: return 15.0   // User turns left → Vision sees positive yaw
            case .right: return -15.0  // User turns right → Vision sees negative yaw
            case .up: return 0.0
            case .down: return 0.0
            }
        }

        /// Target pitch in degrees.
        /// Apple Vision pitch: positive = looking up, negative = looking down
        public var targetPitchDeg: Float {
            switch self {
            case .frontal: return 0.0
            case .left: return 0.0
            case .right: return 0.0
            case .up: return 10.0
            case .down: return -10.0
            }
        }

        /// Per-angle tolerance — frontal is tighter, side angles are wider
        public var yawTolerance: Float {
            switch self {
            case .frontal: return 10.0
            case .left, .right: return 12.0
            case .up, .down: return 12.0
            }
        }

        public var pitchTolerance: Float {
            switch self {
            case .frontal: return 10.0
            case .left, .right: return 10.0
            case .up, .down: return 10.0
            }
        }
    }

    public enum EnrollmentState: Equatable {
        case idle
        case preparing
        case capturing(angle: EnrollmentAngle, progress: Float)
        case processingCentroid
        case completed(userName: String)
        case failed(reason: String)
    }

    @Published public private(set) var state: EnrollmentState = .idle
    @Published public private(set) var currentAngleIndex: Int = 0
    @Published public private(set) var currentYaw: Float = 0.0
    @Published public private(set) var currentPitch: Float = 0.0
    @Published public private(set) var isAngleMatched: Bool = false
    @Published public private(set) var holdProgress: Float = 0.0

    private var enrolledUserName: String = "User"
    private var angleEmbeddings: [[Float]] = []
    private var holdCounter: Int = 0
    private let requiredHoldFrames: Int = 10 // ~0.33s at 30fps — snappier than before
    private let processingQueue = DispatchQueue(label: "com.codenebula.macdynamicisland.enrollment", qos: .userInitiated)

    public var isEnrolling: Bool {
        switch state {
        case .preparing, .capturing, .processingCentroid:
            return true
        default:
            return false
        }
    }

    private init() {}

    // MARK: - Enrollment Control

    /// Initiates a fresh 5-angle enrollment session
    public func startEnrollment(userName: String = "User") {
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            self.enrolledUserName = userName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "User" : userName
            self.angleEmbeddings.removeAll()
            self.holdCounter = 0

            DispatchQueue.main.async {
                self.currentAngleIndex = 0
                self.holdProgress = 0.0
                self.isAngleMatched = false
                self.state = .capturing(angle: .frontal, progress: 0.0)
            }

            // Ensure FaceID camera session is active
            FaceIDManager.shared.startCameraForEnrollment()
        }
    }

    /// Cancels active enrollment session
    public func cancelEnrollment() {
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            self.angleEmbeddings.removeAll()
            self.holdCounter = 0

            DispatchQueue.main.async {
                self.state = .idle
                self.holdProgress = 0.0
                self.isAngleMatched = false
            }

            FaceIDManager.shared.stopAuthentication()
        }
    }

    // MARK: - Frame Processing

    /// Process a frame received from FaceID camera delegate
    public func processFrame(face: VNFaceObservation, pixelBuffer: CVPixelBuffer) {
        guard isEnrolling else { return }

        // Extract head pose yaw & pitch in degrees
        //
        // CRITICAL: Apple Vision framework yaw convention:
        //   face.yaw is in RADIANS
        //   Positive yaw = head turned LEFT from user's perspective
        //                 (which is right from the camera's POV)
        //   Negative yaw = head turned RIGHT from user's perspective
        //
        // We convert to degrees and keep Vision's native sign convention.
        let yawRad = face.yaw?.floatValue ?? 0.0
        let pitchRad = face.pitch?.floatValue ?? 0.0
        let yawDeg = yawRad * 180.0 / .pi
        let pitchDeg = pitchRad * 180.0 / .pi

        let angles = EnrollmentAngle.allCases
        guard currentAngleIndex < angles.count else { return }
        let targetAngle = angles[currentAngleIndex]

        // Check if head pose matches target angle within per-angle tolerance
        let yawDiff = abs(yawDeg - targetAngle.targetYawDeg)
        let pitchDiff = abs(pitchDeg - targetAngle.targetPitchDeg)
        let matched = (yawDiff <= targetAngle.yawTolerance) && (pitchDiff <= targetAngle.pitchTolerance)

        DispatchQueue.main.async {
            self.currentYaw = yawDeg
            self.currentPitch = pitchDeg
            self.isAngleMatched = matched
        }

        if matched {
            holdCounter += 1
            let progress = min(1.0, Float(holdCounter) / Float(requiredHoldFrames))

            DispatchQueue.main.async {
                self.holdProgress = progress
                self.state = .capturing(angle: targetAngle, progress: progress)
            }

            if holdCounter >= requiredHoldFrames {
                // Angle capture fulfilled! Extract embedding
                holdCounter = 0
                captureCurrentAngle(face: face, pixelBuffer: pixelBuffer, angle: targetAngle)
            }
        } else {
            // Fast decay when pose drops out (halve counter instead of -1 for snappier feedback)
            holdCounter = max(0, holdCounter - 2)
            let progress = Float(holdCounter) / Float(requiredHoldFrames)
            DispatchQueue.main.async {
                self.holdProgress = progress
                self.state = .capturing(angle: targetAngle, progress: progress)
            }
        }
    }

    // MARK: - Capture & Centroid Computation

    private func captureCurrentAngle(
        face: VNFaceObservation,
        pixelBuffer: CVPixelBuffer,
        angle: EnrollmentAngle
    ) {
        // 1. Align face to canonical 112x112 crop
        let alignedBuffer = FaceAligner.shared.alignFace(from: pixelBuffer, face: face) ?? pixelBuffer

        // 2. Extract 512-D L2-normalized embedding
        let embedding = FaceEmbeddingModel.shared.extractEmbedding(from: alignedBuffer)
        angleEmbeddings.append(embedding)

        print("EnrollmentManager: ✓ Captured angle \(angle.shortTitle) — sample \(angleEmbeddings.count)/5 — yaw=\(String(format: "%.1f°", currentYaw)) pitch=\(String(format: "%.1f°", currentPitch))")

        let nextIndex = currentAngleIndex + 1
        if nextIndex < EnrollmentAngle.allCases.count {
            // Brief pause so user sees the green checkmark before transition
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                self.currentAngleIndex = nextIndex
                self.holdProgress = 0.0
                self.holdCounter = 0
                self.isAngleMatched = false
                let nextAngle = EnrollmentAngle.allCases[nextIndex]
                self.state = .capturing(angle: nextAngle, progress: 0.0)
            }
        } else {
            // All 5 angles captured! Compute centroid and save to Keychain
            finalizeEnrollment()
        }
    }

    private func finalizeEnrollment() {
        DispatchQueue.main.async {
            self.state = .processingCentroid
        }

        guard let centroid = FaceEmbeddingModel.computeCentroid(embeddings: angleEmbeddings) else {
            DispatchQueue.main.async {
                self.state = .failed(reason: "Failed to compute embedding centroid")
            }
            return
        }

        // Store into secure Keychain
        let success = BiometricKeychain.shared.store(
            embedding: centroid,
            userName: enrolledUserName,
            angleEmbeddings: angleEmbeddings
        )

        if success {
            print("EnrollmentManager: ✓ Successfully enrolled biometric template with 5 angles for \(enrolledUserName)")
            DispatchQueue.main.async {
                self.state = .completed(userName: self.enrolledUserName)
                FaceIDManager.shared.reloadEnrollmentStatus()
            }

            // Auto shutoff after showing completion feedback
            DispatchQueue.global().asyncAfter(deadline: .now() + 2.5) { [weak self] in
                self?.cancelEnrollment()
            }
        } else {
            DispatchQueue.main.async {
                self.state = .failed(reason: "Keychain storage failure")
            }
        }
    }
}
