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
            case .frontal: return "Center your face in the guide"
            case .left: return "Turn slightly to your left (~15°)"
            case .right: return "Turn slightly to your right (~15°)"
            case .up: return "Tilt your chin slightly up (~10°)"
            case .down: return "Tilt your chin slightly down (~10°)"
            }
        }

        public var targetYawDeg: Float {
            switch self {
            case .frontal: return 0.0
            case .left: return -15.0
            case .right: return 15.0
            case .up: return 0.0
            case .down: return 0.0
            }
        }

        public var targetPitchDeg: Float {
            switch self {
            case .frontal: return 0.0
            case .left: return 0.0
            case .right: return 0.0
            case .up: return 10.0
            case .down: return -10.0
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
    private let requiredHoldFrames: Int = 12 // ~0.4s at 30fps steady pose
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
        // Vision yaw: radians (positive = right, negative = left)
        // Vision pitch: radians (positive = up, negative = down)
        let yawDeg = face.yaw.map { Float(truncating: $0) * 180.0 / .pi } ?? 0.0
        let pitchDeg = face.pitch.map { Float(truncating: $0) * 180.0 / .pi } ?? 0.0

        let angles = EnrollmentAngle.allCases
        guard currentAngleIndex < angles.count else { return }
        let targetAngle = angles[currentAngleIndex]

        // Check if head pose matches target angle within tolerance
        let yawDiff = abs(yawDeg - targetAngle.targetYawDeg)
        let pitchDiff = abs(pitchDeg - targetAngle.targetPitchDeg)
        let matched = (yawDiff <= 8.5) && (pitchDiff <= 7.5)

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
            // Gradually decay hold progress if pose drops out
            if holdCounter > 0 {
                holdCounter -= 1
                let progress = Float(holdCounter) / Float(requiredHoldFrames)
                DispatchQueue.main.async {
                    self.holdProgress = progress
                    self.state = .capturing(angle: targetAngle, progress: progress)
                }
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

        print("EnrollmentManager: Captured angle \(angle.shortTitle) (sample \(angleEmbeddings.count)/5)")

        let nextIndex = currentAngleIndex + 1
        if nextIndex < EnrollmentAngle.allCases.count {
            // Move to next angle
            DispatchQueue.main.async {
                self.currentAngleIndex = nextIndex
                self.holdProgress = 0.0
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
            print("EnrollmentManager: Successfully enrolled template with 5 angles for \(enrolledUserName)")
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
