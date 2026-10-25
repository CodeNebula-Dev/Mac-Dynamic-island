import Foundation
import AVFoundation
import Vision
import CoreImage
import Combine

/// Manages Computer Vision-based Face Recognition & Liveness for macOS ("Face ID for Mac")
public final class FaceIDManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    public static let shared = FaceIDManager()

    public enum AuthState: Equatable {
        case idle
        case detecting
        case verifying(similarity: Float)
        case authorized(user: String, similarity: Float)
        case unauthorized(reason: String)
    }

    @Published public private(set) var authState: AuthState = .idle
    @Published public private(set) var isCameraActive: Bool = false
    @Published public private(set) var detectedFaceCount: Int = 0
    @Published public private(set) var livenessScore: Float = 0.0
    @Published public private(set) var lastSimilarity: Float = 0.0
    @Published public private(set) var isEnrolled: Bool = false
    @Published public private(set) var enrolledUserName: String? = nil

    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.codenebula.macdynamicisland.camera", qos: .userInitiated)
    private var isConfigured = false

    // Vision Requests
    private lazy var faceDetectionRequest: VNDetectFaceLandmarksRequest = {
        let request = VNDetectFaceLandmarksRequest { [weak self] request, error in
            self?.handleFaceLandmarks(request: request, error: error)
        }
        return request
    }()

    // Temporal tracking for Anti-Spoofing & Liveness (Blink & Micro-motion)
    private var earHistory: [Float] = []
    private var latestPixelBuffer: CVPixelBuffer?
    private let pixelBufferLock = NSLock()

    public override init() {
        super.init()
        reloadEnrollmentStatus()
    }

    // MARK: - Enrollment Sync

    public func reloadEnrollmentStatus() {
        let enrolled = BiometricKeychain.shared.isEnrolled
        let name = BiometricKeychain.shared.enrolledUserName
        DispatchQueue.main.async {
            self.isEnrolled = enrolled
            self.enrolledUserName = name
        }
    }

    // MARK: - Public Control

    /// Initiates a Face ID authentication cycle
    public func startAuthentication(completion: ((Bool, String?) -> Void)? = nil) {
        reloadEnrollmentStatus()
        startCameraSession { [weak self] in
            DispatchQueue.main.async {
                self?.authState = .detecting
            }
        }
    }

    /// Initiates camera session specifically for enrollment flow
    public func startCameraForEnrollment() {
        startCameraSession(onStarted: nil)
    }

    /// Stops camera and returns to idle to preserve battery life
    public func stopAuthentication() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
                DispatchQueue.main.async {
                    self.isCameraActive = false
                    self.authState = .idle
                }
            }
        }
    }

    private func startCameraSession(onStarted: (() -> Void)?) {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self = self, granted else {
                DispatchQueue.main.async {
                    self?.authState = .unauthorized(reason: "Camera permission denied")
                }
                return
            }

            self.sessionQueue.async {
                self.setupCameraIfNeeded()
                if !self.captureSession.isRunning {
                    self.captureSession.startRunning()
                    DispatchQueue.main.async {
                        self.isCameraActive = true
                        onStarted?()
                    }
                } else {
                    DispatchQueue.main.async {
                        onStarted?()
                    }
                }
            }
        }
    }

    // MARK: - Camera Setup

    private func setupCameraIfNeeded() {
        guard !isConfigured else { return }
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .high

        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .externalUnknown],
            mediaType: .video,
            position: .front
        )

        guard let device = discoverySession.devices.first,
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input) else {
            captureSession.commitConfiguration()
            return
        }

        captureSession.addInput(input)

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)

        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }

        captureSession.commitConfiguration()
        isConfigured = true
    }

    // MARK: - Video Output Delegate

    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        pixelBufferLock.lock()
        latestPixelBuffer = pixelBuffer
        pixelBufferLock.unlock()

        let imageRequestHandler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )

        do {
            try imageRequestHandler.perform([faceDetectionRequest])
        } catch {
            print("Vision request failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Vision Processing

    private func handleFaceLandmarks(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNFaceObservation], !results.isEmpty else {
            DispatchQueue.main.async {
                self.detectedFaceCount = 0
            }
            return
        }

        DispatchQueue.main.async {
            self.detectedFaceCount = results.count
        }

        guard let primaryFace = results.first else { return }

        pixelBufferLock.lock()
        let currentBuffer = latestPixelBuffer
        pixelBufferLock.unlock()

        guard let buffer = currentBuffer else { return }

        // Route to EnrollmentManager if enrollment is currently active
        if EnrollmentManager.shared.isEnrolling {
            EnrollmentManager.shared.processFrame(face: primaryFace, pixelBuffer: buffer)
            return
        }

        // Otherwise process live authentication
        processFaceLivenessAndIdentity(face: primaryFace, pixelBuffer: buffer)
    }

    /// Evaluates Eye Aspect Ratio (EAR), Anti-Spoofing, and ArcFace 512-D Cosine Matching
    private func processFaceLivenessAndIdentity(face: VNFaceObservation, pixelBuffer: CVPixelBuffer) {
        guard let landmarks = face.landmarks else { return }

        // Step 1: Check enrollment status
        guard let enrolledTemplate = BiometricKeychain.shared.retrieveEmbedding() else {
            DispatchQueue.main.async {
                self.authState = .unauthorized(reason: "Face ID not enrolled")
            }
            return
        }

        // Step 2: Temporal EAR blink tracking
        if let leftEye = landmarks.leftEye, let rightEye = landmarks.rightEye {
            let leftEAR = computeEyeAspectRatio(points: leftEye.normalizedPoints)
            let rightEAR = computeEyeAspectRatio(points: rightEye.normalizedPoints)
            let averageEAR = (leftEAR + rightEAR) / 2.0

            earHistory.append(averageEAR)
            if earHistory.count > 15 {
                earHistory.removeFirst()
            }
        }

        let isLiveHeuristic = evaluateLiveness()
        guard isLiveHeuristic else {
            DispatchQueue.main.async {
                self.authState = .unauthorized(reason: "No micro-motion detected — photo suspected")
            }
            return
        }

        // Step 3: Anti-spoof inference (MiniFASNet)
        if let antiSpoofCrop = FaceAligner.shared.cropForAntiSpoof(from: pixelBuffer, face: face) {
            let spoofScore = FaceEmbeddingModel.shared.checkLiveness(from: antiSpoofCrop)
            DispatchQueue.main.async {
                self.livenessScore = spoofScore
            }

            guard spoofScore >= FaceEmbeddingModel.livenessThreshold else {
                DispatchQueue.main.async {
                    self.authState = .unauthorized(
                        reason: spoofScore < 0.45 ? "Screen/Photo Detected" : "Anti-spoof check failed"
                    )
                }
                return
            }
        }

        // Step 4: Face Alignment to canonical 112x112 crop
        guard let alignedBuffer = FaceAligner.shared.alignFace(from: pixelBuffer, face: face) else {
            return
        }

        // Step 5: Extract 512-D CoreML embedding vector
        let liveEmbedding = FaceEmbeddingModel.shared.extractEmbedding(from: alignedBuffer)

        // Step 6: Cosine Similarity Matching
        let similarity = FaceEmbeddingModel.cosineSimilarity(u: liveEmbedding, v: enrolledTemplate)

        DispatchQueue.main.async {
            self.lastSimilarity = similarity

            if similarity >= FaceEmbeddingModel.matchThreshold {
                let name = self.enrolledUserName ?? "User"
                self.authState = .authorized(user: name, similarity: similarity)

                // Auto-shutoff camera after authorized
                DispatchQueue.global().asyncAfter(deadline: .now() + 1.8) { [weak self] in
                    self?.stopAuthentication()
                }
            } else {
                self.authState = .unauthorized(
                    reason: String(format: "Identity not recognized (%.0f%% match)", similarity * 100)
                )
            }
        }
    }

    private func computeEyeAspectRatio(points: [CGPoint]) -> Float {
        guard points.count >= 6 else { return 0.3 }
        let p1 = points[1], p5 = points[5]
        let p2 = points[2], p4 = points[4]
        let p0 = points[0], p3 = points[3]

        let vertical1 = hypot(p1.x - p5.x, p1.y - p5.y)
        let vertical2 = hypot(p2.x - p4.x, p2.y - p4.y)
        let horizontal = hypot(p0.x - p3.x, p0.y - p3.y)

        guard horizontal > 0 else { return 0.3 }
        return Float((vertical1 + vertical2) / (2.0 * horizontal))
    }

    private func evaluateLiveness() -> Bool {
        guard earHistory.count >= 5 else { return true }
        let mean = earHistory.reduce(0, +) / Float(earHistory.count)
        let variance = earHistory.map { pow($0 - mean, 2) }.reduce(0, +) / Float(earHistory.count)
        return variance >= 0.00008
    }
}
