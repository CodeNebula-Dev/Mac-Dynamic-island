import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import Vision

/// Handles Face Alignment and Affine Similarity Transformation
/// Maps detected facial landmarks to canonical 112x112 coordinates standard for ArcFace / MobileFaceNet,
/// and produces 80x80 square crops for MiniFASNet anti-spoofing.
public final class FaceAligner {
    public static let shared = FaceAligner()

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // Standard InsightFace 112x112 canonical reference points (origin top-left)
    // Converted to bottom-left origin for CoreImage coordinate system (y_ci = 112 - y_top)
    public static let canonicalPoints112: [CGPoint] = [
        CGPoint(x: 38.2946, y: 112.0 - 51.6963), // Left eye center
        CGPoint(x: 73.5318, y: 112.0 - 51.5014), // Right eye center
        CGPoint(x: 56.0252, y: 112.0 - 71.7366), // Nose tip
        CGPoint(x: 41.5493, y: 112.0 - 92.3655), // Left mouth corner
        CGPoint(x: 70.7299, y: 112.0 - 92.2041)  // Right mouth corner
    ]

    private init() {}

    // MARK: - Public API

    /// Aligns a face from a camera frame into a canonical 112x112 CVPixelBuffer
    public func alignFace(
        from pixelBuffer: CVPixelBuffer,
        face: VNFaceObservation
    ) -> CVPixelBuffer? {
        let imageWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let imageHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        guard let detectedPoints = extract5LandmarkPoints(
            face: face,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        ) else {
            // Fallback to bounding box crop if landmarks are unavailable
            return cropBoundingBox(from: pixelBuffer, face: face, targetSize: CGSize(width: 112, height: 112))
        }

        // Compute 2D similarity transform: source -> canonical 112x112
        guard let transform = computeSimilarityTransform(
            sourcePoints: detectedPoints,
            targetPoints: Self.canonicalPoints112
        ) else {
            return cropBoundingBox(from: pixelBuffer, face: face, targetSize: CGSize(width: 112, height: 112))
        }

        let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
        let transformedImage = sourceImage.transformed(by: transform)
        let croppedImage = transformedImage.cropped(to: CGRect(x: 0, y: 0, width: 112, height: 112))

        return renderToPixelBuffer(image: croppedImage, width: 112, height: 112)
    }

    /// Crops the face region scaled for anti-spoofing input (80x80)
    public func cropForAntiSpoof(
        from pixelBuffer: CVPixelBuffer,
        face: VNFaceObservation
    ) -> CVPixelBuffer? {
        return cropBoundingBox(
            from: pixelBuffer,
            face: face,
            targetSize: CGSize(width: 80, height: 80),
            scaleFactor: 1.25
        )
    }

    // MARK: - Landmark Extraction

    /// Extracts 5 key landmark points (left eye, right eye, nose tip, left mouth, right mouth) in pixel coords
    private func extract5LandmarkPoints(
        face: VNFaceObservation,
        imageWidth: CGFloat,
        imageHeight: CGFloat
    ) -> [CGPoint]? {
        guard let landmarks = face.landmarks else { return nil }

        let bbox = face.boundingBox

        // Helper to convert normalized landmark point into pixel coordinates
        func toPixelPoint(_ pt: CGPoint) -> CGPoint {
            let px = (bbox.origin.x + pt.x * bbox.size.width) * imageWidth
            let py = (bbox.origin.y + pt.y * bbox.size.height) * imageHeight
            return CGPoint(x: px, y: py)
        }

        func centerOfPoints(_ points: [CGPoint]) -> CGPoint? {
            guard !points.isEmpty else { return nil }
            let total = points.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            let count = CGFloat(points.count)
            return CGPoint(x: total.x / count, y: total.y / count)
        }

        // 1. Left eye
        let leftEyePt: CGPoint
        if let pupil = landmarks.leftPupil?.normalizedPoints.first {
            leftEyePt = toPixelPoint(pupil)
        } else if let eyePts = landmarks.leftEye?.normalizedPoints, let center = centerOfPoints(eyePts) {
            leftEyePt = toPixelPoint(center)
        } else {
            return nil
        }

        // 2. Right eye
        let rightEyePt: CGPoint
        if let pupil = landmarks.rightPupil?.normalizedPoints.first {
            rightEyePt = toPixelPoint(pupil)
        } else if let eyePts = landmarks.rightEye?.normalizedPoints, let center = centerOfPoints(eyePts) {
            rightEyePt = toPixelPoint(center)
        } else {
            return nil
        }

        // 3. Nose tip
        let nosePt: CGPoint
        if let nosePts = landmarks.nose?.normalizedPoints, !nosePts.isEmpty {
            // Usually center/tip point
            let midIndex = nosePts.count / 2
            nosePt = toPixelPoint(nosePts[midIndex])
        } else if let crestPts = landmarks.noseCrest?.normalizedPoints, let first = crestPts.first {
            nosePt = toPixelPoint(first)
        } else {
            return nil
        }

        // 4. Left mouth corner & 5. Right mouth corner
        let leftMouthPt: CGPoint
        let rightMouthPt: CGPoint
        if let outerLips = landmarks.outerLips?.normalizedPoints, outerLips.count >= 2 {
            // Find leftmost and rightmost points in lip region
            let sortedByX = outerLips.sorted { $0.x < $1.x }
            leftMouthPt = toPixelPoint(sortedByX.first!)
            rightMouthPt = toPixelPoint(sortedByX.last!)
        } else {
            return nil
        }

        return [leftEyePt, rightEyePt, nosePt, leftMouthPt, rightMouthPt]
    }

    // MARK: - Umeyama 2D Similarity Transform

    /// Computes affine similarity transform (scale, rotation, translation) from source points to target points
    public func computeSimilarityTransform(
        sourcePoints: [CGPoint],
        targetPoints: [CGPoint]
    ) -> CGAffineTransform? {
        guard sourcePoints.count == targetPoints.count, sourcePoints.count >= 3 else {
            return nil
        }

        let n = CGFloat(sourcePoints.count)

        // Centroids
        let meanSrc = sourcePoints.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        let srcCenter = CGPoint(x: meanSrc.x / n, y: meanSrc.y / n)

        let meanTgt = targetPoints.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        let tgtCenter = CGPoint(x: meanTgt.x / n, y: meanTgt.y / n)

        // Centered coordinates and variance/covariance
        var srcVar: CGFloat = 0.0
        var sumA: CGFloat = 0.0
        var sumB: CGFloat = 0.0

        for i in 0..<sourcePoints.count {
            let sx = sourcePoints[i].x - srcCenter.x
            let sy = sourcePoints[i].y - srcCenter.y
            let tx = targetPoints[i].x - tgtCenter.x
            let ty = targetPoints[i].y - tgtCenter.y

            srcVar += (sx * sx + sy * sy)
            sumA += (sx * tx + sy * ty)
            sumB += (sx * ty - sy * tx)
        }

        guard srcVar > 1e-6 else { return nil }

        // Scale and rotation
        let scale = sqrt(sumA * sumA + sumB * sumB) / srcVar
        let angle = atan2(sumB, sumA)

        let cosA = cos(angle) * scale
        let sinA = sin(angle) * scale

        // Translation mapping srcCenter to tgtCenter
        let tx = tgtCenter.x - (cosA * srcCenter.x - sinA * srcCenter.y)
        let ty = tgtCenter.y - (sinA * srcCenter.x + cosA * srcCenter.y)

        // CGAffineTransform:
        // [ x']   [ a  c  tx ] [ x ]
        // [ y'] = [ b  d  ty ] [ y ]
        // [ 1 ]   [ 0  0  1  ] [ 1 ]
        // Where a = cosA, b = sinA, c = -sinA, d = cosA
        return CGAffineTransform(a: cosA, b: sinA, c: -sinA, d: cosA, tx: tx, ty: ty)
    }

    // MARK: - Cropping & PixelBuffer Rendering

    /// Simple bounding-box crop with aspect scaling
    private func cropBoundingBox(
        from pixelBuffer: CVPixelBuffer,
        face: VNFaceObservation,
        targetSize: CGSize,
        scaleFactor: CGFloat = 1.1
    ) -> CVPixelBuffer? {
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        let bbox = face.boundingBox
        var cropRect = CGRect(
            x: bbox.origin.x * width,
            y: bbox.origin.y * height,
            width: bbox.size.width * width,
            height: bbox.size.height * height
        )

        // Inflate by scaleFactor around center
        let dx = (cropRect.width * (scaleFactor - 1.0)) / 2.0
        let dy = (cropRect.height * (scaleFactor - 1.0)) / 2.0
        cropRect = cropRect.insetBy(dx: -dx, dy: -dy)

        // Clamp to image bounds
        cropRect = cropRect.intersection(CGRect(x: 0, y: 0, width: width, height: height))
        guard cropRect.width > 10, cropRect.height > 10 else { return nil }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer).cropped(to: cropRect)

        // Scale to targetSize
        let scaleX = targetSize.width / cropRect.width
        let scaleY = targetSize.height / cropRect.height
        let scaledImage = ciImage
            .transformed(by: CGAffineTransform(translationX: -cropRect.origin.x, y: -cropRect.origin.y))
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        return renderToPixelBuffer(image: scaledImage, width: Int(targetSize.width), height: Int(targetSize.height))
    }

    private func renderToPixelBuffer(
        image: CIImage,
        width: Int,
        height: Int
    ) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        ciContext.render(image, to: buffer, bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: CGColorSpaceCreateDeviceRGB())
        return buffer
    }
}
