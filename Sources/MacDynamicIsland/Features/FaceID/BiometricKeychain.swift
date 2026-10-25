import Foundation
import Security

/// Represents the securely enrolled face biometric template
public struct EnrolledFaceTemplate: Codable {
    public let userName: String
    public let enrollmentDate: Date
    public let embedding: [Float] // 512-D L2-normalized centroid vector
    public let angleEmbeddings: [[Float]] // Gallery of per-angle embeddings (5 angles)
    public let version: Int

    public init(
        userName: String,
        enrollmentDate: Date = Date(),
        embedding: [Float],
        angleEmbeddings: [[Float]] = [],
        version: Int = 2
    ) {
        self.userName = userName
        self.enrollmentDate = enrollmentDate
        self.embedding = embedding
        self.angleEmbeddings = angleEmbeddings
        self.version = version
    }
}

/// Secure biometric storage utilizing macOS Keychain Services
public final class BiometricKeychain {
    public static let shared = BiometricKeychain()

    private let serviceName = "com.codenebula.macdynamicisland.faceid"
    private let accountName = "enrolled_face_template"

    private init() {}

    // MARK: - Status

    /// Returns true if a face template is currently enrolled in Keychain
    public var isEnrolled: Bool {
        return retrieveTemplate() != nil
    }

    /// Returns enrolled user's display name if enrolled
    public var enrolledUserName: String? {
        return retrieveTemplate()?.userName
    }

    // MARK: - Storage & Retrieval

    /// Encrypts and securely stores an enrolled face template into the macOS Keychain
    @discardableResult
    public func store(template: EnrolledFaceTemplate) -> Bool {
        guard let data = try? JSONEncoder().encode(template) else {
            print("BiometricKeychain: Failed to encode template data")
            return false
        }

        // Delete any existing item first to ensure clean state
        deleteTemplate()

        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: serviceName,
            kSecAttrAccount: accountName,
            kSecAttrAccessible: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            kSecValueData: data
        ]

        let status = SecItemAdd(query as CFDictionary, nil)
        if status == errSecSuccess {
            print("BiometricKeychain: Successfully saved enrolled face template for \(template.userName)")
            return true
        } else {
            print("BiometricKeychain: SecItemAdd failed with OSStatus \(status)")
            return false
        }
    }

    /// Convenience to store centroid embedding and optional per-angle gallery
    @discardableResult
    public func store(
        embedding: [Float],
        userName: String,
        angleEmbeddings: [[Float]] = []
    ) -> Bool {
        let template = EnrolledFaceTemplate(
            userName: userName,
            embedding: embedding,
            angleEmbeddings: angleEmbeddings
        )
        return store(template: template)
    }

    /// Retrieves the enrolled face template from macOS Keychain
    public func retrieveTemplate() -> EnrolledFaceTemplate? {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: serviceName,
            kSecAttrAccount: accountName,
            kSecReturnData: true,
            kSecMatchLimit: kSecMatchLimitOne
        ]

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)

        guard status == errSecSuccess, let data = item as? Data else {
            return nil
        }

        return try? JSONDecoder().decode(EnrolledFaceTemplate.self, from: data)
    }

    /// Retrieves just the 512-D centroid embedding
    public func retrieveEmbedding() -> [Float]? {
        return retrieveTemplate()?.embedding
    }

    /// Deletes the enrolled template from macOS Keychain
    @discardableResult
    public func deleteTemplate() -> Bool {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: serviceName,
            kSecAttrAccount: accountName
        ]

        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }
}
