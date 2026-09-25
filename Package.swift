// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MacDynamicIsland",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "MacDynamicIsland",
            targets: ["MacDynamicIsland"]
        )
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "MacDynamicIsland",
            dependencies: [],
            path: "Sources/MacDynamicIsland"
        )
    ]
)
