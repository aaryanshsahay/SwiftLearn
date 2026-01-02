// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SwiftLearn",
    products: [
        .library(
            name: "SwiftLearn",
            targets: ["SwiftLearn"]),
        .executable(
            name: "Examples",
            targets: ["Examples"]),
    ],
    targets: [
        .target(
            name: "SwiftLearn"),
        .executableTarget(
            name: "Examples",
            dependencies: ["SwiftLearn"]),
        .testTarget(
            name: "SwiftLearnTests",
            dependencies: ["SwiftLearn"]),
    ]
)
