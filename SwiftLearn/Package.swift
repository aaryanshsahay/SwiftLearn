// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SwiftLearn",
    products: [
        .library(
            name: "SwiftLearn",
            targets: ["SwiftLearn"]),
        .executable(
            name: "SwiftLearnDemo",
            targets: ["SwiftLearnDemo"]),
    ],
    targets: [
        .target(
            name: "SwiftLearn"),
        .executableTarget(
            name: "SwiftLearnDemo",
            dependencies: ["SwiftLearn"]),
        .testTarget(
            name: "SwiftLearnTests",
            dependencies: ["SwiftLearn"]),
    ]
)
