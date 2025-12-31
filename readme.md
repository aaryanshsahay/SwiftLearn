# SwiftLearn

A native Swift machine learning library inspired by scikit-learn. Built for performance using Apple's Accelerate framework.

## Features

- **Familiar API**: Follows scikit-learn's `fit`/`transform`/`predict` patterns
- **Native Performance**: Uses Apple's Accelerate framework (vDSP) for vectorized operations
- **Type Safe**: Leverages Swift's type system for compile-time safety
- **Codable**: All models support serialization for saving/loading

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/aaryanshsahay/SwiftLearn.git", branch: "dev")
]
```

Or in Xcode: File > Add Package Dependencies > Enter the repository URL.

## Quick Start

```swift
import SwiftLearn

// Create sample data
let X = Matrix([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 1.0],
    [1.0, 1.0]
])

// Standardize features (zero mean, unit variance)
var scaler = StandardScaler()
let scaled = try scaler.fitTransform(X)

print(scaler.mean_!)   // [0.5, 0.5]
print(scaler.scale_!)  // [0.5, 0.5]
print(scaled.prettyPrint)
// Matrix(4x2):
//   [-1.0000, -1.0000]
//   [-1.0000, -1.0000]
//   [1.0000, 1.0000]
//   [1.0000, 1.0000]

// Transform new data
let newData = Matrix([[2.0, 2.0]])
let newScaled = try scaler.transform(newData)

// Inverse transform to original scale
let restored = try scaler.inverseTransform(scaled)
```

## Available Components

### Preprocessing

| Class | Description |
|-------|-------------|
| `StandardScaler` | Standardize features by removing mean and scaling to unit variance |

### Core Types

| Type | Description |
|------|-------------|
| `Matrix` | 2D array with row-major storage, supports subscripting and column access |

## API Design

SwiftLearn follows scikit-learn conventions:

- **`fit(_:)`** - Learn parameters from training data
- **`transform(_:)`** - Apply learned transformation
- **`fitTransform(_:)`** - Fit and transform in one step
- **`inverseTransform(_:)`** - Reverse the transformation

Learned parameters use trailing underscores (e.g., `mean_`, `scale_`) to distinguish them from configuration parameters.

## Requirements

- Swift 5.9+
- macOS 10.15+ / iOS 13+

## Roadmap

- [ ] MinMaxScaler
- [ ] LabelEncoder
- [ ] Train/Test Split
- [ ] Linear Regression
- [ ] Logistic Regression
- [ ] K-Nearest Neighbors
- [ ] Decision Trees
- [ ] Cross Validation

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details.
