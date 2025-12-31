# SwiftLearn

A native Swift inference library for scikit-learn models. Train your models in Python, run predictions natively in Swift.

## Why SwiftLearn?

- **Train in Python, Deploy in Swift**: Use familiar sklearn workflows, then run inference on Apple platforms without Python dependencies
- **Native Performance**: Built on Apple's Accelerate framework (vDSP) for vectorized operations
- **No Runtime Dependencies**: Pure Swift - no Python bridge, no CoreML conversion
- **Type Safe**: Leverages Swift's type system for compile-time safety

## How It Works

1. **Train** your model in Python using scikit-learn
2. **Export** model weights to JSON/H5
3. **Load** weights in Swift and run `predict`/`predict_proba`

```python
# Python: Train and export
from sklearn.preprocessing import StandardScaler
import json

scaler = StandardScaler()
scaler.fit(X_train)

# Export weights
weights = {
    "mean_": scaler.mean_.tolist(),
    "scale_": scaler.scale_.tolist()
}
json.dump(weights, open("scaler.json", "w"))
```

```swift
// Swift: Load and predict
import SwiftLearn

let scaler = try StandardScaler.load(from: "scaler.json")
let predictions = try scaler.transform(newData)
```

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

// Create data
let X = Matrix([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 1.0],
    [1.0, 1.0]
])

// StandardScaler example
var scaler = StandardScaler()
let scaled = try scaler.fitTransform(X)

print(scaler.mean_!)   // [0.5, 0.5]
print(scaler.scale_!)  // [0.5, 0.5]

// Transform new data
let newData = Matrix([[2.0, 2.0]])
let result = try scaler.transform(newData)
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

## Roadmap

### Preprocessing
- [x] StandardScaler
- [ ] MinMaxScaler
- [ ] LabelEncoder
- [ ] OneHotEncoder

### Model Loading
- [ ] Load model weights from JSON
- [ ] Load model weights from H5

### Inference
- [ ] Linear Regression (`predict`)
- [ ] Logistic Regression (`predict`, `predict_proba`)
- [ ] K-Nearest Neighbors (`predict`, `predict_proba`)
- [ ] Decision Trees (`predict`, `predict_proba`)
- [ ] Random Forest (`predict`, `predict_proba`)
- [ ] SVM (`predict`, `decision_function`)

### Utilities
- [ ] Train/Test Split
- [ ] Accuracy, Precision, Recall metrics

## Requirements

- Swift 5.9+
- macOS 10.15+ / iOS 13+ / tvOS 13+ / watchOS 6+

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details.
