import SwiftLearn

// Create sample data
let X = Matrix([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 1.0],
    [1.0, 1.0]
])

// Create & fit scaler
var scaler = StandardScaler()
let scaled = try scaler.fitTransform(X)

print(scaler)
// StandardScaler(withMean: true, withStd: true, nFeatures: 2)

print("Mean:", scaler.mean_!)
// Mean: [0.5, 0.5]

print("Scale:", scaler.scale_!)
// Scale: [0.5, 0.5]

print(scaled.prettyPrint)
// Matrix(4x2):
// [-1.0000, -1.0000]
// [-1.0000, -1.0000]
// [1.0000, 1.0000]
// [1.0000, 1.0000]

// Transform new data
let newData = Matrix([[2.0, 2.0]])
let newScaled = try scaler.transform(newData)
print("New data scaled:")
print(newScaled.prettyPrint)
// Matrix(1x2):
// [3.0000, 3.0000]

// Inverse transform
let restored = try scaler.inverseTransform(scaled)
print("Restored original data:")
print(restored.prettyPrint)
// Gets back original data
