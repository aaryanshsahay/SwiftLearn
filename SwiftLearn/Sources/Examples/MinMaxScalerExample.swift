import SwiftLearn

func runMinMaxScalerExample() throws {
    print("=== MinMaxScaler Example ===\n")

    // Example from sklearn docs
    let X = Matrix([
        [-1.0, 2.0],
        [-0.5, 6.0],
        [0.0, 10.0],
        [1.0, 18.0]
    ])

    // Default range [0, 1]
    var scaler = MinMaxScaler()
    let scaled = try scaler.fitTransform(X)

    print(scaler)
    // MinMaxScaler(featureRange: (0.0, 1.0), clip: false, nFeatures: 2)

    print("Data max:", scaler.dataMax_!)
    // Data max: [1.0, 18.0]

    print("Data min:", scaler.dataMin_!)
    // Data min: [-1.0, 2.0]

    print(scaled.prettyPrint)
    // Matrix(4x2):
    //   [0.0000, 0.0000]
    //   [0.2500, 0.2500]
    //   [0.5000, 0.5000]
    //   [1.0000, 1.0000]

    // Transform new data (can exceed [0, 1] range)
    let newData = Matrix([[2.0, 2.0]])
    let newScaled = try scaler.transform(newData)
    print("New scaled:")
    print(newScaled.prettyPrint)
    // Matrix(1x2):
    //   [1.5000, 0.0000]

    // Custom range [-1, 1]
    var scaler2 = MinMaxScaler(featureRange: (-1.0, 1.0))
    let scaled2 = try scaler2.fitTransform(X)
    print("Custom range [-1, 1]:")
    print(scaled2.prettyPrint)
    // Matrix(4x2):
    //   [-1.0000, -1.0000]
    //   [-0.5000, -0.5000]
    //   [0.0000, 0.0000]
    //   [1.0000, 1.0000]

    // With clipping
    var scaler3 = MinMaxScaler(clip: true)
    _ = try scaler3.fitTransform(X)
    let clipped = try scaler3.transform(Matrix([[5.0, 50.0]]))
    print("With clipping (values clipped to [0, 1]):")
    print(clipped.prettyPrint)
    // Matrix(1x2):
    //   [1.0000, 1.0000]

    // --- Load from JSON example ---
    print("=== Loading from JSON ===\n")

    // Simulate JSON data (as if exported from Python)
    let jsonString = """
    {
        "data_min_": [-1.0, 2.0],
        "data_max_": [1.0, 18.0],
        "feature_range_min": 0.0,
        "feature_range_max": 1.0,
        "clip": false
    }
    """

    let jsonData = jsonString.data(using: .utf8)!
    let loadedScaler = try MinMaxScaler.load(from: jsonData)

    print("Loaded scaler:", loadedScaler)
    print("Loaded data_min_:", loadedScaler.dataMin_!)
    print("Loaded data_max_:", loadedScaler.dataMax_!)

    // Transform using loaded scaler
    let testData = Matrix([[0.0, 10.0]])
    let testScaled = try loadedScaler.transform(testData)
    print("Transform with loaded scaler:")
    print(testScaled.prettyPrint)
    // Should output [0.5, 0.5]

    print("")
}
