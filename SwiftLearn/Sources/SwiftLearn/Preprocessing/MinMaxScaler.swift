// MinMaxScaler.swift
// Author: AaryanshSahay
// 
// Transform features by scaling each feature to a given range

import Foundation
import Accelerate 

/// Transform features by scaling each feature to a given range
///
/// This estimator scales and translates each feature individually so 
/// that it is in the given range on the training set, e.g. between zero and one
///
/// This transformation is achieved by:
///
///     X_std = (X - X.min)/ (X.max - X.min)
///     X_scaled = X_std * (max - min) + min
///
/// -> where `min, max = featureRange`
///
/// ## Example
/// ```Swift
/// var scaler = MinMaxScaler()
/// let X = Matrix([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
/// let scaled = try scaler.fitTransform(X)
/// // scaled values will be in range [0,1]
/// ````
///
/// ## Notes
/// - Features with zero range (constant) are scaled to `featureRange.min`
/// - Use `clip: true` to ensure transformed values stay within the feature range
public struct MinMaxScaler {
    // MARK: - Configuration Parameters
    
    /// desired range of transformed data (min, max). Default is (0,1)
    public var featureRange: (min: Double, max: Double)

    /// if `true`, clip transformed values to feature range
    public var clip: Bool

    // MARK: - Learned Attributes (set after fit)
    
    /// Per-feature minimum seen in the training data.
    public private(set) var dataMin_: [Double]?

    /// Per-feature maximum seen in the training data
    public private(set) var dataMax_: [Double]?

    /// Per-feature range `(dataMax_ - dataMin_)` seen in the training data
    public private(set) var dataRange_: [Double]?

    /// Per-feature relative scaling factor
    /// equivalent to `(featureRange.max - featureRange.min)/ dataRange_`
    public private(set) var scale_: [Double]?

    /// Per-feature adjustment for minimum
    /// equivalent to `(featureRange.min - dataMin_ * scale_)`
    public private(set) var min_: [Double]?

    /// Number of features seen during `fit`
    public private(set) var nFeaturesIn_: Int?

    /// Number of samples seen during `fit`
    public private(set) var nSamplesSeen_: Int?

    // MARK: - Computed Properties

    /// Returns `true` if the scaler has been fitted.
    public var isFitted: Bool {
        return nFeaturesIn_ != nil
    } 

    // MARK: - Initialization

    /// Creates a new MinMaxScaler
    ///
    /// - Parameters:
    ///     -> featureRange: Desired range of transformed data (min, max). Default is (0, 1).
    ///     -> clip: If `true`, clip transformed values to feature range. Default is `false`.
    public init(featureRange: (min: Double, max: Double) = (0.0, 1.0), clip: Bool = false) {
        self.featureRange = featureRange
        self.clip = clip
    }

    // MARK: - Load from File

    /// Load a MinMaxScaler from a JSON file exported from Python sklearn.
    ///
    /// Expected JSON format:
    /// ```json
    /// {
    ///     "data_min_": [-1.0, 2.0],
    ///     "data_max_": [1.0, 18.0],
    ///     "feature_range_min": 0.0,
    ///     "feature_range_max": 1.0,
    ///     "clip": false
    /// }
    /// ```
    ///
    /// Python export example:
    /// ```python
    /// from sklearn.preprocessing import MinMaxScaler
    /// import json
    ///
    /// scaler = MinMaxScaler(feature_range=(0, 1))
    /// scaler.fit(X_train)
    ///
    /// weights = {
    ///     "data_min_": scaler.data_min_.tolist(),
    ///     "data_max_": scaler.data_max_.tolist(),
    ///     "feature_range_min": scaler.feature_range[0],
    ///     "feature_range_max": scaler.feature_range[1],
    ///     "clip": False
    /// }
    /// json.dump(weights, open("minmax_scaler.json", "w"))
    /// ```
    ///
    /// - Parameter path: Path to the JSON file
    /// - Returns: A fitted MinMaxScaler ready for transform
    /// - Throws: `SwiftLearnError.serializationError` if file cannot be read or parsed
    public static func load(from path: String) throws -> MinMaxScaler {
        let url = URL(fileURLWithPath: path)

        guard let data = try? Data(contentsOf: url) else {
            throw SwiftLearnError.serializationError("Could not read file at path: \(path)")
        }

        return try load(from: data)
    }

    /// Load a MinMaxScaler from JSON Data exported from Python sklearn.
    ///
    /// - Parameter data: JSON data
    /// - Returns: A fitted MinMaxScaler ready for transform
    /// - Throws: `SwiftLearnError.serializationError` if data cannot be parsed
    public static func load(from data: Data) throws -> MinMaxScaler {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SwiftLearnError.serializationError("Invalid JSON format")
        }

        guard let dataMin = json["data_min_"] as? [Double] else {
            throw SwiftLearnError.serializationError("Missing or invalid 'data_min_' array in JSON")
        }

        guard let dataMax = json["data_max_"] as? [Double] else {
            throw SwiftLearnError.serializationError("Missing or invalid 'data_max_' array in JSON")
        }

        guard dataMin.count == dataMax.count else {
            throw SwiftLearnError.serializationError("data_min_ and data_max_ arrays must have the same length")
        }

        let featureRangeMin = json["feature_range_min"] as? Double ?? 0.0
        let featureRangeMax = json["feature_range_max"] as? Double ?? 1.0
        let clip = json["clip"] as? Bool ?? false

        var scaler = MinMaxScaler(featureRange: (featureRangeMin, featureRangeMax), clip: clip)
        scaler.dataMin_ = dataMin
        scaler.dataMax_ = dataMax
        scaler.nFeaturesIn_ = dataMin.count

        // Compute derived values
        scaler.dataRange_ = zip(dataMin, dataMax).map { $1 - $0 }

        let featureRangeSpan = featureRangeMax - featureRangeMin
        scaler.scale_ = scaler.dataRange_!.map { range in
            if range < 1e-10 {
                return 1.0
            }
            return featureRangeSpan / range
        }

        scaler.min_ = zip(dataMin, scaler.scale_!).map { dataMin, scale in
            featureRangeMin - dataMin * scale
        }

        return scaler
    }

    // MARK: - Fit Methods

    /// Compute the minimum and maximum to be used for later scaling.
    ///
    /// - Parameters:
    ///     -> X: Training data of shape (nSamples, nFeatures)
    ///
    /// - Throws:
    ///     -> `SwiftLearnError.invalidInput` if the input is empty or feature range is invalid.
    public mutating func fit(_ X: Matrix) throws {
        try _reset()
        try _fit(X)
    }

    /// Compute the minimum and maximum, then transform the data.
    
    /// Parameters:
    ///     -> X: Training data of shape (nSamples, nFeatures)
    ///
    /// Returns:
    ///     -> Transformed data scaled to feature range
    ///
    /// Throws: 
    ///     -> `SwiftLearnError.invalidInput` if the input is empty
    public mutating func fitTransform(_ X: Matrix) throws -> Matrix {
        try fit(X)
        return try transform(X)
    }

    // MARK: - Transform methods

    /// Scale features of X according to feature range
    ///
    /// Parameters:
    ///     -> X: Data to transform of shape (nSamples, nFeatures)
    /// 
    /// Returns:
    ///     -> Transformed data
    ///
    /// Throws:
    ///     -> `SwiftLearnError.notFitted` if `fit` has not been called.
    ///     -> `SwiftLearnError.dimensionMismatch` if number of features don't match training data.
    public func transform(_ X: Matrix) throws -> Matrix {
        guard isFitted else {
            throw SwiftLearnError.notFitted(
                "MinMaxScaler has not been fitted. Call fit() first."
            )
        }

        guard X.cols == nFeaturesIn_! else {
            throw SwiftLearnError.dimensionMismatch(
                expected: nFeaturesIn_!,
                got: X.cols
            )
        }

        guard let scale = scale_, let minAdj = min_ else {
            throw SwiftLearnError.notFitted(
                "Scaler parameters not set."
            )
        }

        var result = X

        // Transform: X_scaled = X * scaled + min_adjustment
        for i in 0..<result.rows {
            for j in 0..<result.cols {
                result[i,j] = result[i,j] * scale[j] + minAdj[j]
            }
        }

        // clip to feature range if requested.

        if clip {
            for i in 0..<result.rows{
                for j in 0..<result.cols {
                    result[i,j] = Swift.max(featureRange.min, Swift.min(featureRange.max, result[i,j]))
                }
            }
        }
        return result
    }
    
    /// Undo the scaling of X according to feature range.
    ///
    /// Parameters:
    ///     -> X: Transformed data of shape (nSamples, nFeatures)
    ///
    /// Returns:
    ///     -> Data in original scale
    ///
    /// Throws:
    ///     -> `SwiftLearnError.notFitted` if `fit` has not been called.
    ///     -> `SwiftLearnError.dimensionMismatch` if number of featuers doesn't match training data
    public func inverseTransform(_ X: Matrix) throws -> Matrix {
        guard isFitted else {
            throw SwiftLearnError.notFitted(
                "MinMaxScaler has not been fitted. Call fit() first."
            )
        }

        guard X.cols == nFeaturesIn_! else {
            throw SwiftLearnError.dimensionMismatch(
                expected: nFeaturesIn_!,
                got: X.cols
            )
        }

        guard let scale = scale_, let minAdj = min_ else {
            throw SwiftLearnError.notFitted(
                "Scaler parameters not set."
            )
        }

        var result = X

        // Inverse transform: X_original = (X_scaled - min_adjustment) / scale
        for i in 0..<result.rows {
            for j in 0..<result.cols {
                result[i,j] = (result[i,j] - minAdj[j])/scale[j]
            }
        }
        return result
    }

    // MARK: - Private Methods

    /// Reset internal state before fitting
    private mutating func _reset() throws {
        dataMin_ = nil
        dataMax_ = nil
        dataRange_ = nil
        scale_ = nil
        min_ = nil
        nFeaturesIn_ = nil
        nSamplesSeen_ = nil
    }

    /// Internal fit implementation.
    private mutating func _fit(_ X: Matrix) throws {
        // Validate feature range
        guard featureRange.min < featureRange.max else {
            throw SwiftLearnError.invalidInput(
                "Minimum of desired feature range must be smaller than maximum. " + 
                "Got (\(featureRange.min), \(featureRange.max))."
            )
        }

        // validate input
        guard X.rows > 0 else {
            throw SwiftLearnError.invalidInput(
                "Input matrix has no samples (rows)"
            )
        }

        guard X.cols > 0 else {
            throw SwiftLearnError.invalidInput(
                "Input matrix has no features (columns)"
            )
        }

        nFeaturesIn_ = X.cols
        nSamplesSeen_ = X.rows

        // compute min and max for each feature (column)
        dataMin_ = (0..<X.cols).map { col in
            _columnMin(X, column: col)
        }

        dataMax_ = (0..<X.cols).map { col in
            _columnMax(X, column: col)
        }

        // compute data range

        dataRange_ = zip(dataMin_!, dataMax_!).map { minVal, maxVal in
            maxVal - minVal
        }

        // Compute scale: (feature_max - feature_min)/ data_range
        // Handle zero range (constant features) by setting scale to 1.0
        let featureRangeSpan = featureRange.max - featureRange.min
        scale_ = dataRange_!.map { range in
            if range < 1e-10 {
                // constant feature - avoid division by zero
                return 1.0
            }
            return featureRangeSpan / range
        }

        // compute min adjustment: feature_min - data_min * scale
        min_ = zip(dataMin_!, scale_!).map { dataMin, scale in
            featureRange.min - dataMin * scale
        }
    }

    /// compute minimum of a column using accelerate
    private func _columnMin(_ X: Matrix, column: Int) -> Double {
        let colData = X.column(column)
        var result: Double = 0
        vDSP_minvD(colData, 1, &result, vDSP_Length(colData.count))
        return result
    }

    /// compute maximum of a column using accelerate
    private func _columnMax(_ X: Matrix, column: Int) -> Double {
        let colData = X.column(column)
        var result: Double = 0
        vDSP_maxvD(colData, 1, &result, vDSP_Length(colData.count))
        return result
    }
}

// MARK: - Codable support 

extension MinMaxScaler: Codable {
    enum CodingKeys: String, CodingKey {
        case featureRangeMin = "feature_range_min"
        case featureRangeMax = "feature_range_max"
        case clip
        case dataMin = "data_min_"
        case dataMax = "data_max_"
        case dataRange = "data_range_"
        case scale = "scale_"
        case min = "min_"
        case nFeaturesIn = "n_features_in_"
        case nSamplesSeen = "n_samples_seen_"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let featureMin = try container.decode(Double.self, forKey: .featureRangeMin)
        let featureMax = try container.decode(Double.self, forKey: .featureRangeMax)
        featureRange = (featureMin, featureMax)

        clip = try container.decode(Bool.self, forKey: .clip)
        dataMin_ = try container.decodeIfPresent([Double].self, forKey: .dataMin)
        dataMax_ = try container.decodeIfPresent([Double].self, forKey: .dataMax)
        dataRange_ = try container.decodeIfPresent([Double].self, forKey: .dataRange)
        scale_ = try container.decodeIfPresent([Double].self, forKey: .scale)
        min_ = try container.decodeIfPresent([Double].self, forKey: .min)
        nFeaturesIn_ = try container.decodeIfPresent(Int.self, forKey: .nFeaturesIn)
        nSamplesSeen_ = try container.decodeIfPresent(Int.self, forKey: .nSamplesSeen)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(featureRange.min, forKey: .featureRangeMin)
        try container.encode(featureRange.max, forKey: .featureRangeMax)
        try container.encode(clip, forKey: .clip)
        try container.encodeIfPresent(dataMin_, forKey: .dataMin)
        try container.encodeIfPresent(dataMax_, forKey: .dataMax)
        try container.encodeIfPresent(dataRange_, forKey: .dataRange)
        try container.encodeIfPresent(scale_, forKey: .scale)
        try container.encodeIfPresent(min_, forKey: .min)
        try container.encodeIfPresent(nFeaturesIn_, forKey: .nFeaturesIn)
        try container.encodeIfPresent(nSamplesSeen_, forKey: .nSamplesSeen)
    }
}

// MARK: - CustomStringConvertible

extension MinMaxScaler: CustomStringConvertible {
    public var description: String {
        if isFitted {
            return "MinMaxScaler(featureRange: (\(featureRange.min), \(featureRange.max)), clip: \(clip), nFeatures: \(nFeaturesIn_!))"
        } else {
            return "MinMaxScaler(featureRange: (\(featureRange.min), \(featureRange.max)), clip: \(clip), unfitted)"
        }
    }
}




   


