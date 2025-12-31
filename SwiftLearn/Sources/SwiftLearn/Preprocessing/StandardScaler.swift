// StandardScaler.swift
// Author: AaryanshSahay
// 
// Standardizes features by removing the mean and scaling to unit variance

import Foundation
import Accelerate 

/// Standardize features by removing the mean and scaling to unit variance.
/// The standard score of a sample `x` is calculated as:
/// 
///     z = (x-u) / v
///
/// -> where `u` is the mean (or zero if `withMean=false`)
/// -> where 'v' is the standard deviation (or one if `withStd=false`)
///
/// ## Example
/// ```swift
/// var scaler = StandardScaler()
/// let X = Matrix([[0,0],[0,0],[1,1],[1,1]])
/// let scaled = try scaler.fitTransform(X)
/// // scaled will have mean = 0 and std = 1 for each column
/// ```
///
/// ## Notes
/// - We use a biased estimator for the standard deviation (ddof=0), matching sklearn's behaviour.
/// - Features with zero variance are left as-is (scale factor of 1).

public struct StandardScaler {
    // MARK: - Configuration Parameters

    /// if `true`, center the data by subtracting the mean before scaling.
    public var withMean: Bool

    /// if `true`, scale the data to unit variance (divide by standard deviation).
    public var withStd: Bool

    // MARK: - Learned Attributes (set after fit)

    /// Per-feature mean of the array. `nil` if `withmean` and `withStd` are both `false`.
    public private(set) var mean_: [Double]?

    /// Per-feature variance of the array. `nil` if `withStd` is `false`
    public private(set) var var_: [Double]?

    /// Per-feature scale (std deviation) used for scaling
    /// Features with zero variance have a scale of 1.0. `nil` if `withStd` is false.
    public private(set) var scale_: [Double]?

    /// Number of features seen during `fit`
    public private(set) var nFeaturesIn_: Int?

    /// Number of samples seen duting `fit`
    public private(set) var nSamplesSeen_: Int?

    // MARK: - Computed Properties

    /// Returns `true` if the scaler has been fitted
    public var isFitted: Bool {
        return nFeaturesIn_ != nil
    }

    // MARK: - Initialization

    /// Creates a new StandardScaler
    ///
    /// - Parameters:
    ///     -> withMean: if `true`, center the data before scaling. Default is `true`.
    ///     -> withStd: if `true`, scale the data to unit variance. Default is `true`.
    public init(withMean: Bool = true, withStd: Bool = true){
        self.withMean = withMean
        self.withStd = withStd
    }

    // MARK: - Fit Methods

    /// Compute the mean and standard deviation from the array.
    ///
    /// - Parameters:
    ///     -> X: Array of shape (nSamples, nFeatures)
    ///
    /// - Throws:
    ///     -> `SwiftLearnError.invalidInput` if the input is empty
    public mutating func fit(_ X: Matrix) throws {
        try _reset()
        try _fit(X)
    }

    /// Compute the mean and standard deviation, then transform the data
    ///
    /// - Parameters
    ///     -> X: Array of shape (nSamples, nFeatures)
    ///
    /// - Returns
    ///     -> Transformed data with zero mean and unit variance
    ///
    /// - Throws
    ///     -> `SwiftLearnError.invalidInput` if the input is empty.
    public mutating func fitTransform(_ X: Matrix) throws -> Matrix{
        try fit(X)
        return try transform(X)
    }

    // MARK: - Transform Methods
    
    /// Standardize the data by centering and scaling.
    ///
    /// - Parameter
    ///     -> X: Data to transform of shape (nSamples, nFeatures)
    ///
    /// - Returns:
    ///     -> Transformed data
    ///
    /// - Throws:
    ///     -> `SwiftLearnError.notFitted` if `fit` has not been called
    ///     -> `SwiftLearnError.dimensionMismatch` if number of features doesn't match training data.
    public func transform(_ X: Matrix) throws -> Matrix {
        guard isFitted else {
            throw SwiftLearnError.notFitted(
                "StandardScaler has not been fitted. Call fit() first."
            )
        }

        guard X.cols == nFeaturesIn_ else {
            throw SwiftLearnError.dimensionMismatch(
                expected: nFeaturesIn_!,
                got: X.cols
            )
        }

        var result = X

        // Apply centering: X = X - mean
        if withMean, let mean = mean_ {
            for i in 0..<result.rows {
                for j in 0..<result.cols {
                    result[i,j] -= mean[j]
                }
            }
        }

        // Apply scaling: X = X / scale
        if withStd, let scale = scale_ {
            for i in 0..<result.rows {
                for j in 0..<result.cols {
                    result[i,j] /= scale[j]
                }
            }
        }
        return result 
    }

    /// Transform data back to original scale.
    ///
    /// - Parameter:
    ///     -> X: Transformed data of shape (nSamples, nFeatures).
    ///
    /// - Returns: 
    ///     -> Data in original scale
    ///
    /// - Throws:
    ///     -> `SwiftLearnError.notFitted` if `fit` has not been called.
    ///     -> `SwiftLearnError.dimensionMismatch` if number of features doesn't match training data.
    public func inverseTransform(_ X: Matrix) throws -> Matrix {
        guard isFitted else {
            throw SwiftLearnError.notFitted(
                "StandardScaler has not been fitted. Call fit() first."
            )
        }

        guard X.cols == nFeaturesIn_! else {
            throw SwiftLearnError.dimensionMismatch(
                expected: nFeaturesIn_!,
                got: X.cols
            )
        }

        var result = X

        // Reverse scaling: X = X * scale
        if withStd, let scale = scale_ {
            for i in 0..<result.rows {
                for j in 0..<result.cols {
                    result[i,j] *= scale[j]
                }
            }
        }

        // Reverse centering: X = X + mean
        if withMean, let mean = mean_ {
            for i in 0..<result.rows{
                for j in 0..<result.cols{
                    result[i,j] += mean[j]
                }
            }
        }

        return result
    }

    // MARK: - Private Methods

    /// reset interanl state before fitting.
    private mutating func _reset() throws {
        mean_ = nil
        var_ = nil
        scale_ = nil
        nFeaturesIn_ = nil
        nSamplesSeen_ = nil
    }

    /// Internal fit implemenation 
    private mutating func _fit(_ X: Matrix) throws {
        // validate input
        guard X.rows > 0 else {
            throw SwiftLearnError.invalidInput("Input matrix has no samples (rows).")
        }

        guard X.cols > 0 else {
            throw SwiftLearnError.invalidInput("Input matrix has no features (columns).")
        }

        nFeaturesIn_ = X.cols
        nSamplesSeen_ = X.rows

        let nSamples = Double(X.rows)

        // Compute mean for each feature (column)
        if withMean || withStd {
            mean_ = (0..<X.cols).map { col in
                _columnMean(X, column: col)
            }
        }

        if withStd{
            var_ = (0..<X.cols).map { col in 
                _columnVariance(X, column: col, mean: mean_![col])
            }
            // Scale = sqrt(variance), handles zero variance
            scale_ = var_!.map { variance in 
                let std = sqrt(variance)
                // if variance is zeor (constant feature), use scale of 1.0 to avoid division by zero
                return std < 1e-10 ? 1.0 : std

            }
        }
    }

    /// Compute mean of a column using Accelerate.
    private func _columnMean(_ X: Matrix, column: Int) -> Double {
        let colData = X.column(column)
        var result: Double = 0
        vDSP_meanvD(colData, 1, &result, vDSP_Length(colData.count))
        return result
    }

    /// Compute variance of a column (biased, ddof=0) using Accelerate.
    private func _columnVariance(_ X: Matrix, column: Int, mean: Double) -> Double {
        let colData = X.column(column)
        let n = Double(colData.count)

        // compute sum of squared differences from mean
        // variance = sum((x-mean)^2)/n
        var squaredDiffs = colData.map{ ($0 - mean) * ($0 - mean)}
        var sumSquared: Double = 0
        vDSP_sveD(squaredDiffs, 1, &sumSquared, vDSP_Length(squaredDiffs.count))

        return sumSquared/n
    }
}

// MARK: - Codable support (for serialization)

extension StandardScaler: Codable {
    enum CodingKeys: String, CodingKey{
        case withMean = "with_mean"
        case withStd = "with_std"
        case mean = "mean_"
        case variance = "var_"
        case scale = "scale_"
        case nFeaturesIn = "n_features_in_"
        case nSamplesSeen = "n_samples_seen_"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        withMean = try container.decode(Bool.self, forKey: .withMean)
        withStd = try container.decode(Bool.self, forKey: .withStd)
        mean_ = try container.decodeIfPresent([Double].self, forKey: .mean)
        var_ = try container.decodeIfPresent([Double].self, forKey: .variance)
        scale_ = try container.decodeIfPresent([Double].self, forKey: .scale)
        nFeaturesIn_ = try container.decodeIfPresent(Int.self, forKey: .nFeaturesIn)
        nSamplesSeen_ = try container.decodeIfPresent(Int.self, forKey: .nSamplesSeen)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(withMean, forKey: .withMean)
        try container.encode(withStd, forKey: .withStd)
        try container.encodeIfPresent(mean_, forKey: .mean)
        try container.encodeIfPresent(var_, forKey: .variance)
        try container.encodeIfPresent(scale_, forKey: .scale)
        try container.encodeIfPresent(nFeaturesIn_, forKey: .nFeaturesIn)
        try container.encodeIfPresent(nSamplesSeen_, forKey: .nSamplesSeen)
    }
}

// MARK: - CustomStringConvertible

extension StandardScaler: CustomStringConvertible {
    public var description: String {
        if isFitted {
            return "StandardScaler(withMean: \(withMean), withStd: \(withStd), nFeatures: \(nFeaturesIn_!))"
        } else {
            return "StandardScaler(withMean: \(withMean), withStd: \(withStd), unfitted)"
        }
    }
}