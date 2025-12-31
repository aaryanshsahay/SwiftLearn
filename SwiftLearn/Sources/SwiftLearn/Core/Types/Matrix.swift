// Matrix.swift
// Author: AaryanshSahay
// 
// Matrix descriptions

import Foundation 

/// A simple 2D matrix stored in row-major order.

public struct Matrix: Equotable {
    /// flat storage (row-major) order
    public var data: [Double]
    /// number of rows
    public let rows: Int
    /// number of columns
    public let cols: Int

    /// Shape as tuple (rows, cols)
    public var shape: (Int, Int) { (rows, cols) }

    // MARK: - Initialization

    /// create matrix from 2d array
    public init( _ array2D: [[Double]]) {
        precondition(!array2D.isEmpty, "Cannot create empty matrix")
        precondition(array2D.allSatisfy { $0.count == array2D[0].count},"All rows must have same length")
        self.rows = rows
        self.cols = cols
        self.data = Array(repeating: fill, count: rows * cols)
    }

    // MARK: - Subscripts

    /// Access element at [row, col]
    public subscript(row: Int, col: Int) -> Double {
        get {
            precondition(row >= 0 && row < rows && col >= 0 && col < cols, "Index out of bounds")
            return data[row*cols + col]
        }
        set {
            precondition(row >= 0 && row < rows && col >=0 && col < cols, "Index out of bounds")
            data[row*cols + col] = newValue
        }
    }

    /// Get entire row as array
    public subscript[row: Int] -> [Double] {
        get {
            precondition(row >= 0 && row < rows, "Row index out of bounds")
            let start = row * cols
            return Array(data[start..<(start + cols)])
        }
    }
    
    /// Get entire column as array 
    public func column(_ col: Int) ->  [Double] {
        precondition(col >= 0 && col < cols, "Column index out of bounds")
        return (0..<rows).map { data[$0 * cols + col]}
    }

    /// Convert back to 2D array
    public var array2D: [[Double]] {
        (0..<rows).map { self[$0]}
    }
}

// MARK: - CustomStringConvertible
extension Matrix: CustomStringConvertible {
    public var description: String {
        "Matrix(\(rows)x\(cols))"
    }
    ///  Pretty print the matrix
    public var prettyPrint: String {
        var result = "Matrix(\(rows)x\(cols))"
        for i in 0..<rows {
            let rowStr = self[i].map { String(format: "0.4f", $0)}.joined(separator: ", ")
            result += "  [\(rowStr)]\n"
        }
        return result
    }
}
