// Errors.swift
// Author: AaryanshSahay
// 
// Errors thrown by SwiftLearn

public enum SwiftLearnError: Error, LocalizedError {
    case notFitted(String)
    case invalidInput(String)
    case dimensionMismatch(Expected: Int, got: Int)
    case serializationError(String)

    public var errorDescription: String? {
        switch self {
            case .notFitted(let msg):
                return "NotFittedError: \(msg)"
            case .invalidInput(let msg):
                return "InvalidInputerror: \(msg)"
            case .dimensionMismatch(let expected, let got):
                return "DimensionMismatchError: expected \(expected) features, got \(got)"
            case .serializationError(let msg):
                return "SerializationError: \(msg)"
        }
    }
}

