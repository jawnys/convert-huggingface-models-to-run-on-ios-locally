#!/usr/bin/env swift

import AppKit
import CoreML
import Foundation

let modelPackagePath = "../step-1-convert/outputs/DepthAnythingV2Small.mlpackage"
let imagePath = "../shirt.png"
let outputPreviewPath = "outputs/shirt-depth-coreml-step2.png"
let expectedInputSize = CGSize(width: 518, height: 518)

func loadModel(at packageURL: URL) throws -> MLModel {
    guard FileManager.default.fileExists(atPath: packageURL.path) else {
        throw NSError(
            domain: "DepthAnything",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Model not found at \(packageURL.path)"]
        )
    }

    let compiledURL = try MLModel.compileModel(at: packageURL)
    return try MLModel(contentsOf: compiledURL)
}

func resizedPixelBuffer(from image: NSImage, size: CGSize) -> CVPixelBuffer? {
    guard
        let tiff = image.tiffRepresentation,
        let bitmap = NSBitmapImageRep(data: tiff),
        let cgImage = bitmap.cgImage
    else {
        return nil
    }

    let attrs: [String: Any] = [
        kCVPixelBufferCGImageCompatibilityKey as String: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
    ]

    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        Int(size.width),
        Int(size.height),
        kCVPixelFormatType_32ARGB,
        attrs as CFDictionary,
        &pixelBuffer
    )

    guard status == kCVReturnSuccess, let pb = pixelBuffer else {
        return nil
    }

    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
    else {
        return nil
    }

    context.interpolationQuality = .high
    context.draw(cgImage, in: CGRect(origin: .zero, size: size))

    return pb
}

func firstMultiArray(from prediction: MLFeatureProvider) -> MLMultiArray? {
    for name in prediction.featureNames.sorted() {
        if let value = prediction.featureValue(for: name), let array = value.multiArrayValue {
            return array
        }
    }
    return nil
}

func multiArrayStats(_ array: MLMultiArray) -> (min: Double, max: Double, mean: Double) {
    var minValue = Double.greatestFiniteMagnitude
    var maxValue = -Double.greatestFiniteMagnitude
    var sum: Double = 0

    for i in 0..<array.count {
        let value = array[i].doubleValue
        minValue = min(minValue, value)
        maxValue = max(maxValue, value)
        sum += value
    }

    let meanValue = sum / Double(array.count)
    return (minValue, maxValue, meanValue)
}

func value(at row: Int, col: Int, in array: MLMultiArray, height: Int, width: Int) -> Double {
    let rank = array.shape.count
    var indices = Array(repeating: 0, count: rank)
    indices[rank - 2] = row
    indices[rank - 1] = col
    let nsIndices = indices.map { NSNumber(value: $0) }
    return array[nsIndices].doubleValue
}

func saveDepthPreviewImage(
    depthArray: MLMultiArray,
    height: Int,
    width: Int,
    outputURL: URL
) throws {
    var minValue = Double.greatestFiniteMagnitude
    var maxValue = -Double.greatestFiniteMagnitude
    var values = Array(repeating: 0.0, count: height * width)

    for row in 0..<height {
        for col in 0..<width {
            let index = row * width + col
            let v = value(at: row, col: col, in: depthArray, height: height, width: width)
            values[index] = v
            minValue = min(minValue, v)
            maxValue = max(maxValue, v)
        }
    }

    let range = max(maxValue - minValue, 1e-8)
    var pixels = Array(repeating: UInt8(0), count: height * width)

    for i in 0..<(height * width) {
        let normalized = (values[i] - minValue) / range
        let clamped = min(max(normalized, 0.0), 1.0)
        pixels[i] = UInt8((clamped * 255.0).rounded())
    }

    let data = Data(pixels)
    guard let provider = CGDataProvider(data: data as CFData) else {
        throw NSError(domain: "DepthAnything", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGDataProvider"])
    }

    guard let cgImage = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 8,
        bytesPerRow: width,
        space: CGColorSpaceCreateDeviceGray(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
        provider: provider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
    ) else {
        throw NSError(domain: "DepthAnything", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGImage"])
    }

    let rep = NSBitmapImageRep(cgImage: cgImage)
    guard let pngData = rep.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "DepthAnything", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to encode PNG"])
    }

    let outputDir = outputURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
    try pngData.write(to: outputURL)
}

func main() {
    print("Start testing Core ML depth model with shirt.png...")

    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let modelURL = cwd.appendingPathComponent(modelPackagePath)
    let imageURL = cwd.appendingPathComponent(imagePath)
    let outputURL = cwd.appendingPathComponent(outputPreviewPath)

    guard let image = NSImage(contentsOf: imageURL) else {
        print("Failed to open image at: \(imageURL.path)")
        exit(1)
    }

    guard let pixelBuffer = resizedPixelBuffer(from: image, size: expectedInputSize) else {
        print("Failed to create a resized pixel buffer")
        exit(1)
    }

    let model: MLModel
    do {
        model = try loadModel(at: modelURL)
    } catch {
        print("Failed to load model: \(error.localizedDescription)")
        exit(1)
    }

    guard let inputName = model.modelDescription.inputDescriptionsByName.keys.first else {
        print("Failed to find model input")
        exit(1)
    }

    do {
        let input = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(pixelBuffer: pixelBuffer)]
        )
        let prediction = try model.prediction(from: input)

        guard let depthArray = firstMultiArray(from: prediction) else {
            print("No MLMultiArray output found in prediction")
            exit(1)
        }

        let shape = depthArray.shape.map { $0.intValue }
        guard shape.count >= 2 else {
            print("Unexpected output shape: \(shape)")
            exit(1)
        }

        let height = shape[shape.count - 2]
        let width = shape[shape.count - 1]
        let stats = multiArrayStats(depthArray)

        let center = value(at: height / 2, col: width / 2, in: depthArray, height: height, width: width)
        let topCenter = value(at: 0, col: width / 2, in: depthArray, height: height, width: width)
        let bottomCenter = value(at: height - 1, col: width / 2, in: depthArray, height: height, width: width)
        let leftCenter = value(at: height / 2, col: 0, in: depthArray, height: height, width: width)
        let rightCenter = value(at: height / 2, col: width - 1, in: depthArray, height: height, width: width)

        try saveDepthPreviewImage(depthArray: depthArray, height: height, width: width, outputURL: outputURL)

        print("Model: \(modelURL.path)")
        print("Image: \(imageURL.path)")
        print("Output shape: \(shape)")
        print(String(format: "Depth min: %.6f", stats.min))
        print(String(format: "Depth max: %.6f", stats.max))
        print(String(format: "Depth mean: %.6f", stats.mean))

        print("Sample depth values:")
        print(String(format: "  center:        %.6f", center))
        print(String(format: "  top_center:    %.6f", topCenter))
        print(String(format: "  bottom_center: %.6f", bottomCenter))
        print(String(format: "  left_center:   %.6f", leftCenter))
        print(String(format: "  right_center:  %.6f", rightCenter))
        print("Saved preview image: \(outputURL.path)")

        print("Finished testing Core ML model.")
    } catch {
        print("Prediction failed: \(error.localizedDescription)")
        exit(1)
    }
}

main()
