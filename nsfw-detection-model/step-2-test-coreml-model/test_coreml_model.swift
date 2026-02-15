#!/usr/bin/env swift

import Foundation
import CoreML
import AppKit

func main() {
  print("🏎️ Start testing CoreML model...")

  let MODEL_PACKAGE_PATH = "../step-1-convert-pytorch-model-to-coreml-model/outputs/NsfwDetector.mlpackage"
  let TEST_IMAGES_PATH = "../test-images"

  let repoRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
  let modelPackage = repoRoot.appendingPathComponent(MODEL_PACKAGE_PATH)
  let imagesDir = repoRoot.appendingPathComponent(TEST_IMAGES_PATH)

  func loadModel(at packageURL: URL) throws -> MLModel {
      // If a .mlmodelc or compiled artifact already exists, `MLModel(contentsOf:)` can load it directly.
      // Compile if needed and return the model.
      if FileManager.default.fileExists(atPath: packageURL.path) {
          let compiled = try MLModel.compileModel(at: packageURL)
          return try MLModel(contentsOf: compiled)
      }
      throw NSError(domain: "Model", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model package not found at \(packageURL.path)"])
  }

  func imageURLs(in directory: URL) -> [URL] {
      guard let items = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { return [] }
      let exts = ["jpg", "jpeg", "png", "bmp", "gif", "avif", "heic"]
      return items.filter { exts.contains($0.pathExtension.lowercased()) }.sorted { $0.lastPathComponent < $1.lastPathComponent }
  }

  func resizedPixelBuffer(from image: NSImage, size: CGSize) -> CVPixelBuffer? {
      guard let tiff = image.tiffRepresentation,
            let src = NSBitmapImageRep(data: tiff)
      else { return nil }

      let cg = src.cgImage
      let attrs: [String: Any] = [
          kCVPixelBufferCGImageCompatibilityKey as String: true,
          kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
      ]
      var pb: CVPixelBuffer?
      let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height), kCVPixelFormatType_32ARGB, attrs as CFDictionary, &pb)
      guard status == kCVReturnSuccess, let pixelBuffer = pb else { return nil }

      CVPixelBufferLockBaseAddress(pixelBuffer, [])
      defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

      let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
      guard let context = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer), width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else { return nil }

      context.interpolationQuality = .high
      let rect = CGRect(origin: .zero, size: size)
      if let cg = cg {
          context.draw(cg, in: rect)
      } else {
          return nil
      }

      return pixelBuffer
  }

  func predict(model: MLModel, pixelBuffer: CVPixelBuffer) -> (isNSFW: Bool, confidence: Double)? {
      guard let inputName = model.modelDescription.inputDescriptionsByName.keys.first,
            let outputName = model.modelDescription.outputDescriptionsByName.keys.first
      else { return nil }

      do {
          let features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pixelBuffer)])
          let result = try model.prediction(from: features)
          guard let feature = result.featureValue(for: outputName), let arr = feature.multiArrayValue else { return nil }

          if arr.count >= 2 {
              let normal = arr[0].doubleValue
              let nsfw = arr[1].doubleValue
              return (nsfw > normal, nsfw)
          }
      } catch {
          return nil
      }
      return nil
  }

  print("NSFW CoreML quick tester")
  print("Model: \(modelPackage.path)")
  print("Images: \(imagesDir.path)")

let images = imageURLs(in: imagesDir)
guard images.count > 0 else {
    print("No images found in \(imagesDir.path)")
    exit(1)
}

  let model: MLModel
  do {
      model = try loadModel(at: modelPackage)
      print("Loaded model: \(model.modelDescription.description)")
  } catch {
      print("Failed to load model: \(error.localizedDescription)")
      exit(1)
  }

  for url in images {
      autoreleasepool {
          let name = url.lastPathComponent
          print("\nTesting: \(name)")
          guard let img = NSImage(contentsOf: url) else { print(" - failed to open image"); return }
          guard let pb = resizedPixelBuffer(from: img, size: CGSize(width: 224, height: 224)) else { print(" - failed to create pixel buffer"); return }
          if let r = predict(model: model, pixelBuffer: pb) {
              let confPct = String(format: "%.2f", r.confidence * 100)
              print(" - Result: \(r.isNSFW ? "🔞 NSFW" : "✅ SAFE") (nsfw confidence: \(confPct)%)")
          } else {
              print(" - prediction failed")
          }
      }
  }

  print("🏁 Finished testing CoreML model")
}

main();
