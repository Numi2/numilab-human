import AppKit
import CoreGraphics
import Foundation

for raw in CommandLine.arguments.dropFirst() {
    let url = URL(fileURLWithPath: raw)
    let data = try Data(contentsOf: url)
    guard let rep = NSBitmapImageRep(data: data), let image = rep.cgImage else {
        fatalError("cannot decode PNG: \(raw)")
    }
    let width = image.width
    let height = image.height
    guard width > 0 && height > 0 else { fatalError("empty image dimensions: \(raw)") }
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
    let ok = rgba.withUnsafeMutableBytes { storage -> Bool in
        guard let context = CGContext(data: storage.baseAddress, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: colorSpace, bitmapInfo: bitmapInfo) else { return false }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    guard ok else { fatalError("could not rasterize PNG: \(raw)") }
    var unique = Set<UInt32>()
    var nonblack = 0
    var sum = 0.0
    var sumSquares = 0.0
    let pixels = width * height
    for i in 0..<pixels {
        let offset = i * 4
        let r = UInt32(rgba[offset])
        let g = UInt32(rgba[offset + 1])
        let b = UInt32(rgba[offset + 2])
        if (r | g | b) != 0 { nonblack += 1 }
        unique.insert((r << 16) | (g << 8) | b)
        let luminance = (Double(r) + Double(g) + Double(b)) / 3.0
        sum += luminance
        sumSquares += luminance * luminance
    }
    let mean = sum / Double(pixels)
    let std = sqrt(max(0, sumSquares / Double(pixels) - mean * mean))
    guard nonblack > 0 && unique.count > 1 else { fatalError("decoded PNG is visually empty: \(raw)") }
    print("file=\(url.lastPathComponent) width=\(width) height=\(height) mean_rgb=\(mean) std_rgb=\(std) unique_rgb=\(unique.count) nonblack_pixels=\(nonblack)")
}
