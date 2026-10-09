import Foundation
import AVFoundation
import CoreVideo
let url = URL(fileURLWithPath: CommandLine.arguments[1])
for (label, settings) in [("compressed", nil as [String: Any]?), ("bgra", [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA] as [String: Any]?)] {
  let asset = AVURLAsset(url: url)
  let track = asset.tracks(withMediaType: .video).first!
  let reader = try AVAssetReader(asset: asset)
  let out = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
  out.alwaysCopiesSampleData = false; reader.add(out)
  guard reader.startReading() else { fatalError("reader start \(label)") }
  var lines=[String](); var n=0
  while let sb=out.copyNextSampleBuffer() { lines.append("\(n)\t\(CMSampleBufferGetOutputPresentationTimeStamp(sb).seconds)\t\(CMSampleBufferGetNumSamples(sb))"); n += 1 }
  guard reader.status == .completed else { fatalError("reader completion \(label): \(String(describing: reader.error))") }
  try (lines.joined(separator:"\n")+"\n").write(toFile: CommandLine.arguments[2]+"/\(label)-pts.tsv", atomically:true, encoding:.utf8)
  print("\(label) samples=\(n) first=\(lines.first ?? "") last=\(lines.last ?? "")")
}
