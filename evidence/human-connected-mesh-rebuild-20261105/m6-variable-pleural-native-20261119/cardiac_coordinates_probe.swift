import Foundation
import Metal

enum ProbeError: Error { case input, initialization, execution }

func main() throws {
    let args = CommandLine.arguments
    guard args.count == 6 else { throw ProbeError.input }
    let data = try args[2..<5].map { try Data(contentsOf: URL(fileURLWithPath: $0)) }
    guard !data[0].isEmpty, data[0].count % 16 == 0,
          data[1].count == 3504, !data[2].isEmpty, data[2].count % 64 == 0 else {
        throw ProbeError.input
    }
    let count = data[0].count / 16
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else { throw ProbeError.initialization }
    let lib = try device.makeLibrary(URL: URL(fileURLWithPath: args[1]))
    guard let kernel = lib.makeFunction(name: "solve_cardiac_targets"),
          let output = device.makeBuffer(length: count*64, options: .storageModeShared),
          let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else { throw ProbeError.initialization }
    let pipeline = try device.makeComputePipelineState(function: kernel)
    guard pipeline.threadExecutionWidth == 32 else { throw ProbeError.initialization }
    let buffers = try data.map { item in
        guard let gpu = item.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: item.count, options: .storageModeShared)
        }) else { throw ProbeError.initialization }
        return gpu
    }
    encoder.setComputePipelineState(pipeline)
    for (i, buffer) in buffers.enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: i)
    }
    encoder.setBuffer(output, offset: 0, index: 3)
    encoder.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else { throw ProbeError.execution }
    try Data(bytes: output.contents(), count: output.length).write(to: URL(fileURLWithPath: args[5]))
    print("device=\(device.name) solved_samples=\(count)")
}

try main()
