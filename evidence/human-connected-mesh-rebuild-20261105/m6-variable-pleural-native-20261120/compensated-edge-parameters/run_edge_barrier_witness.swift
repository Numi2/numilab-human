import Foundation
import Metal

let args = CommandLine.arguments
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: args[1]))
let name = "numi_matter_metal::nm_contact_retained_edge_derivative_check"
let function = try library.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
let pipeline = try device.makeComputePipelineState(function: function)
let inputBytes = try Data(contentsOf: URL(fileURLWithPath: args[2]))
precondition(inputBytes.count == 17 * MemoryLayout<SIMD4<Float>>.stride)
let input = inputBytes.withUnsafeBytes {
    device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)!
}
let output = device.makeBuffer(length: 2 * MemoryLayout<SIMD4<Float>>.stride,
                               options: .storageModeShared)!
let command = queue.makeCommandBuffer()!
let encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(input, offset: 0, index: 0)
encoder.setBuffer(output, offset: 0, index: 1)
encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
encoder.endEncoding()
command.commit()
command.waitUntilCompleted()
precondition(command.status == .completed, String(describing: command.error))
let values = output.contents().bindMemory(to: Float.self, capacity: 8)
let result: [String: Any] = ["device": device.name, "analytic_j": values[0],
    "centered_j": values[1], "base_energy_j": values[2], "base_distance_m": values[3],
    "valid": values[4] == 1, "plus_m": values[5], "minus_m": values[6]]
let json = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
print(String(decoding: json, as: UTF8.self))
