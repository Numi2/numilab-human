import Foundation
import Metal

enum ProbeError: Error {
    case input, metal, failedCommand
}

func main() throws {
    let args=CommandLine.arguments
    guard args.count==3 else { throw ProbeError.input }
    let input=try Data(contentsOf:URL(fileURLWithPath:args[1]),options:.mappedIfSafe)
    func u32(_ i:Int)->Int { Int(input.withUnsafeBytes { $0.loadUnaligned(fromByteOffset:i,as:UInt32.self) }) }
    let vertices=u32(0), faces=u32(4), windows=u32(8)
    let windowsBegin=12+vertices*96+faces*12
    guard vertices>0, faces>0, windows>0, windowsBegin+windows*56==input.count else { throw ProbeError.input }
    guard let device=MTLCreateSystemDefaultDevice(), let queue=device.makeCommandQueue(),
        let vertexBuffer=input.withUnsafeBytes({ device.makeBuffer(bytes:$0.baseAddress!.advanced(by:12),
             length:vertices*96,options:.storageModeShared) }),
        let output=device.makeBuffer(length:vertices*12,options:.storageModeShared) else { throw ProbeError.metal }
    let lib=try device.makeLibrary(URL:URL(fileURLWithPath:"common-field-probe.metallib"))
    guard let function=lib.makeFunction(name:"pose_common_field") else { throw ProbeError.metal }
    let pipe=try device.makeComputePipelineState(function:function)
    var all = Data()
    for i in 0..<windows {
        let start=windowsBegin+i*56
        if input[start ..< start+28] != input[start+28 ..< start+56] { continue }
        guard let job=queue.makeCommandBuffer(), let encoder=job.makeComputeCommandEncoder() else { throw ProbeError.metal }
        encoder.setComputePipelineState(pipe)
        encoder.setBuffer(vertexBuffer,offset:0,index:0)
        input.withUnsafeBytes { encoder.setBytes($0.baseAddress!.advanced(by:start),length:28,index:1) }
        encoder.setBuffer(output,offset:0,index:2)
        encoder.dispatchThreads(MTLSize(width:vertices,height:1,depth:1),
                                threadsPerThreadgroup:MTLSize(width:pipe.threadExecutionWidth*4,height:1,depth:1))
        encoder.endEncoding()
        job.commit()
        job.waitUntilCompleted()
        guard job.status == .completed else { throw ProbeError.failedCommand }
        var index=UInt32(i)
        withUnsafeBytes(of:&index) { all.append(contentsOf:$0) }
        all.append(Data(bytes:output.contents(),count:output.length))
    }
    try all.write(to:URL(fileURLWithPath:args[2]),options:.atomic)
    print("device=\(device.name) vertices=\(vertices) poses=\(all.count/(4+vertices*12))")
}

try main()
