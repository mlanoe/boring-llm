import Metal
import Foundation
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device const float* a [[buffer(0)]], device float* o [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) o[i] = a[i] + 1.0f; }
"""
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "add_one")!)
let nElems = Int(CommandLine.arguments[2])!
let launches = Int(CommandLine.arguments[1])!
let a = dev.makeBuffer(length: nElems * 4, options: .storageModeShared)!
let b = dev.makeBuffer(length: nElems * 4, options: .storageModeShared)!
var n = UInt32(nElems)
func encode(_ enc: MTLComputeCommandEncoder, _ i: Int) {
    enc.setComputePipelineState(pso)
    enc.setBuffer(i % 2 == 0 ? a : b, offset: 0, index: 0)
    enc.setBuffer(i % 2 == 0 ? b : a, offset: 0, index: 1)
    enc.setBytes(&n, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: nElems, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}
func perCommandBuffer() -> Double {
    let t0 = Date()
    var last: MTLCommandBuffer! = nil
    for i in 0..<launches {
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        encode(enc, i); enc.endEncoding(); cb.commit(); last = cb
    }
    last.waitUntilCompleted()
    return Date().timeIntervalSince(t0)
}
func oneEncoder() -> Double {
    let t0 = Date()
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    for i in 0..<launches { encode(enc, i) }
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return Date().timeIntervalSince(t0)
}
func oneCommandBufferManyEncoders() -> Double {
    let t0 = Date()
    let cb = q.makeCommandBuffer()!
    for i in 0..<launches { let enc = cb.makeComputeCommandEncoder()!; encode(enc, i); enc.endEncoding() }
    cb.commit(); cb.waitUntilCompleted()
    return Date().timeIntervalSince(t0)
}
_ = perCommandBuffer(); _ = oneEncoder()
func med(_ f: () -> Double) -> Double { var v = (0..<7).map { _ in f() }; v.sort(); return v[3] }
let A = med(perCommandBuffer), B = med(oneCommandBufferManyEncoders), C = med(oneEncoder)
print(String(format: "launches=%d n=%d: one command buffer per launch %.1f us/launch; one command buffer, one encoder per launch %.1f us/launch; one command buffer, one encoder %.1f us/launch",
             launches, nElems, A / Double(launches) * 1e6, B / Double(launches) * 1e6, C / Double(launches) * 1e6))
