import Metal
import Foundation

// usage: harness file.metal kernel tg_threads tile_m tile_n [--D NAME=VAL]... [--m M --n N --k K --iters I --reps R --out f --ref f --cpu]
var args = Array(CommandLine.arguments.dropFirst())
func opt(_ name: String, _ def: String) -> String {
    if let i = args.firstIndex(of: name), i + 1 < args.count { return args[i + 1] }
    return def
}
var defines: [String: NSObject] = [:]
var j = 0
while j < args.count {
    if args[j] == "--D" {
        let kv = args[j + 1].split(separator: "=", maxSplits: 1).map(String.init)
        if let iv = Int(kv[1]) { defines[kv[0]] = NSNumber(value: iv) as NSObject } else { defines[kv[0]] = kv[1] as NSString }
    }
    j += 1
}
let file = args[0], kname = args[1]
let tg = Int(args[2])!, tileM = Int(args[3])!, tileN = Int(args[4])!
let M = Int(opt("--m", "512"))!, N = Int(opt("--n", "18944"))!, K = Int(opt("--k", "3584"))!
let iters = Int(opt("--iters", "4"))!, reps = Int(opt("--reps", "5"))!

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let src = try! String(contentsOfFile: file, encoding: .utf8)
let opts = MTLCompileOptions()
opts.preprocessorMacros = defines
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: opts) } catch { print("compile error: \(error)"); exit(2) }
let fn = lib.makeFunction(name: kname)!
let pso = try! dev.makeComputePipelineState(function: fn)

// deterministic data
var seed: UInt64 = 0x9E3779B97F4A7C15
func rnd() -> UInt64 { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed }
func frnd() -> Float { Float(rnd() % 2000001) / 1000000.0 - 1.0 }
let blocksPerRow = K / 256
var wbytes = [UInt8](repeating: 0, count: N * blocksPerRow * 144)
for r in 0..<N * blocksPerRow {
    let base = r * 144
    let d = Float16(0.002).bitPattern, dm = Float16(0.001).bitPattern
    wbytes[base] = UInt8(d & 255); wbytes[base + 1] = UInt8(d >> 8)
    wbytes[base + 2] = UInt8(dm & 255); wbytes[base + 3] = UInt8(dm >> 8)
    for i in 4..<144 { wbytes[base + i] = UInt8(truncatingIfNeeded: rnd() >> 24) }
}
var act = [Float](repeating: 0, count: M * K)
for i in 0..<act.count { act[i] = frnd() }
let ba = dev.makeBuffer(bytes: act, length: act.count * 4, options: .storageModeShared)!
let bb = dev.makeBuffer(bytes: wbytes, length: wbytes.count, options: .storageModeShared)!
let bc = dev.makeBuffer(length: M * N * 4, options: .storageModeShared)!
var mu = UInt32(M), nu = UInt32(N), ku = UInt32(K)

func run(_ count: Int) -> Double {
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(ba, offset: 0, index: 0); enc.setBuffer(bb, offset: 0, index: 1); enc.setBuffer(bc, offset: 0, index: 2)
    enc.setBytes(&mu, length: 4, index: 3); enc.setBytes(&nu, length: 4, index: 4); enc.setBytes(&ku, length: 4, index: 5)
    for _ in 0..<count {
        enc.dispatchThreadgroups(MTLSize(width: (N + tileN - 1) / tileN, height: (M + tileM - 1) / tileM, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}
_ = run(1)
var times: [Double] = []
for _ in 0..<reps { times.append(run(iters) / Double(iters)) }
times.sort()
let med = times[times.count / 2]
let flop = 2.0 * Double(M) * Double(N) * Double(K)
print(String(format: "%@ m=%d n=%d k=%d  median %.2f ms  %.3f TFLOP/s  (min %.2f max %.2f ms)  maxThreads=%d tgmem=%d",
             kname, M, N, K, med * 1000, flop / med / 1e12, times[0] * 1000, times[times.count - 1] * 1000,
             pso.maxTotalThreadsPerThreadgroup, 0))

let outp = bc.contents().bindMemory(to: Float.self, capacity: M * N)
if let path = args.firstIndex(of: "--out") { 
    _ = run(1)
    FileManager.default.createFile(atPath: args[path + 1], contents: Data(bytes: outp, count: M * N * 4))
}
if let path = args.firstIndex(of: "--ref") {
    _ = run(1)
    let ref = try! Data(contentsOf: URL(fileURLWithPath: args[path + 1]))
    ref.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
        let r = p.bindMemory(to: Float.self)
        var maxd: Float = 0, maxr: Float = 0, sum: Double = 0
        for i in 0..<(M * N) { maxd = max(maxd, abs(outp[i] - r[i])); maxr = max(maxr, abs(r[i])); sum += Double(abs(outp[i] - r[i])) }
        print(String(format: "  vs ref: max|diff| %.3e  (%.2e of max|ref| %.2f)  mean|diff| %.3e", maxd, maxd / maxr, maxr, sum / Double(M * N)))
    }
}
if args.contains("--cpu") {
    _ = run(1)
    func f16(_ lo: UInt8, _ hi: UInt8) -> Double { Double(Float16(bitPattern: UInt16(lo) | UInt16(hi) << 8)) }
    func sm(_ blk: Int, _ s: Int) -> (Int, Int) {
        let b = blk * 144 + 4
        if s < 4 { return (Int(wbytes[b + s]) & 63, Int(wbytes[b + s + 4]) & 63) }
        let sc = (Int(wbytes[b + s + 4]) & 15) | ((Int(wbytes[b + s - 4]) >> 6) << 4)
        let mn = (Int(wbytes[b + s + 4]) >> 4) | ((Int(wbytes[b + s]) >> 6) << 4)
        return (sc, mn)
    }
    func w(_ col: Int, _ kk: Int) -> Double {
        let blk = col * blocksPerRow + kk / 256, pos = kk % 256
        let chunk = pos / 64, within = pos % 64, half = within / 32, lane = within % 32
        let qb = Int(wbytes[blk * 144 + 16 + chunk * 32 + lane])
        let qv = half == 0 ? qb & 15 : qb >> 4
        let (sc, mn) = sm(blk, chunk * 2 + half)
        let d = f16(wbytes[blk * 144], wbytes[blk * 144 + 1]), dmin = f16(wbytes[blk * 144 + 2], wbytes[blk * 144 + 3])
        return Double(qv * sc) * d - Double(mn) * dmin
    }
    var worst = 0.0, scale = 0.0
    for _ in 0..<48 {
        let r = Int(rnd() % UInt64(M)), c = Int(rnd() % UInt64(N))
        var acc = 0.0
        for kk in 0..<K { acc += Double(act[r * K + kk]) * w(c, kk) }
        worst = max(worst, abs(acc - Double(outp[r * N + c]))); scale = max(scale, abs(acc))
    }
    print(String(format: "  vs CPU (48 samples): max|diff| %.3e  (%.2e of max|value| %.2f)", worst, worst / scale, scale))
}
