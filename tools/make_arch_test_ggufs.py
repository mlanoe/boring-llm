#!/usr/bin/env python3
"""Builds tiny synthetic GGUF models (random weights) plus the logits an
independent float64 numpy implementation of llama.cpp's graph produces for them,
for test/test_arch.br: Phi-3's fused attn_qkv / ffn_up layout, Gemma's GeGLU +
embedding scale + head size != d_model / n_head, and the RoPE scaling variants
(linear, YaRN, per-pair factors). The numpy reference is transcribed from
llama.cpp/ggml (ggml_rope_yarn, build_phi3, build_gemma), not from boring-llm.

    .venv/bin/python tools/make_arch_test_ggufs.py

Two sets are written: `arch-<case>.gguf` (F32, d_model 16: the interpreter test)
and `arch-q8-<case>.gguf` (Q8_0, d_model 32 so rows hold whole 32-value blocks:
for the compiled GPU path, `main --quantized --tokens` / `--generate-tokens`).
The Q8_0 references use the dequantized weights, so only float rounding is left.
"""
import math
import numpy as np
import gguf

OUT = "test/fixtures"
TOKENS = [3, 1, 4, 1, 5, 9, 2, 6]
VOCAB, NL, NH, NKV = 10, 2, 4, 2
D, FFN = 16, 24  # rebound by main() for the Q8_0 set
Q8 = False
PREFIX = "arch-"
N_GEN = 6
EPS = 1e-6


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def silu(x):
    return x / (1 + np.exp(-x))


def corr_dim(d, orig, n_rot, base):
    return d * math.log(orig / (n_rot * 2 * math.pi)) / (2 * math.log(base))


def rope_table(d_head, base, kind="none", factor=1.0, orig=4096, bf=32.0, bs=1.0, attn=1.0, factors=None):
    half = d_head // 2
    theta = np.array([base ** (-2.0 * j / d_head) for j in range(half)])
    mscale = 1.0
    if factors is not None:
        theta = theta / factors
        mscale = attn
    elif kind == "linear":
        theta = theta / factor
    elif kind == "yarn":
        low = max(0, math.floor(corr_dim(d_head, orig, bf, base)))
        high = min(d_head - 1, math.ceil(corr_dim(d_head, orig, bs, base)))
        out = []
        for j in range(half):
            y = (j - low) / max(0.001, high - low)
            ramp_mix = 1 - min(1, max(0, y))
            out.append(theta[j] / factor * (1 - ramp_mix) + theta[j] * ramp_mix)
        theta = np.array(out)
        mscale = attn * (1 + 0.1 * math.log(factor))
    return theta, mscale


def rope(x, pos, theta, mscale, neox):  # x: (seq, heads, d_head)
    half = x.shape[-1] // 2
    ang = pos[:, None] * theta[None, :]
    c, s = np.cos(ang)[:, None, :] * mscale, np.sin(ang)[:, None, :] * mscale
    out = np.empty_like(x)
    if neox:
        a, b = x[..., :half], x[..., half:]
        out[..., :half], out[..., half:] = a * c - b * s, a * s + b * c
    else:
        a, b = x[..., 0::2], x[..., 1::2]
        out[..., 0::2], out[..., 1::2] = a * c - b * s, a * s + b * c
    return out


def reference(W, c):
    seq = len(TOKENS)
    pos = np.arange(seq, dtype=np.float64)
    x = W["emb"][TOKENS] * c["embd_scale"]
    dh = c["d_head"]
    for l in range(NL):
        L = W["layers"][l]
        h = rms(x, L["attn_norm"])
        q = (h @ L["q"].T).reshape(seq, NH, dh)
        k = (h @ L["k"].T).reshape(seq, NKV, dh)
        v = (h @ L["v"].T).reshape(seq, NKV, dh)
        if c.get("qk_norm"):
            q, k = rms(q, L["q_norm"]), rms(k, L["k_norm"])
        win = c.get("window", 0)
        swa = win > 0 and (c.get("all_swa") or (l % c["pattern"]) < c["pattern"] - 1)
        tab = c["rope_local"] if swa and "rope_local" in c else c["rope"]
        q, k = rope(q, pos, *tab, c["neox"]), rope(k, pos, *tab, c["neox"])
        out = np.zeros((seq, NH, dh))
        for hd in range(NH):
            kv = hd // (NH // NKV)
            sc = q[:, hd] @ k[:, kv].T / math.sqrt(dh)
            if c.get("attn_softcap"):
                sc = c["attn_softcap"] * np.tanh(sc / c["attn_softcap"])
            idx = np.arange(seq)
            allowed = idx[None, :] <= idx[:, None]
            if swa:
                allowed &= idx[None, :] > idx[:, None] - win
            sc = np.where(allowed, sc, -1e30)
            p = np.exp(sc - sc.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            out[:, hd] = p @ v[:, kv]
        proj = out.reshape(seq, NH * dh) @ L["o"].T
        x = x + (rms(proj, L["post_attn"]) if c.get("post_norms") else proj)
        h = rms(x, L["ffn_norm"])
        act = gelu if c["act"] == "gelu" else silu
        ffn = (act(h @ L["gate"].T) * (h @ L["up"].T)) @ L["down"].T
        x = x + (rms(ffn, L["post_ffn"]) if c.get("post_norms") else ffn)
    x = rms(x, W["out_norm"])
    logits = x @ W["out"].T
    if c.get("final_softcap"):
        logits = c["final_softcap"] * np.tanh(logits / c["final_softcap"])
    return logits


def make_weights(rng, d_head, tied):
    s = 0.3
    W = {"emb": rng.normal(0, s, (VOCAB, D)), "out_norm": rng.uniform(0.5, 1.5, D), "layers": []}
    W["out"] = W["emb"] if tied else rng.normal(0, s, (VOCAB, D))
    for _ in range(NL):
        W["layers"].append({
            "attn_norm": rng.uniform(0.5, 1.5, D), "ffn_norm": rng.uniform(0.5, 1.5, D),
            "q": rng.normal(0, s, (NH * d_head, D)), "k": rng.normal(0, s, (NKV * d_head, D)),
            "v": rng.normal(0, s, (NKV * d_head, D)), "o": rng.normal(0, s, (D, NH * d_head)),
            "gate": rng.normal(0, s, (FFN, D)), "up": rng.normal(0, s, (FFN, D)),
            "down": rng.normal(0, s, (D, FFN)),
            "q_norm": rng.uniform(0.5, 1.5, d_head), "k_norm": rng.uniform(0.5, 1.5, d_head),
            "post_attn": rng.uniform(0.5, 1.5, D), "post_ffn": rng.uniform(0.5, 1.5, D)})
    return W


def requantize(W):
    """Round-trips every matrix through Q8_0 so the reference sees what the file holds."""
    def rt(a):
        a = np.asarray(a, dtype=np.float32)
        return gguf.quants.dequantize(gguf.quants.quantize(a, gguf.GGMLQuantizationType.Q8_0),
                                      gguf.GGMLQuantizationType.Q8_0).astype(np.float64)
    out = {**W, "emb": rt(W["emb"]), "layers": [{k: (rt(v) if k in ("q", "k", "v", "o", "gate", "up", "down") else v)
                                                 for k, v in L.items()} for L in W["layers"]]}
    out["out"] = out["emb"] if W["out"] is W["emb"] else rt(W["out"])
    return out


def add_vocab(w):
    """A minimal SentencePiece-style vocabulary so that llama.cpp can load these tiny models too."""
    w.add_tokenizer_model("llama")
    w.add_token_list(["<unk>", "<s>", "</s>"] + [f"t{i}" for i in range(3, VOCAB)])
    w.add_token_scores([0.0] * VOCAB)
    w.add_token_types([2, 3, 3] + [1] * (VOCAB - 3))
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)


def write(name, arch, W, c, fused=False, tied=False, meta=None, extra_tensors=None):
    if Q8:
        W = requantize(W)
    w = gguf.GGUFWriter(f"{OUT}/{PREFIX}{name}.gguf", arch=arch)
    w.add_uint32(f"{arch}.block_count", NL)
    w.add_uint32(f"{arch}.context_length", 512)
    w.add_uint32(f"{arch}.rope.dimension_count", c["d_head"])
    w.add_uint32(f"{arch}.embedding_length", D)
    w.add_uint32(f"{arch}.attention.head_count", NH)
    w.add_uint32(f"{arch}.attention.head_count_kv", NKV)
    w.add_uint32(f"{arch}.attention.key_length", c["d_head"])
    w.add_uint32(f"{arch}.attention.value_length", c["d_head"])
    w.add_uint32(f"{arch}.feed_forward_length", FFN)
    w.add_float32(f"{arch}.rope.freq_base", c["base"])
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", EPS)
    add_vocab(w)
    for k, v in (meta or {}).items():
        (w.add_float32 if isinstance(v, float) else w.add_uint32 if isinstance(v, int) else w.add_string)(f"{arch}.{k}", v)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    def mat(name, a):  # matrices are Q8_0 in the quantized set, F32 otherwise
        if Q8:
            w.add_tensor(name, gguf.quants.quantize(f32(a), gguf.GGMLQuantizationType.Q8_0),
                         raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, f32(a))
    mat("token_embd.weight", W["emb"])
    w.add_tensor("output_norm.weight", f32(W["out_norm"]))
    if not tied:
        mat("output.weight", W["out"])
    for l, L in enumerate(W["layers"]):
        p = f"blk.{l}"
        w.add_tensor(f"{p}.attn_norm.weight", f32(L["attn_norm"]))
        w.add_tensor(f"{p}.ffn_norm.weight", f32(L["ffn_norm"]))
        if c.get("post_norms"):
            w.add_tensor(f"{p}.post_attention_norm.weight", f32(L["post_attn"]))
            w.add_tensor(f"{p}.post_ffw_norm.weight", f32(L["post_ffn"]))
        if c.get("qk_norm"):
            w.add_tensor(f"{p}.attn_q_norm.weight", f32(L["q_norm"]))
            w.add_tensor(f"{p}.attn_k_norm.weight", f32(L["k_norm"]))
        mat(f"{p}.attn_output.weight", L["o"])
        mat(f"{p}.ffn_down.weight", L["down"])
        if fused:
            mat(f"{p}.attn_qkv.weight", np.concatenate([L["q"], L["k"], L["v"]]))
            mat(f"{p}.ffn_up.weight", np.concatenate([L["gate"], L["up"]]))
        else:
            for n in ("q", "k", "v"):
                mat(f"{p}.attn_{n}.weight", L[n])
            mat(f"{p}.ffn_gate.weight", L["gate"])
            mat(f"{p}.ffn_up.weight", L["up"])
    for n, a in (extra_tensors or {}).items():
        w.add_tensor(n, f32(a))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    logits = reference(W, c).astype("<f4")
    logits.tofile(f"{OUT}/{PREFIX}{name}.logits.f32")
    # greedy continuation of the prompt through full re-forwards (what --generate-tokens must reproduce)
    global TOKENS
    saved, gen = TOKENS, []
    for _ in range(N_GEN):
        gen.append(int(np.argmax(reference(W, c)[-1])))
        TOKENS = TOKENS + [gen[-1]]
    TOKENS = saved
    open(f"{OUT}/{PREFIX}{name}.generated.txt", "w").write(" ".join(map(str, gen)) + "\n")
    print(f"wrote {PREFIX}{name}: logits {logits.shape}, greedy {gen}")


def main():
    global D, FFN, Q8, PREFIX, NL
    run()
    D, FFN, Q8, PREFIX = 32, 64, True, "arch-q8-"
    run()


def run():
    global NL
    rng = np.random.default_rng(7)
    dh = D // NH
    base = dict(d_head=dh, base=10000.0, neox=True, act="silu", embd_scale=1.0)

    W = make_weights(rng, dh, False)
    write("plain", "qwen2", W, {**base, "rope": rope_table(dh, 10000.0)})

    write("yarn", "qwen2", W, {**base, "rope": rope_table(dh, 10000.0, "yarn", factor=4.0, orig=64, attn=1.0)},
          meta={"rope.scaling.type": "yarn", "rope.scaling.factor": 4.0,
                "rope.scaling.original_context_length": 64})

    write("linear-llama", "llama", W, {**base, "neox": False, "rope": rope_table(dh, 10000.0, "linear", factor=4.0)},
          meta={"rope.scaling.type": "linear", "rope.scaling.factor": 4.0})

    # llama arch (Mistral) with a sliding_window key: llama.cpp (checked with tools/arch_vs_llamacpp.py)
    # ignores it for this architecture, so the reference has no window and boring-llm must not apply one.
    write("llama-swa", "llama", W, {**base, "neox": False, "rope": rope_table(dh, 10000.0)},
          meta={"attention.sliding_window": 3})

    factors = rng.uniform(1.0, 3.0, dh // 2)
    write("phi3", "phi3", W, {**base, "rope": rope_table(dh, 10000.0, factors=factors, attn=1.2)}, fused=True,
          meta={"rope.scaling.attn_factor": 1.2}, extra_tensors={"rope_factors_short.weight": factors})

    # Phi-3 with both factor sets and an original context of 32: which set does llama.cpp pick at n_ctx 16 / 64?
    f_long = rng.uniform(2.0, 4.0, dh // 2)
    meta_lr = {"rope.scaling.attn_factor": 1.2, "rope.scaling.original_context_length": 1024}
    for nm, fac, att in (("phi3-short", factors, 1.2), ("phi3-long", f_long, 1.2)):
        write(nm, "phi3", W, {**base, "rope": rope_table(dh, 10000.0, factors=fac, attn=att)}, fused=True, meta=meta_lr,
              extra_tensors={"rope_factors_short.weight": factors, "rope_factors_long.weight": f_long})

    dg = D // 2  # Gemma: head size is not d_model / n_head (4 * 8 = 32 != 16)
    Wg = make_weights(rng, dg, True)
    write("gemma", "gemma", Wg, {"d_head": dg, "base": 10000.0, "neox": True, "act": "gelu",
                                  "embd_scale": math.sqrt(D), "rope": rope_table(dg, 10000.0)}, tied=True)

    # Gemma 2: post-norms, attention + logit soft-capping, sliding window on even layers
    gem2 = {"d_head": dg, "base": 10000.0, "neox": True, "act": "gelu", "embd_scale": math.sqrt(D),
            "rope": rope_table(dg, 10000.0), "post_norms": True, "attn_softcap": 2.0, "final_softcap": 3.0,
            "window": 3, "pattern": 2}
    Wg2 = make_weights(rng, dg, True)
    write("gemma2", "gemma2", Wg2, gem2, tied=True,
          meta={"attn_logit_softcapping": 2.0, "final_logit_softcapping": 3.0, "attention.sliding_window": 3})

    # Gemma 3: q/k norms, post-norms, window 3 on 5 of 6 layers with their own unscaled RoPE base,
    # the global layer (l = 5) uses the scaled one
    NL_saved = NL
    NL = 6
    gem3 = {"d_head": dg, "base": 10000.0, "neox": True, "act": "gelu", "embd_scale": math.sqrt(D),
            "rope": rope_table(dg, 10000.0, "linear", factor=2.0), "rope_local": rope_table(dg, 100.0),
            "post_norms": True, "qk_norm": True, "window": 3, "pattern": 6}
    Wg3 = make_weights(rng, dg, True)
    write("gemma3", "gemma3", Wg3, gem3, tied=True,
          meta={"attention.sliding_window": 3, "rope.freq_base_swa": 100.0,
                "rope.scaling.type": "linear", "rope.scaling.factor": 2.0})
    NL = NL_saved


def fmt_matrix(rng, qtype, a):
    """(raw bytes as a numpy array, the float64 matrix those bytes decode to) for one weight matrix."""
    T = gguf.GGMLQuantizationType
    a = np.asarray(a, dtype=np.float32)
    if qtype == T.F32:
        return a, a.astype(np.float64)
    if qtype == T.F16:
        h = a.astype(np.float16)
        return h, h.astype(np.float64)
    if qtype in (T.Q5_K, T.IQ4_XS, T.IQ4_NL):  # gguf.py cannot quantize these: random blocks with sane scales
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        raw = rng.integers(0, 256, size=(a.shape[0], a.shape[1] // block * size), dtype=np.uint8)
        view = raw.reshape(a.shape[0], -1, size)
        n = view.shape[0] * view.shape[1]
        sd = 0.004 if qtype == T.Q5_K else 0.0001
        view[:, :, 0:2] = rng.normal(0, sd, n).astype(np.float16).view(np.uint8).reshape(view.shape[0], -1, 2)
        view[:, :, 2:4] = rng.normal(0, sd, n).astype(np.float16).view(np.uint8).reshape(view.shape[0], -1, 2)
        return raw, gguf.quants.dequantize(raw, qtype).astype(np.float64)
    raw = gguf.quants.quantize(a, qtype)
    return raw, gguf.quants.dequantize(raw, qtype).astype(np.float64)


def run_formats():
    """One model per weight format (every matrix and the embeddings in that format):
    exercises the GPU kernels and the loaders of Q4_1, Q5_1, Q5_K, F16, BF16 and F32."""
    global D, FFN, NL, NH, NKV, Q8, PREFIX, TOKENS
    T = gguf.GGMLQuantizationType
    saved = (D, FFN, NL, NH, NKV, Q8, PREFIX)
    saved_tokens = TOKENS
    TOKENS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4]  # 20 rows: also takes the dequantize-then-matmul prefill path (>= 16 rows)
    D, FFN, NL, NH, NKV, Q8, PREFIX = 256, 256, 1, 2, 1, False, "fmt-"
    dh = 128
    for name, qtype in (("q4_0", T.Q4_0), ("q5_0", T.Q5_0), ("q4_1", T.Q4_1), ("q5_1", T.Q5_1), ("q5_k", T.Q5_K), ("iq4_xs", T.IQ4_XS), ("iq4_nl", T.IQ4_NL), ("f16", T.F16), ("bf16", T.BF16)):
        rng = np.random.default_rng(23)
        W = make_weights(rng, dh, False)
        raws = {}
        def conv(key, a):
            raw, deq = fmt_matrix(rng, qtype, a)
            raws[key] = raw
            return deq
        W["emb"] = conv("emb", W["emb"])
        W["out"] = conv("out", W["out"])
        for l, L in enumerate(W["layers"]):
            for k in ("q", "k", "v", "o", "gate", "up", "down"):
                L[k] = conv((l, k), L[k])
        c = {"d_head": dh, "base": 10000.0, "neox": True, "act": "silu", "embd_scale": 1.0, "rope": rope_table(dh, 10000.0)}
        arch = "qwen2"
        w = gguf.GGUFWriter(f"{OUT}/fmt-{name}.gguf", arch=arch)
        w.add_uint32(f"{arch}.block_count", NL); w.add_uint32(f"{arch}.embedding_length", D)
        w.add_uint32(f"{arch}.context_length", 512); w.add_uint32(f"{arch}.rope.dimension_count", dh)
        w.add_uint32(f"{arch}.attention.head_count", NH); w.add_uint32(f"{arch}.attention.head_count_kv", NKV)
        w.add_uint32(f"{arch}.attention.key_length", dh); w.add_uint32(f"{arch}.attention.value_length", dh)
        w.add_uint32(f"{arch}.feed_forward_length", FFN); w.add_float32(f"{arch}.rope.freq_base", 10000.0)
        w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", EPS)
        add_vocab(w)
        def mat(tname, key):
            raw = raws[key]
            if qtype in (T.F32, T.F16):
                w.add_tensor(tname, raw)
            else:
                w.add_tensor(tname, raw, raw_dtype=qtype)
        mat("token_embd.weight", "emb"); mat("output.weight", "out")
        w.add_tensor("output_norm.weight", np.asarray(W["out_norm"], dtype=np.float32))
        L = W["layers"][0]
        w.add_tensor("blk.0.attn_norm.weight", np.asarray(L["attn_norm"], dtype=np.float32))
        w.add_tensor("blk.0.ffn_norm.weight", np.asarray(L["ffn_norm"], dtype=np.float32))
        for tname, key in (("attn_q", "q"), ("attn_k", "k"), ("attn_v", "v"), ("attn_output", "o"),
                           ("ffn_gate", "gate"), ("ffn_up", "up"), ("ffn_down", "down")):
            mat(f"blk.0.{tname}.weight", (0, key))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        np.asarray(reference(W, c), dtype="<f4").tofile(f"{OUT}/fmt-{name}.logits.f32")
        print(f"wrote fmt-{name}")
    D, FFN, NL, NH, NKV, Q8, PREFIX = saved
    TOKENS = saved_tokens


if __name__ == "__main__":
    import sys
    if "--formats" in sys.argv:
        run_formats()
    else:
        main()
