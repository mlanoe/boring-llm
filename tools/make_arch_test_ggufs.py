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


def lnorm(x, w, b=None):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    out = (x - m) / np.sqrt(v + EPS) * w
    return out + b if b is not None else out


def norm_apply(x, w, b, use_ln):
    if use_ln:
        return lnorm(x, w, b)
    out = rms(x, w)
    return out + b if b is not None else out


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def silu(x):
    return x / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0.0, x)


def relu2(x):
    r = relu(x)
    return r * r


def corr_dim(d, orig, n_rot, base):
    return d * math.log(orig / (n_rot * 2 * math.pi)) / (2 * math.log(base))


def rope_table(d_head, base, kind="none", factor=1.0, orig=4096, bf=32.0, bs=1.0, attn=1.0, factors=None, n_rot=None):
    """(theta, mscale); theta has n_rot/2 entries: only the first n_rot dims of a head rotate (partial rotary)."""
    n_rot = n_rot or d_head
    half = n_rot // 2
    theta = np.array([base ** (-2.0 * j / n_rot) for j in range(half)])
    mscale = 1.0
    if factors is not None:
        theta = theta / factors
        mscale = attn
    elif kind == "linear":
        theta = theta / factor
    elif kind == "yarn":
        low = max(0, math.floor(corr_dim(n_rot, orig, bf, base)))
        high = min(n_rot - 1, math.ceil(corr_dim(n_rot, orig, bs, base)))
        out = []
        for j in range(half):
            y = (j - low) / max(0.001, high - low)
            ramp_mix = 1 - min(1, max(0, y))
            out.append(theta[j] / factor * (1 - ramp_mix) + theta[j] * ramp_mix)
        theta = np.array(out)
        mscale = attn * (1 + 0.1 * math.log(factor))
    return theta, mscale


def rope(x, pos, theta, mscale, neox):  # x: (seq, heads, d_head); only the first 2*len(theta) dims rotate
    n_rot = 2 * len(theta)
    half = n_rot // 2
    xr, rest = x[..., :n_rot], x[..., n_rot:]
    ang = pos[:, None] * theta[None, :]
    c, s = np.cos(ang)[:, None, :] * mscale, np.sin(ang)[:, None, :] * mscale
    out = np.empty_like(xr)
    if neox:
        a, b = xr[..., :half], xr[..., half:]
        out[..., :half], out[..., half:] = a * c - b * s, a * s + b * c
    else:
        a, b = xr[..., 0::2], xr[..., 1::2]
        out[..., 0::2], out[..., 1::2] = a * c - b * s, a * s + b * c
    return np.concatenate([out, rest], axis=-1)


def reference(W, c):
    seq = len(TOKENS)
    pos = np.arange(seq, dtype=np.float64)
    x = W["emb"][TOKENS] * c["embd_scale"]
    if "pos_embd" in W:
        x = x + W["pos_embd"][np.arange(seq)]
    dh = c["d_head"]
    use_ln = c.get("layernorm", False)
    res_type = c.get("residual_type", 0)
    for l in range(NL):
        L = W["layers"][l]
        pre_router_logits = x @ L["router"].T if ("router" in L and c.get("pre_attn_router")) else None
        h = x if c.get("no_pre_norm") else norm_apply(x, L["attn_norm"], L.get("attn_norm_b"), use_ln)
        q_flat = h @ L["q"].T + L.get("q_b", 0.0)
        k_flat = h @ L["k"].T + L.get("k_b", 0.0)
        if c.get("qk_norm_full"):
            q_flat, k_flat = rms(q_flat, L["q_norm_full"]), rms(k_flat, L["k_norm_full"])
        q = q_flat.reshape(seq, NH, dh)
        k = k_flat.reshape(seq, NKV, dh)
        v = (h @ L["v"].T + L.get("v_b", 0.0)).reshape(seq, NKV, dh)
        if c.get("qk_norm") and not c.get("qk_norm_post_rope"):
            q, k = norm_apply(q, L["q_norm"], None, use_ln), norm_apply(k, L["k_norm"], None, use_ln)
        win = c.get("window", 0)
        pat = c.get("pattern", 4)
        swa = win > 0 and (c.get("all_swa") or ((l % pat) > 0 if c.get("swa_first_global") else (l % pat) < pat - 1))
        tab = c["rope_local"] if swa and "rope_local" in c else c["rope"]
        is_nope = c.get("no_rope_all") or (c.get("nope") and ((l % pat == 0) if c.get("swa_first_global") else (l + 1) % 4 == 0))
        if not is_nope:
            q, k = rope(q, pos, *tab, c["neox"]), rope(k, pos, *tab, c["neox"])
        if c.get("qk_norm") and c.get("qk_norm_post_rope"):
            q, k = norm_apply(q, L["q_norm"], None, use_ln), norm_apply(k, L["k_norm"], None, use_ln)
        out = np.zeros((seq, NH, dh))
        for hd in range(NH):
            kv = hd // (NH // NKV)
            sc = q[:, hd] @ k[:, kv].T * c.get("attn_scale", 1 / math.sqrt(dh))
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
        out_flat = out.reshape(seq, NH * dh)
        if "attn_gate" in L:
            ag = 1.0 / (1.0 + np.exp(-(h @ L["attn_gate"].T)))
            out_flat = out_flat * ag
        proj = (out_flat @ L["o"].T + L.get("o_b", 0.0)) * c.get("residual_scale", 1.0)
        attn_res = rms(proj, L["post_attn"]) if c.get("post_norms") else proj
        if res_type == 1:
            h2 = h
        elif res_type == 2:
            h2 = norm_apply(x, L["ffn_norm"], L.get("ffn_norm_b"), use_ln)
        else:
            x = x + attn_res
            h2 = x if c.get("no_pre_norm") else norm_apply(x, L["ffn_norm"], L.get("ffn_norm_b"), use_ln)
        act_name = c["act"]
        act = gelu if act_name == "gelu" else relu2 if act_name == "relu2" else relu if act_name == "relu" else silu
        if "router" in L:  # mixture of experts, as llama.cpp's build_moe_ffn: softmax/sigmoid/identity gate, top-k, optionally renormalized + shared expert
            gate_logits = pre_router_logits if pre_router_logits is not None else (h2 @ L["router"].T + L.get("router_b", 0.0))
            gf = c.get("moe_gating_func")
            if gf == 2:
                probs = 1.0 / (1.0 + np.exp(-gate_logits))
            elif gf == 3:
                probs = gate_logits
            else:
                probs = np.exp(gate_logits - gate_logits.max(-1, keepdims=True))
                probs /= probs.sum(-1, keepdims=True)
            sel_probs = probs + L["exp_probs_b"] if "exp_probs_b" in L else probs
            ffn = np.zeros_like(h2)
            norm_topk = c.get("moe_norm_weights", True)
            w_scale = c.get("moe_weights_scale", 1.0)
            for t in range(seq):
                top = np.argsort(-sel_probs[t])[: c["n_used"]]
                wts = probs[t, top]
                if norm_topk:
                    wts = wts / wts.sum()
                wts = wts * w_scale
                for e, wt in zip(top, wts):
                    ffn[t] += wt * ((act(h2[t] @ L["eg"][e].T) * (h2[t] @ L["eu"][e].T)) @ L["ed"][e].T)
            if "sh_gate" in L:
                sh = (act(h2 @ L["sh_gate"].T) * (h2 @ L["sh_up"].T)) @ L["sh_down"].T
                if "sh_inp" in L:
                    sh_g = 1.0 / (1.0 + np.exp(-(h2 @ L["sh_inp"])))
                    sh = sh * sh_g[:, None]
                ffn = ffn + sh
        elif c.get("ffn_gated", True):
            ffn = (act(h2 @ L["gate"].T) * (h2 @ L["up"].T)) @ L["down"].T
        else:
            ffn = act(h2 @ L["up"].T + L.get("up_b", 0.0)) @ L["down"].T + L.get("down_b", 0.0)
        ffn = ffn * c.get("residual_scale", 1.0)
        ffn_res = rms(ffn, L["post_ffn"]) if c.get("post_norms") else ffn
        if res_type != 0:
            x = x + attn_res + ffn_res
        else:
            x = x + ffn_res
    x = norm_apply(x, W["out_norm"], W.get("out_norm_b"), use_ln)
    logits = (x @ W["out"].T + W.get("out_b", 0.0)) * c.get("logit_mul", 1.0)
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


def add_moe(rng, W, n_expert, ffn_e, sh_ffn=0, sh_gate=False, start_layer=0, exp_bias=False):
    """Replaces layers[start_layer:]'s dense FFN by `n_expert` experts of width `ffn_e` and a router (plus optional shared expert)."""
    for L in W["layers"][start_layer:]:
        L["router"] = rng.normal(0, 0.6, (n_expert, D))
        L["eg"] = rng.normal(0, 0.3, (n_expert, ffn_e, D))
        L["eu"] = rng.normal(0, 0.3, (n_expert, ffn_e, D))
        L["ed"] = rng.normal(0, 0.3, (n_expert, D, ffn_e))
        if exp_bias:
            L["exp_probs_b"] = rng.normal(0, 0.1, (n_expert,))
        if sh_ffn > 0:
            L["sh_gate"] = rng.normal(0, 0.3, (sh_ffn, D))
            L["sh_up"] = rng.normal(0, 0.3, (sh_ffn, D))
            L["sh_down"] = rng.normal(0, 0.3, (D, sh_ffn))
            if sh_gate:
                L["sh_inp"] = rng.normal(0, 0.5, (D,))


def requantize(W):
    """Round-trips every matrix through Q8_0 so the reference sees what the file holds."""
    def rt(a):
        a = np.asarray(a, dtype=np.float32)
        return gguf.quants.dequantize(gguf.quants.quantize(a, gguf.GGMLQuantizationType.Q8_0),
                                      gguf.GGMLQuantizationType.Q8_0).astype(np.float64)
    out = {**W, "emb": rt(W["emb"]), "layers": [{k: (rt(v) if k in ("q", "k", "v", "o", "attn_gate", "gate", "up", "down", "eg", "eu", "ed", "sh_gate", "sh_up", "sh_down") else v)
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


def add_biases(rng, W, d_head, norm_b=True, qkv_b=True, o_b=True, ffn_b=True, out_b=False):
    s = 0.08
    if norm_b:
        W["out_norm_b"] = rng.normal(0, s, D)
    if out_b:
        W["out_b"] = rng.normal(0, s, VOCAB)
    for L in W["layers"]:
        if norm_b:
            L["attn_norm_b"] = rng.normal(0, s, D)
            L["ffn_norm_b"] = rng.normal(0, s, D)
        if qkv_b:
            L["q_b"] = rng.normal(0, s, NH * d_head)
            L["k_b"] = rng.normal(0, s, NKV * d_head)
            L["v_b"] = rng.normal(0, s, NKV * d_head)
        if o_b:
            L["o_b"] = rng.normal(0, s, D)
        if ffn_b:
            L["up_b"] = rng.normal(0, s, FFN)
            L["down_b"] = rng.normal(0, s, D)


def write(name, arch, W, c, fused=False, fused_qkv_only=False, tied=False, meta=None, extra_tensors=None):
    if Q8:
        W = requantize(W)
    w = gguf.GGUFWriter(f"{OUT}/{PREFIX}{name}.gguf", arch=arch)
    w.add_uint32(f"{arch}.block_count", NL)
    w.add_uint32(f"{arch}.context_length", 512)
    w.add_uint32(f"{arch}.rope.dimension_count", c.get("n_rot", c["d_head"]))
    w.add_uint32(f"{arch}.embedding_length", D)
    w.add_uint32(f"{arch}.attention.head_count", NH)
    w.add_uint32(f"{arch}.attention.head_count_kv", NKV)
    w.add_uint32(f"{arch}.attention.key_length", c["d_head"])
    w.add_uint32(f"{arch}.attention.value_length", c["d_head"])
    w.add_uint32(f"{arch}.feed_forward_length", FFN)
    w.add_float32(f"{arch}.rope.freq_base", c["base"])
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", EPS)
    if c.get("layernorm"):
        w.add_float32(f"{arch}.attention.layer_norm_epsilon", EPS)
    add_vocab(w)
    for k, v in (meta or {}).items():
        (w.add_bool if isinstance(v, bool) else w.add_float32 if isinstance(v, float) else w.add_uint32 if isinstance(v, int) else w.add_string)(f"{arch}.{k}", v)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    def mat(name, a):  # matrices are Q8_0 in the quantized set, F32 otherwise
        if Q8:
            w.add_tensor(name, gguf.quants.quantize(f32(a), gguf.GGMLQuantizationType.Q8_0),
                         raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, f32(a))
    mat("token_embd.weight", W["emb"])
    if "pos_embd" in W:
        w.add_tensor("position_embd.weight", f32(W["pos_embd"]))
    if not c.get("weightless_norm"):
        w.add_tensor("output_norm.weight", f32(W["out_norm"]))
    if "out_norm_b" in W:
        w.add_tensor("output_norm.bias", f32(W["out_norm_b"]))
    if not tied:
        mat("output.weight", W["out"])
    if "out_b" in W:
        w.add_tensor("output.bias", f32(W["out_b"]))
    for l, L in enumerate(W["layers"]):
        p = f"blk.{l}"
        if not c.get("no_pre_norm") and not c.get("weightless_norm"):
            if c.get("falcon_norm2"):
                w.add_tensor(f"{p}.attn_norm_2.weight", f32(L["attn_norm"]))
                if "attn_norm_b" in L:
                    w.add_tensor(f"{p}.attn_norm_2.bias", f32(L["attn_norm_b"]))
                w.add_tensor(f"{p}.attn_norm.weight", f32(L["ffn_norm"]))
                if "ffn_norm_b" in L:
                    w.add_tensor(f"{p}.attn_norm.bias", f32(L["ffn_norm_b"]))
            else:
                w.add_tensor(f"{p}.attn_norm.weight", f32(L["attn_norm"]))
                if "attn_norm_b" in L:
                    w.add_tensor(f"{p}.attn_norm.bias", f32(L["attn_norm_b"]))
                if c.get("residual_type", 0) != 1:
                    ffn_norm_tname = f"{p}.post_attention_norm" if c.get("post_as_ffn") else f"{p}.ffn_norm"
                    w.add_tensor(f"{ffn_norm_tname}.weight", f32(L["ffn_norm"]))
                    if "ffn_norm_b" in L:
                        w.add_tensor(f"{ffn_norm_tname}.bias", f32(L["ffn_norm_b"]))
        if c.get("post_norms"):
            w.add_tensor(f"{p}.post_attention_norm.weight", f32(L["post_attn"]))
            w.add_tensor(f"{p}.post_ffw_norm.weight", f32(L["post_ffn"]))
        if c.get("qk_norm"):
            w.add_tensor(f"{p}.attn_q_norm.weight", f32(L["q_norm"]))
            w.add_tensor(f"{p}.attn_k_norm.weight", f32(L["k_norm"]))
        if c.get("qk_norm_full"):
            w.add_tensor(f"{p}.attn_q_norm.weight", f32(L["q_norm_full"]))
            w.add_tensor(f"{p}.attn_k_norm.weight", f32(L["k_norm_full"]))
        if "attn_gate" in L:
            mat(f"{p}.attn_gate.weight", L["attn_gate"])
        mat(f"{p}.attn_output.weight", L["o"])
        if "o_b" in L:
            w.add_tensor(f"{p}.attn_output.bias", f32(L["o_b"]))
        if "router" in L:
            w.add_tensor(f"{p}.ffn_gate_inp.weight", f32(L["router"]))
            if "exp_probs_b" in L:
                w.add_tensor(f"{p}.exp_probs_b.bias", f32(L["exp_probs_b"]))
            mat(f"{p}.ffn_gate_exps.weight", L["eg"])
            mat(f"{p}.ffn_up_exps.weight", L["eu"])
            mat(f"{p}.ffn_down_exps.weight", L["ed"])
            if "sh_gate" in L:
                mat(f"{p}.ffn_gate_shexp.weight", L["sh_gate"])
                mat(f"{p}.ffn_up_shexp.weight", L["sh_up"])
                mat(f"{p}.ffn_down_shexp.weight", L["sh_down"])
                if "sh_inp" in L:
                    w.add_tensor(f"{p}.ffn_gate_inp_shexp.weight", f32(L["sh_inp"]))
            if fused_qkv_only:
                mat(f"{p}.attn_qkv.weight", np.concatenate([L["q"], L["k"], L["v"]]))
                if "q_b" in L:
                    w.add_tensor(f"{p}.attn_qkv.bias", f32(np.concatenate([L["q_b"], L["k_b"], L["v_b"]])))
            else:
                for n in ("q", "k", "v"):
                    mat(f"{p}.attn_{n}.weight", L[n])
                    if f"{n}_b" in L:
                        w.add_tensor(f"{p}.attn_{n}.bias", f32(L[f"{n}_b"]))
            continue
        mat(f"{p}.ffn_down.weight", L["down"])
        if "down_b" in L:
            w.add_tensor(f"{p}.ffn_down.bias", f32(L["down_b"]))
        if fused:
            mat(f"{p}.attn_qkv.weight", np.concatenate([L["q"], L["k"], L["v"]]))
            if "q_b" in L:
                w.add_tensor(f"{p}.attn_qkv.bias", f32(np.concatenate([L["q_b"], L["k_b"], L["v_b"]])))
            mat(f"{p}.ffn_up.weight", np.concatenate([L["gate"], L["up"]]))
        elif fused_qkv_only:
            mat(f"{p}.attn_qkv.weight", np.concatenate([L["q"], L["k"], L["v"]]))
            if "q_b" in L:
                w.add_tensor(f"{p}.attn_qkv.bias", f32(np.concatenate([L["q_b"], L["k_b"], L["v_b"]])))
            if c.get("ffn_gated", True):
                mat(f"{p}.ffn_gate.weight", L["gate"])
            mat(f"{p}.ffn_up.weight", L["up"])
            if "up_b" in L:
                w.add_tensor(f"{p}.ffn_up.bias", f32(L["up_b"]))
        else:
            for n in ("q", "k", "v"):
                mat(f"{p}.attn_{n}.weight", L[n])
                if f"{n}_b" in L:
                    w.add_tensor(f"{p}.attn_{n}.bias", f32(L[f"{n}_b"]))
            if c.get("ffn_gated", True):
                mat(f"{p}.ffn_gate.weight", L["gate"])
            mat(f"{p}.ffn_up.weight", L["up"])
            if "up_b" in L:
                w.add_tensor(f"{p}.ffn_up.bias", f32(L["up_b"]))
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
    global NL, NKV
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

    # Mixture of experts: Mixtral (llama arch with expert_count) and Qwen3-MoE (qk-norm, per-expert width)
    Wm = make_weights(rng, dh, False)
    add_moe(rng, Wm, 4, FFN)
    write("mixtral", "llama", Wm, {**base, "neox": False, "rope": rope_table(dh, 10000.0), "n_used": 2},
          meta={"expert_count": 4, "expert_used_count": 2})
    Wq = make_weights(rng, dh, False)
    add_moe(rng, Wq, 6, FFN // 2)
    write("qwen3moe", "qwen3moe", Wq, {**base, "qk_norm": True, "rope": rope_table(dh, 10000.0), "n_used": 3},
          meta={"expert_count": 6, "expert_used_count": 3, "expert_feed_forward_length": FFN // 2})

    # Granite: explicit embedding / attention / residual / logit scales (NORM rope, llama-like tensors)
    write("granite", "granite", W, {**base, "neox": False, "rope": rope_table(dh, 10000.0), "embd_scale": 2.0,
                                    "attn_scale": 0.31, "residual_scale": 0.6, "logit_mul": 1 / 4.0},
          meta={"embedding_scale": 2.0, "attention.scale": 0.31, "residual_scale": 0.6, "logit_scale": 4.0})
    # SmolLM3: every 4th layer has no rotary embedding (needs 4 layers)
    NL_s = NL
    NL = 4
    Wn = make_weights(rng, dh, False)
    write("smollm3", "smollm3", Wn, {**base, "neox": False, "rope": rope_table(dh, 10000.0), "nope": True})
    NL = NL_s

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

    # Qwen2-MoE: NEOX RoPE, routed experts (unnormalized top-k) + sigmoid-gated shared expert
    rng = np.random.default_rng(10)
    Wq2m = make_weights(rng, dh, False)
    add_moe(rng, Wq2m, 4, FFN // 2, sh_ffn=FFN // 2, sh_gate=True)
    write("qwen2moe", "qwen2moe", Wq2m,
          {**base, "rope": rope_table(dh, 10000.0), "n_used": 2, "moe_norm_weights": False},
          meta={"expert_count": 4, "expert_used_count": 2,
                "expert_feed_forward_length": FFN // 2, "expert_shared_feed_forward_length": FFN // 2})

    # OLMoE: NEOX RoPE, MHA (n_kv_head == n_head in llama.cpp), full-projection Q/K RMSNorm (width D), unnormalized top-k routing
    NKV_saved = NKV
    NKV = NH
    Wolm = make_weights(rng, dh, False)
    for L in Wolm["layers"]:
        L["q_norm_full"] = rng.uniform(0.5, 1.5, NH * dh)
        L["k_norm_full"] = rng.uniform(0.5, 1.5, NKV * dh)
    add_moe(rng, Wolm, 4, FFN)
    write("olmoe", "olmoe", Wolm,
          {**base, "qk_norm_full": True, "rope": rope_table(dh, 10000.0), "n_used": 2, "moe_norm_weights": False},
          meta={"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": FFN})
    NKV = NKV_saved

    # Granite-MoE: NORM RoPE, Granite scales, routed experts (normalized top-k) + ungated shared expert
    Wgm = make_weights(rng, dh, False)
    add_moe(rng, Wgm, 4, FFN, sh_ffn=FFN // 2, sh_gate=False)
    write("granitemoe", "granitemoe", Wgm,
          {**base, "neox": False, "rope": rope_table(dh, 10000.0), "n_used": 2, "moe_norm_weights": True,
           "embd_scale": 2.0, "attn_scale": 0.31, "residual_scale": 0.6, "logit_mul": 1 / 4.0},
          meta={"expert_count": 4, "expert_used_count": 2,
                "expert_shared_feed_forward_length": FFN // 2,
                "embedding_scale": 2.0, "attention.scale": 0.31, "residual_scale": 0.6, "logit_scale": 4.0})

    # DeepSeek-MoE (v1): NORM RoPE, leading dense layer 0 + MoE layer 1 with shared experts, unnormalized top-k
    rng = np.random.default_rng(12)
    Wds = make_weights(rng, dh, False)
    add_moe(rng, Wds, 4, FFN // 2, sh_ffn=FFN, sh_gate=False, start_layer=1)
    write("deepseek", "deepseek", Wds,
          {**base, "neox": False, "rope": rope_table(dh, 10000.0), "n_used": 2, "moe_norm_weights": False},
          meta={"leading_dense_block_count": 1, "expert_count": 4, "expert_used_count": 2,
                "expert_shared_count": 2, "expert_feed_forward_length": FFN // 2})

    # OLMo 2: NEOX RoPE, full-row Q/K RMSNorm (width n_head*d_head / n_kv_head*d_head), post-norms only (no pre-norms)
    rng = np.random.default_rng(14)
    Wol2 = make_weights(rng, dh, False)
    for L in Wol2["layers"]:
        L["q_norm_full"] = rng.uniform(0.5, 1.5, NH * dh)
        L["k_norm_full"] = rng.uniform(0.5, 1.5, NKV * dh)
    write("olmo2", "olmo2", Wol2,
          {**base, "qk_norm_full": True, "post_norms": True, "no_pre_norm": True, "rope": rope_table(dh, 10000.0)})

    # EXAONE 3.0 / 3.5: NEOX RoPE, standard pre-norm RMSNorm + SwiGLU
    Wex = make_weights(rng, dh, False)
    write("exaone", "exaone", Wex, {**base, "rope": rope_table(dh, 10000.0)})

    # EXAONE 4.0: NEOX RoPE, per-head Q/K RMSNorm, post-norms only (no pre-norms)
    Wex4 = make_weights(rng, dh, False)
    write("exaone4", "exaone4", Wex4,
          {**base, "qk_norm": True, "post_norms": True, "no_pre_norm": True, "rope": rope_table(dh, 10000.0)})

    # InternLM2, Mistral3: NORM RoPE, standard pre-norm RMSNorm + SwiGLU
    Win = make_weights(rng, dh, False)
    write("internlm2", "internlm2", Win, {**base, "neox": False, "rope": rope_table(dh, 10000.0)})
    write("mistral3", "mistral3", Win, {**base, "neox": False, "rope": rope_table(dh, 10000.0)})

    # GLM-4 (0414): NORM RoPE, both pre-norms and post-norms, fused QKV + gate/up
    rng = np.random.default_rng(16)
    Wglm = make_weights(rng, dh, False)
    write("glm4", "glm4", Wglm, {**base, "neox": False, "post_norms": True, "rope": rope_table(dh, 10000.0)},
          fused=True)
    # Partial rotary (rope.dimension_count = d_head / 2, as GLM-4 ships): NORM pairs (2j, 2j+1) over the first
    # n_rot dims, the rest of every head passes through; theta_j = base^(-2j / n_rot).
    write("partial-glm4", "glm4", Wglm, {**base, "neox": False, "post_norms": True, "n_rot": dh // 2,
          "rope": rope_table(dh, 10000.0, n_rot=dh // 2)}, fused=True)

    # MiniCPM: NORM RoPE, Granite-like scaling (embedding_scale, residual_scale, logit_scale)
    Wmc = make_weights(rng, dh, False)
    write("minicpm", "minicpm", Wmc,
          {**base, "neox": False, "rope": rope_table(dh, 10000.0),
           "embd_scale": 3.0, "residual_scale": 0.7, "logit_mul": 1.0 / 2.0},
          meta={"embedding_scale": 3.0, "residual_scale": 0.7, "logit_scale": 2.0})

    # EXAONE-MoE: NEOX RoPE, per-head Q/K RMSNorm, leading dense block + sigmoid-gated MoE with expert bias & shared expert
    Wexm = make_weights(rng, dh, False)
    add_moe(rng, Wexm, 4, FFN // 2, sh_ffn=FFN // 2, sh_gate=False, start_layer=1, exp_bias=True)
    write("exaone-moe", "exaone-moe", Wexm,
          {**base, "qk_norm": True, "rope": rope_table(dh, 10000.0), "n_used": 2,
           "moe_gating_func": 2, "moe_norm_weights": True, "moe_weights_scale": 1.5},
          meta={"leading_dense_block_count": 1, "expert_count": 4, "expert_used_count": 2,
                "expert_shared_count": 1, "expert_feed_forward_length": FFN // 2,
                "expert_shared_feed_forward_length": FFN // 2, "expert_gating_func": 2,
                "expert_weights_norm": True, "expert_weights_scale": 1.5,
                "attention.sliding_window": 16, "attention.sliding_window_pattern": 4})

    # Phi-2: LayerNorm with bias, single-norm parallel residual, non-gated GELU FFN, full linear biases
    rng = np.random.default_rng(18)
    Wphi2 = make_weights(rng, dh, False)
    add_biases(rng, Wphi2, dh, norm_b=True, qkv_b=True, o_b=True, ffn_b=True, out_b=True)
    write("phi2", "phi2", Wphi2,
          {**base, "act": "gelu", "layernorm": True, "residual_type": 1, "ffn_gated": False,
           "rope": rope_table(dh, 10000.0)})

    # StarCoder2: LayerNorm with bias, sequential residual, non-gated GELU FFN, linear biases
    Wsc2 = make_weights(rng, dh, False)
    add_biases(rng, Wsc2, dh, norm_b=True, qkv_b=True, o_b=True, ffn_b=True, out_b=False)
    write("starcoder2", "starcoder2", Wsc2,
          {**base, "act": "gelu", "layernorm": True, "residual_type": 0, "ffn_gated": False,
           "rope": rope_table(dh, 10000.0)})

    # GPT-NeoX: LayerNorm with bias, dual-norm parallel residual, fused QKV + bias, non-gated GELU FFN
    Wnx = make_weights(rng, dh, False)
    add_biases(rng, Wnx, dh, norm_b=True, qkv_b=True, o_b=True, ffn_b=True, out_b=False)
    write("gptneox", "gptneox", Wnx,
          {**base, "act": "gelu", "layernorm": True, "residual_type": 2, "ffn_gated": False,
           "rope": rope_table(dh, 10000.0)},
          fused_qkv_only=True, meta={"use_parallel_residual": True})
    # Partial rotary as GPT-NeoX ships it (rotary_pct): NEOX pairs (j, j + n_rot/2) over the first n_rot dims,
    # the tail passes through; with YaRN the correction dims are computed from n_rot.
    nr = dh // 2
    write("partial-gptneox", "gptneox", Wnx,
          {**base, "act": "gelu", "layernorm": True, "residual_type": 2, "ffn_gated": False, "n_rot": nr,
           "rope": rope_table(dh, 10000.0, n_rot=nr)},
          fused_qkv_only=True, meta={"use_parallel_residual": True})
    write("partial-gptneox-yarn", "gptneox", Wnx,
          {**base, "act": "gelu", "layernorm": True, "residual_type": 2, "ffn_gated": False, "n_rot": nr,
           "rope": rope_table(dh, 10000.0, "yarn", factor=4.0, orig=64, attn=1.0, n_rot=nr)},
          fused_qkv_only=True, meta={"use_parallel_residual": True, "rope.scaling.type": "yarn",
                                     "rope.scaling.factor": 4.0, "rope.scaling.original_context_length": 64})

    # Falcon: LayerNorm with bias, dual-norm parallel residual via attn_norm_2, fused QKV, non-gated GELU FFN (no linear bias)
    Wfal = make_weights(rng, dh, False)
    add_biases(rng, Wfal, dh, norm_b=True, qkv_b=False, o_b=False, ffn_b=False, out_b=False)
    write("falcon", "falcon", Wfal,
          {**base, "act": "gelu", "layernorm": True, "residual_type": 2, "falcon_norm2": True,
           "ffn_gated": False, "rope": rope_table(dh, 10000.0)},
          fused_qkv_only=True)

    # Command-R: LayerNorm without bias, single-norm parallel residual, NORM RoPE, SwiGLU, logit_scale
    Wcr = make_weights(rng, dh, True)
    write("command-r", "command-r", Wcr,
          {**base, "neox": False, "layernorm": True, "residual_type": 1, "logit_mul": 0.5,
           "rope": rope_table(dh, 10000.0)},
          tied=True, meta={"logit_scale": 0.5})

    # Cohere2 (Command R7B / Command A): LayerNorm without bias, single-norm parallel residual, SWA on 3/4 layers + NoPE on 4th
    NL_c = NL
    NL = 4
    Wch2 = make_weights(rng, dh, True)
    write("cohere2", "cohere2", Wch2,
          {**base, "neox": False, "layernorm": True, "residual_type": 1, "logit_mul": 0.5,
           "window": 3, "pattern": 4, "nope": True, "rope": rope_table(dh, 10000.0)},
          tied=True, meta={"attention.sliding_window": 3, "logit_scale": 0.5})
    NL = NL_c

    # PhiMoE: RMSNorm with bias, sequential residual, attention + output linear biases, routed MoE experts
    Wpm = make_weights(rng, dh, False)
    add_biases(rng, Wpm, dh, norm_b=True, qkv_b=True, o_b=True, ffn_b=False, out_b=True)
    add_moe(rng, Wpm, 4, FFN)
    write("phimoe", "phimoe", Wpm,
          {**base, "layernorm": False, "residual_type": 0, "n_used": 2, "moe_norm_weights": True,
           "rope": rope_table(dh, 10000.0)},
          meta={"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": FFN})

    # DOTS1: NEOX RoPE, per-head Q/K RMSNorm, leading dense layer, routed MoE + exp_probs_b + shared expert
    # (llama.cpp's dots1 loader checks MHA Q/K/V dimensions n_head * d_head for all three)
    NKV_d1 = NKV
    NKV = NH
    Wd1 = make_weights(rng, dh, False)
    add_moe(rng, Wd1, 4, FFN // 2, sh_ffn=FFN // 2, start_layer=1, exp_bias=True)
    write("dots1", "dots1", Wd1,
          {**base, "qk_norm": True, "n_used": 2, "moe_gating_func": 2, "moe_norm_weights": False,
           "moe_weights_scale": 1.0, "rope": rope_table(dh, 10000.0)},
          meta={"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": FFN // 2,
                "expert_shared_count": 1, "leading_dense_block_count": 1, "expert_gating_func": 2})
    NKV = NKV_d1

    # ERNIE 4.5 (dense): NORM RoPE, RMSNorm + SwiGLU
    We45 = make_weights(rng, dh, False)
    write("ernie4_5", "ernie4_5", We45,
          {**base, "neox": False, "rope": rope_table(dh, 10000.0)})

    # ERNIE 4.5 MoE: NORM RoPE, leading dense layer, routed MoE + exp_probs_b + shared expert (hardcoded norm_w=True, softmax)
    We45m = make_weights(rng, dh, False)
    add_moe(rng, We45m, 4, FFN // 2, sh_ffn=FFN // 2, start_layer=1, exp_bias=True)
    write("ernie4_5-moe", "ernie4_5-moe", We45m,
          {**base, "neox": False, "n_used": 2, "moe_gating_func": 1, "moe_norm_weights": True,
           "rope": rope_table(dh, 10000.0)},
          meta={"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": FFN // 2,
                "expert_shared_feed_forward_length": FFN // 2, "leading_dense_block_count": 1,
                "interleave_moe_layer_step": 1})

    # Hunyuan Dense: NEOX RoPE, per-head Q/K RMSNorm applied AFTER RoPE
    Whd = make_weights(rng, dh, False)
    write("hunyuan-dense", "hunyuan-dense", Whd,
          {**base, "qk_norm": True, "qk_norm_post_rope": True, "rope": rope_table(dh, 10000.0)})

    # Hunyuan MoE: NEOX RoPE, post-RoPE Q/K RMSNorm, routed MoE + shared expert (norm_w=True, softmax)
    Whm = make_weights(rng, dh, False)
    add_moe(rng, Whm, 4, FFN, sh_ffn=FFN // 2)
    write("hunyuan-moe", "hunyuan-moe", Whm,
          {**base, "qk_norm": True, "qk_norm_post_rope": True, "n_used": 2,
           "moe_gating_func": 1, "moe_norm_weights": True, "rope": rope_table(dh, 10000.0)},
          meta={"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": FFN,
                "expert_shared_feed_forward_length": FFN // 2})

    # GLM-4 MoE: NEOX RoPE, Q/K RMSNorm, post_attention_norm as pre-FFN norm, leading dense layer, sigmoid MoE + exp_probs_b + shared expert
    Wgm = make_weights(rng, dh, False)
    add_moe(rng, Wgm, 4, FFN // 2, sh_ffn=FFN // 2, start_layer=1, exp_bias=True)
    write("glm4moe", "glm4moe", Wgm,
          {**base, "qk_norm": True, "post_as_ffn": True, "n_used": 2, "moe_gating_func": 2,
           "moe_norm_weights": False, "rope": rope_table(dh, 10000.0)},
          meta={"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": FFN // 2,
                "expert_shared_count": 1, "leading_dense_block_count": 1,
                "expert_group_count": 1, "expert_group_used_count": 1})

    # BailingMoE2: NEOX RoPE, fused attn_qkv, Q/K RMSNorm, leading dense layer, routed MoE + exp_probs_b + shared expert
    Wbm2 = make_weights(rng, dh, False)
    add_moe(rng, Wbm2, 4, FFN // 2, sh_ffn=FFN // 2, start_layer=1, exp_bias=True)
    write("bailingmoe2", "bailingmoe2", Wbm2,
          {**base, "qk_norm": True, "n_used": 2, "moe_gating_func": 2, "moe_norm_weights": False,
           "rope": rope_table(dh, 10000.0)},
          fused_qkv_only=True,
          meta={"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": FFN // 2,
                "expert_shared_count": 1, "leading_dense_block_count": 1, "expert_gating_func": 2})

    # Arcee: NORM RoPE, RMSNorm, non-gated relu(x)^2 FFN (ffn_up + ffn_down, no gate)
    Warc = make_weights(rng, dh, False)
    write("arcee", "arcee", Warc,
          {**base, "neox": False, "ffn_gated": False, "act": "relu2", "rope": rope_table(dh, 10000.0)})

    # Seed-OSS: NEOX RoPE, RMSNorm, post_attention_norm as pre-FFN norm, SwiGLU
    Wsoss = make_weights(rng, dh, False)
    write("seed_oss", "seed_oss", Wsoss,
          {**base, "post_as_ffn": True, "rope": rope_table(dh, 10000.0)})

    # BailingMoE (v1): NORM RoPE, RMSNorm, softmax routed MoE (norm_w=False) + shared expert
    Wbm1 = make_weights(rng, dh, False)
    add_moe(rng, Wbm1, 4, FFN // 2, sh_ffn=FFN // 2)
    write("bailingmoe", "bailingmoe", Wbm1,
          {**base, "neox": False, "n_used": 2, "moe_gating_func": 1, "moe_norm_weights": False,
           "rope": rope_table(dh, 10000.0)},
          meta={"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": FFN // 2,
                "expert_shared_count": 1, "expert_shared_feed_forward_length": FFN // 2})

    # OLMo: NORM RoPE, weightless non-parametric LayerNorm (no norm weights/biases), SwiGLU
    Wol1 = make_weights(rng, dh, False)
    Wol1["out_norm"] = np.ones(D)
    for L in Wol1["layers"]:
        L["attn_norm"] = np.ones(D)
        L["ffn_norm"] = np.ones(D)
    write("olmo", "olmo", Wol1,
          {**base, "neox": False, "layernorm": True, "weightless_norm": True, "rope": rope_table(dh, 10000.0)})

    # StableLM: NEOX RoPE, LayerNorm + bias, optional per-head 2D Q/K LayerNorm (no bias), SwiGLU
    Wslm = make_weights(rng, dh, False)
    for L in Wslm["layers"]:
        L["q_norm"] = rng.uniform(0.5, 1.5, (NH, dh))
        L["k_norm"] = rng.uniform(0.5, 1.5, (NKV, dh))
    add_biases(rng, Wslm, dh, norm_b=True, qkv_b=False, o_b=False, ffn_b=False, out_b=False)
    write("stablelm", "stablelm", Wslm,
          {**base, "layernorm": True, "qk_norm": True, "rope": rope_table(dh, 10000.0)})

    # GPT-2: learned position_embd (no RoPE), LayerNorm + bias, fused attn_qkv + bias, output biases, non-gated GELU FFN + bias
    Wgp2 = make_weights(rng, dh, False)
    Wgp2["pos_embd"] = rng.normal(0, 0.2, (512, D))
    add_biases(rng, Wgp2, dh, norm_b=True, qkv_b=True, o_b=True, ffn_b=True, out_b=False)
    write("gpt2", "gpt2", Wgp2,
          {**base, "neox": False, "layernorm": True, "no_rope_all": True, "ffn_gated": False, "act": "gelu",
           "rope": rope_table(dh, 10000.0)},
          fused_qkv_only=True)

    # AfMoE (4 layers): NEOX RoPE, sqrt(d_model) embd scaling, ISWA + NoPE every 4th layer, Q/K RMSNorm, attn_gate * sigmoid, post_attention/ffw norms, leading dense + sigmoid MoE + exp_probs_b + shexp
    NL_af = NL
    NL = 4
    Waf = make_weights(rng, dh, False)
    for L in Waf["layers"]:
        L["attn_gate"] = rng.normal(0, 0.3, (NH * dh, D))
    add_moe(rng, Waf, 4, FFN // 2, sh_ffn=FFN // 2, start_layer=1, exp_bias=True)
    write("afmoe", "afmoe", Waf,
          {**base, "embd_scale": math.sqrt(D), "qk_norm": True, "post_norms": True, "window": 3, "pattern": 4, "nope": True,
           "n_used": 2, "moe_gating_func": 2, "moe_norm_weights": False, "rope": rope_table(dh, 10000.0)},
          meta={"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": FFN // 2,
                "expert_shared_count": 1, "leading_dense_block_count": 1, "expert_gating_func": 2,
                "attention.sliding_window": 3})

    # SmallThinker (5 layers): NEOX RoPE, ISWA (window=4096 when > 0) + NoPE at l % 4 == 0, pre-attention router logits, softmax_weight (norm_w=True), ReGLU experts
    NL = 5
    rng_st = np.random.default_rng(7)
    Wst = make_weights(rng_st, dh, False)
    add_moe(rng_st, Wst, 4, FFN)
    write("smallthinker", "smallthinker", Wst,
          {**base, "act": "relu", "window": 4096, "pattern": 4, "swa_first_global": True, "nope": True,
           "pre_attn_router": True, "n_used": 2, "moe_gating_func": 1, "moe_norm_weights": True,
           "rope": rope_table(dh, 10000.0)},
          meta={"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": FFN,
                "expert_gating_func": 3, "attention.sliding_window": 3})
    NL = NL_af


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
    if qtype in (T.IQ2_XXS, T.IQ2_XS, T.IQ2_S, T.IQ3_XXS, T.IQ3_S, T.IQ1_S, T.IQ1_M):  # grid formats: random blocks too
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        raw = rng.integers(0, 256, size=(a.shape[0], a.shape[1] // block * size), dtype=np.uint8)
        view = raw.reshape(a.shape[0], -1, size)
        n = view.shape[0] * view.shape[1]
        sd = 0.0001 if qtype == T.IQ3_S else 0.0003
        d_bits = rng.normal(0, sd, n).astype(np.float16).view(np.uint16)
        if qtype == T.IQ1_M:  # its f16 scale is spread over the top nibbles of the four u16 scale words at bytes 48..55
            flat = view.reshape(n, size)
            for k in range(4):
                word = flat[:, 48 + 2 * k].astype(np.uint32) | (flat[:, 49 + 2 * k].astype(np.uint32) << 8)
                word = (word & 0x0FFF) | (((d_bits.astype(np.uint32) >> (4 * k)) & 0xF) << 12)
                flat[:, 48 + 2 * k] = (word & 0xFF).astype(np.uint8)
                flat[:, 49 + 2 * k] = (word >> 8).astype(np.uint8)
        else:
            view[:, :, 0:2] = d_bits.view(np.uint8).reshape(view.shape[0], -1, 2)
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
    for name, qtype in (("q4_0", T.Q4_0), ("q5_0", T.Q5_0), ("q4_1", T.Q4_1), ("q5_1", T.Q5_1), ("q5_k", T.Q5_K), ("iq4_xs", T.IQ4_XS), ("iq4_nl", T.IQ4_NL), ("iq2_xxs", T.IQ2_XXS), ("iq2_xs", T.IQ2_XS), ("iq2_s", T.IQ2_S), ("iq3_xxs", T.IQ3_XXS), ("iq3_s", T.IQ3_S), ("iq1_s", T.IQ1_S), ("iq1_m", T.IQ1_M), ("f16", T.F16), ("bf16", T.BF16)):
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
