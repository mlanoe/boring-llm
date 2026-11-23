#!/usr/bin/env python3
"""Inventory quantized matrix shapes in a GGUF without running the model.

The report groups tensors by role, format and matrix shape, computes packed
weight traffic for one decode token, and prints commands for Boring's resident
K-quant Metal microbenchmark.  It is intentionally timing-free: the manifest
is useful while the machine is busy and can drive benchmarks later.
"""
import argparse
import json
import math
import re
from pathlib import Path

import gguf


KQUANTS = {"Q2_K": "q2_k", "Q3_K": "q3_k", "Q4_K": "q4_k", "Q6_K": "q6_k"}
ROLE_PATTERNS = (
    (re.compile(r"^blk\.\d+\.attn_q\.weight$"), "attn-q"),
    (re.compile(r"^blk\.\d+\.attn_k\.weight$"), "attn-k"),
    (re.compile(r"^blk\.\d+\.attn_v\.weight$"), "attn-v"),
    (re.compile(r"^blk\.\d+\.attn_output\.weight$"), "attn-output"),
    (re.compile(r"^blk\.\d+\.ffn_gate\.weight$"), "ffn-gate"),
    (re.compile(r"^blk\.\d+\.ffn_up\.weight$"), "ffn-up"),
    (re.compile(r"^blk\.\d+\.ffn_down\.weight$"), "ffn-down"),
    (re.compile(r"^token_embd\.weight$"), "token-embedding"),
    (re.compile(r"^output\.weight$"), "output-head"),
)


def tensor_role(name):
    for pattern, role in ROLE_PATTERNS:
        if pattern.match(name):
            return role
    return "other"


def shape_row(tensor):
    shape = [int(v) for v in tensor.shape]
    if len(shape) != 2 or not tensor.name.endswith(".weight"):
        return None
    # GGUF stores matrix dimensions as [input columns, output rows].
    k, n = shape
    return {
        "name": tensor.name,
        "role": tensor_role(tensor.name),
        "format": tensor.tensor_type.name,
        "n": n,
        "k": k,
        "packed_bytes": int(tensor.n_bytes),
    }


def group_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["role"], row["format"], row["n"], row["k"], row["packed_bytes"])
        item = grouped.setdefault(key, {**row, "count": 0, "names": []})
        item["count"] += 1
        item["names"].append(row["name"])
    result = sorted(grouped.values(), key=lambda r: (r["role"], r["format"], r["n"], r["k"]))
    for row in result:
        if row["format"] in ("Q4_K", "Q5_K"):
            rows_per_simdgroup, rows_per_threadgroup = 2, 16
            schedule = "kquant-rows-2"
        elif row["format"] == "Q6_K":
            rows_per_simdgroup, rows_per_threadgroup = 1, 8
            schedule = "kquant-warp-1"
        else:
            rows_per_simdgroup = rows_per_threadgroup = None
            schedule = None
        row["decode_schedule"] = schedule
        row["rows_per_simdgroup"] = rows_per_simdgroup
        row["rows_per_threadgroup"] = rows_per_threadgroup
        row["decode_threadgroups"] = (
            math.ceil(row["n"] / rows_per_threadgroup) if rows_per_threadgroup else None
        )
        row["k_superblocks_per_row"] = row["k"] // 256 if row["format"] in KQUANTS else None
    return result


def build_manifest(path, bandwidth_gbs):
    reader = gguf.GGUFReader(str(path))
    rows = [row for tensor in reader.tensors if (row := shape_row(tensor)) is not None]
    if not any(row["name"] == "output.weight" for row in rows):
        for row in rows:
            if row["name"] == "token_embd.weight":
                row["role"] = "tied-output"
    for row in rows:
        # An untied token embedding performs one row lookup; it does not stream
        # the complete vocabulary matrix like the output projection does.
        row["full_scan_per_decode"] = row["role"] != "token-embedding"
    groups = group_rows(rows)
    total_bytes = sum(row["packed_bytes"] for row in rows if row["full_scan_per_decode"])
    kquant_bytes = sum(row["packed_bytes"] for row in rows
                       if row["full_scan_per_decode"] and row["format"] in KQUANTS)
    role_bytes = {}
    for row in rows:
        if row["full_scan_per_decode"]:
            role_bytes[row["role"]] = role_bytes.get(row["role"], 0) + row["packed_bytes"]
    traffic_by_role = [
        {"role": role, "packed_bytes": size, "share": size / total_bytes}
        for role, size in sorted(role_bytes.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "schema": 1,
        "gguf": str(path.resolve()),
        "tensor_count": len(reader.tensors),
        "matrix_count": len(rows),
        "packed_matrix_bytes_per_decode_token": total_bytes,
        "kquant_bytes_per_decode_token": kquant_bytes,
        "streaming_lower_bound_ms": total_bytes / (bandwidth_gbs * 1e6),
        "assumed_bandwidth_gbs": bandwidth_gbs,
        "traffic_by_role": traffic_by_role,
        "groups": groups,
    }


def print_report(manifest, bench_source):
    print(f"GGUF: {manifest['gguf']}")
    print(f"matrices: {manifest['matrix_count']} / tensors: {manifest['tensor_count']}")
    mib = manifest["packed_matrix_bytes_per_decode_token"] / 2**20
    print(f"packed matrix traffic/decode token: {mib:.1f} MiB")
    print(f"ideal streaming lower bound at {manifest['assumed_bandwidth_gbs']:.1f} GB/s: "
          f"{manifest['streaming_lower_bound_ms']:.2f} ms/token")
    print("traffic share: " + ", ".join(
        f"{row['role']} {row['share'] * 100:.1f}%" for row in manifest["traffic_by_role"]
    ))
    print()
    print(f"{'role':<16} {'format':<8} {'count':>5} {'scan':>5} {'n':>8} {'k':>8} {'grid':>7} {'k/256':>6} "
          f"{'MiB each':>10} {'MiB total':>10}")
    for row in manifest["groups"]:
        each = row["packed_bytes"] / 2**20
        grid = row["decode_threadgroups"] if row["decode_threadgroups"] is not None else "--"
        kblocks = row["k_superblocks_per_row"] if row["k_superblocks_per_row"] is not None else "--"
        scan = "yes" if row["full_scan_per_decode"] else "no"
        print(f"{row['role']:<16} {row['format']:<8} {row['count']:>5} {scan:>5} "
              f"{row['n']:>8} {row['k']:>8} "
              f"{grid:>7} {kblocks:>6} {each:>10.2f} {each * row['count']:>10.2f}")

    commands = []
    seen = set()
    for row in manifest["groups"]:
        if not row["full_scan_per_decode"]:
            continue
        fmt = KQUANTS.get(row["format"])
        key = (fmt, row["n"], row["k"])
        if not fmt or key in seen:
            continue
        seen.add(key)
        commands.append(f"cargo run --release -- {fmt} {row['n']} {row['k']} 200")
    if commands:
        print(f"\nResident Metal benchmark source: {bench_source}")
        print("After building it from the boring checkout, run:")
        for command in commands:
            print(f"  {command}")


def selftest():
    class Type:
        name = "Q4_K"

    class Tensor:
        name = "blk.3.ffn_down.weight"
        shape = (18944, 3584)
        tensor_type = Type()
        n_bytes = 123

    row = shape_row(Tensor())
    assert row == {"name": Tensor.name, "role": "ffn-down", "format": "Q4_K",
                   "n": 3584, "k": 18944, "packed_bytes": 123}
    groups = group_rows([row, row])
    assert groups[0]["count"] == 2
    assert groups[0]["names"] == [Tensor.name, Tensor.name]
    assert groups[0]["decode_schedule"] == "kquant-rows-2"
    assert groups[0]["decode_threadgroups"] == 224
    assert groups[0]["k_superblocks_per_row"] == 74
    print("selftest: ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", nargs="?")
    parser.add_argument("--bandwidth-gbs", type=float, default=96.0,
                        help="streaming ceiling used only for the ideal lower bound")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--bench-source", default="../boring/examples/tensor_k_quant_decode_bench.br")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.gguf:
        parser.error("gguf is required unless --selftest is used")
    if args.bandwidth_gbs <= 0:
        parser.error("--bandwidth-gbs must be positive")
    manifest = build_manifest(Path(args.gguf), args.bandwidth_gbs)
    print_report(manifest, args.bench_source)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
