#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairs_from_retrieval_and_sequential.py
- 先用 HLOC 官方 CLI 產生 retrieval pairs
- 再按檔名順序補上時序鄰近 pairs（雙向）
- 合併去重後寫出
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from hloc.utils.parsers import parse_image_list


def read_pairs_txt(path: Path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) >= 2:
                pairs.append((toks[0], toks[1]))
    return pairs


def write_pairs_txt(path: Path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")


def build_sequential_pairs(names, w: int):
    """為每張圖與前後 w 張補 pair；雙向都寫（a b & b a）"""
    out = []
    n = len(names)
    for i in range(n):
        for d in range(1, w + 1):
            j = i + d
            if j < n:
                out.append((names[i], names[j]))
                out.append((names[j], names[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_list", required=True)
    ap.add_argument("--global_feats", required=True)
    ap.add_argument("--num_retrieval", type=int, default=10)
    ap.add_argument("--seq_window", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    db_list_path = Path(args.db_list)
    global_feats = Path(args.global_feats)
    out_path = Path(args.output)

    # 1) 讀 DB 影像清單（保持順序）
    db_names = parse_image_list(db_list_path)
    print(f"[INFO] DB images: {len(db_names)}")

    # 2) 用官方 CLI 做 retrieval，輸出到暫存檔
    with tempfile.TemporaryDirectory() as td:
        tmp_pairs = Path(td) / "retrieval.txt"
        cmd = [
            "python", "-m", "hloc.pairs_from_retrieval",
            "--query_list", str(db_list_path),
            "--db_list", str(db_list_path),
            "--descriptors", str(global_feats),
            "--db_descriptors", str(global_feats),
            "--num_matched", str(args.num_retrieval),
            "--output", str(tmp_pairs),
        ]
        print(f"[Retrieval] Generating top-{args.num_retrieval} pairs ...")
        subprocess.run(cmd, check=True)

        retrieval_pairs = read_pairs_txt(tmp_pairs)
        print(f"[Retrieval] Got {len(retrieval_pairs)} pairs from HLOC CLI.")

    # 3) 時序鄰近 pairs（雙向）
    seq_pairs = build_sequential_pairs(db_names, args.seq_window)
    print(f"[Sequential] Added {len(seq_pairs)} pairs with window={args.seq_window}.")

    # 4) 合併去重（無向去重：把 (a,b) 與 (b,a) 視為同一組）
    def undirected_key(p):
        a, b = p
        return (a, b) if a <= b else (b, a)

    uniq = {}
    for p in retrieval_pairs + seq_pairs:
        k = undirected_key(p)
        # 優先保留 retrieval 的方向，不特別需要就第一個遇到的為主
        if k not in uniq:
            uniq[k] = p

    merged_pairs = list(uniq.values())
    print(f"[Merge] Final unique pairs: {len(merged_pairs)}")

    # 5) 輸出
    write_pairs_txt(out_path, merged_pairs)
    print(f"[OK] Wrote merged pairs to: {out_path}")


if __name__ == "__main__":
    main()
