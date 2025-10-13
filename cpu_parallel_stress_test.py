#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU并行压力测试：遍历 threads × workers × chunk_size，模拟 RBF + PBPA 风格计算，
记录耗时到 OUTPUT/result/cpu_stress_{timestamp}.csv，便于报告集成。
保持与主项目的后端线程控制一致（OMP/MKL等），不修改GPU训练逻辑。
"""

import os
import time
import argparse
import sys
import math
import datetime as dt
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def setup_threads(threads: int):
    t = int(max(1, min(32, threads)))
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
        os.environ[k] = str(t)
    os.environ["EM_THREADS"] = str(t)
    return t

# 进程级只读缓存
_DI_SIM = None
_NZ_IDX = None
def _init_pbpa(di_sim, nz_idx):
    global _DI_SIM, _NZ_IDX
    _DI_SIM = di_sim
    _NZ_IDX = nz_idx

def _pbpa_pair(pair):
    i, j = pair
    idx_i = _NZ_IDX[i]
    idx_j = _NZ_IDX[j]
    if len(idx_i) == 0 or len(idx_j) == 0:
        return 0.0
    sub = _DI_SIM[np.ix_(idx_i, idx_j)]
    return (np.max(sub, axis=0).sum() + np.max(sub, axis=1).sum()) / (sub.shape[0] + sub.shape[1])

def simulate_rbf(n: int, d: int):
    # 构造随机矩阵，模拟RBF相似度计算
    X = np.random.rand(n, d).astype(np.float64)
    G = X @ X.T
    nrm = np.sum(X * X, axis=1)
    D2 = np.maximum(0.0, nrm[:, None] + nrm[None, :] - 2.0 * G)
    sigma = np.median(D2[D2 > 0]) if np.any(D2 > 0) else 1.0
    K = np.exp(-D2 / (2.0 * (sigma if sigma > 1e-12 else 1.0)))
    return float(np.sum(K))  # 防止优化器删除

def simulate_pbpa(n: int, nz_per_row: int, workers: int, chunk_size: int):
    # 构造简单的二值矩阵，模拟 PBPA 索引查找
    di = np.zeros((n, n), dtype=np.float64)
    rng = np.random.default_rng(123)
    for i in range(n):
        idx = rng.choice(n, size=min(nz_per_row, n), replace=False)
        di[i, idx] = rng.random(len(idx))
    nz_idx = [np.flatnonzero(di[i] > 0) for i in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total_tasks = len(pairs)
    eff_workers = max(1, min(8, workers))
    chunk = max(1, (total_tasks // (eff_workers * 4)) or 1)
    with ProcessPoolExecutor(max_workers=eff_workers, initializer=_init_pbpa, initargs=(di, nz_idx)) as ex:
        s = 0.0
        for val in ex.map(_pbpa_pair, pairs, chunksize=chunk):
            s += val
    return s

def run_once(threads, workers, chunk_size, n=1024, d=64, nz_per_row=64):
    setup_threads(threads)
    t0 = time.perf_counter()
    simulate_rbf(n=n, d=d)
    t1 = time.perf_counter()
    simulate_pbpa(n=n//8, nz_per_row=nz_per_row, workers=workers, chunk_size=chunk_size)
    t2 = time.perf_counter()
    return {"threads": threads, "workers": workers, "chunk_size": chunk_size, "rbf_s": round(t1 - t0, 4), "pbpa_s": round(t2 - t1, 4), "total_s": round(t2 - t0, 4)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads_list", type=str, default="8,16,32", help="逗号分隔：例如 8,16,32")
    ap.add_argument("--workers_list", type=str, default="2,4,8", help="逗号分隔：例如 2,4,8")
    ap.add_argument("--chunk_list", type=str, default="10000,20000", help="逗号分隔：例如 10000,20000")
    ap.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../OUTPUT/result"))
    args = ap.parse_args()

    threads_set = [int(x) for x in args.threads_list.split(",") if x.strip()]
    workers_set = [int(x) for x in args.workers_list.split(",") if x.strip()]
    chunk_set = [int(x) for x in args.chunk_list.split(",") if x.strip()]

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"cpu_stress_{ts}.csv")

    rows = []
    for t in threads_set:
        for w in workers_set:
            for c in chunk_set:
                r = run_once(threads=t, workers=w, chunk_size=c)
                print(f"[STRESS] threads={t} workers={w} chunk={c} rbf={r['rbf_s']}s pbpa={r['pbpa_s']}s total={r['total_s']}s")
                rows.append(r)

    # 写CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("threads,workers,chunk_size,rbf_s,pbpa_s,total_s\n")
        for r in rows:
            f.write(f"{r['threads']},{r['workers']},{r['chunk_size']},{r['rbf_s']},{r['pbpa_s']},{r['total_s']}\n")

    print(f"[STRESS] saved: {out_csv}")

if __name__ == "__main__":
    main()