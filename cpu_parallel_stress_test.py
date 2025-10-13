import os
import time
import math
import random
import hashlib
import argparse
import shutil
from pathlib import Path
from multiprocessing import Process, current_process, cpu_count

def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def cpu_burn(n_iter: int = 500_000) -> str:
    """CPU密集计算（哈希+浮点运算混合），返回哈希摘要以避免优化掉。"""
    acc = 0.0
    h = hashlib.sha256()
    for i in range(n_iter):
        x = math.sin(i * 0.001) * math.cos(i * 0.002) + random.random()
        acc += x
        if i % 10000 == 0:
            h.update(str(acc).encode("utf-8"))
    return f"acc={acc:.6f}, sha256={h.hexdigest()}"

def ensure_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    """建立 OUTPUT/result/stress_时间戳 目录结构，返回 (output_dir, result_dir, run_dir)。"""
    output_dir = base_dir / "OUTPUT"
    result_dir = output_dir / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir = result_dir / f"stress_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, result_dir, run_dir

def save_text(run_dir: Path, content: str, filename: str, subdir: str | None = None) -> None:
    """同步写入文本文件。"""
    target_dir = run_dir if not subdir else (run_dir / subdir)
    target_dir.mkdir(parents=True, exist_ok=True)
    fpath = target_dir / filename
    with fpath.open("w", encoding="utf-8") as f:
        f.write(content)

def worker_task(run_dir: Path, idx: int, files_per_worker: int, payload_size: int) -> None:
    """单进程任务：执行CPU负载并写入指定数量的文件到各自子目录。"""
    pid = os.getpid()
    proc_name = current_process().name
    subdir = f"worker_{idx:02d}"  # 每个worker独立子目录，降低目录竞争
    for k in range(files_per_worker):
        summary = cpu_burn(n_iter=payload_size)
        content = [
            f"worker={idx}, pid={pid}, name={proc_name}, file_idx={k}",
            f"time={time.strftime('%Y-%m-%d %H:%M:%S')}",
            summary
        ]
        fname = f"stress_{idx:02d}_{k:04d}.txt"
        save_text(run_dir, "\n".join(content), fname, subdir=subdir)

def main():
    parser = argparse.ArgumentParser(description="CPU并行压力测试（含并行保存文件，默认自动清理）")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="并行进程数")
    parser.add_argument("--files-per-worker", type=int, default=100, help="每个进程写入文件数量")
    parser.add_argument("--payload-size", type=int, default=500_000, help="每次CPU计算迭代数")
    parser.add_argument("--no-cleanup", action="store_true", help="运行结束后不清理结果目录（默认自动清理）")
    args = parser.parse_args()

    # 项目根为脚本的上级目录（EM 的上一级）
    base_dir = Path(__file__).resolve().parent.parent
    _, _, run_dir = ensure_dirs(base_dir)

    print(f"[START] CPU+IO stress test")
    print(f"workers={args.workers}, files_per_worker={args.files_per_worker}, payload_size={args.payload_size}")
    print(f"run_dir={run_dir}")

    procs: list[Process] = []
    for i in range(args.workers):
        p = Process(target=worker_task, args=(run_dir, i, args.files_per_worker, args.payload_size))
        p.start()
        procs.append(p)

    # 等待所有进程完成（确保所有文件写入完成）
    for p in procs:
        p.join()

    print("[DONE] All processes finished.")
    if args.no_cleanup:
        print(f"[KEEP] Results kept at: {run_dir}")
    else:
        try:
            shutil.rmtree(run_dir)
            print(f"[CLEANUP] Results removed: {run_dir}")
        except Exception as e:
            print(f"[CLEANUP-ERROR] Failed to remove {run_dir}: {e}")

if __name__ == "__main__":
    main()