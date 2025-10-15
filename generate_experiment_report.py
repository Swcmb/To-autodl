import os
import re
import argparse
import subprocess
from datetime import datetime
from itertools import product
from typing import List, Dict, Any, Tuple

# ========== 基础配置 ==========
# 用户提供的基础命令（入口由脚本自动探测后替换为 EM/main.py）
BASE_STR_USER = "python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5"

RESULT_ROOT = os.path.join("OUTPUT", "result")
REPORT_MD = os.path.join("EM", "test.md")
ASSET_DIR = os.path.join(RESULT_ROOT, "_report_assets")

# 参数空间默认值（满足题述要求）
DEFAULT_FEATURE_TYPES = ['one_hot']  # one_hot|uniform|normal|position
DEFAULT_AUGMENTS = ['none', 'random_permute_features', 'add_noise', 'attribute_mask', 'noise_then_mask']  # none|random_permute_features|add_noise|attribute_mask|noise_then_mask
DEFAULT_AUGMENT_MODES = ['static',]  # static|online
DEFAULT_ENCODER_TYPES = ['gat','gt','gat_gt_serial','gat_gt_parallel']  # csglmd|csglmd_main|mgacmda|gat|gt|gat_gt_serial|gat_gt_parallel
DEFAULT_FUSION_TYPES = ['basic','dot','additive','self_attn','gat_fusion','gt_fusion']  # basic|dot|additive|self_attn|gat_fusion|gt_fusion
DEFAULT_QUEUE_WARMUPS = [5]  # int

# ========== 工具函数 ==========
def detect_entry_command(base_str: str) -> str:
    """
    自动探测入口脚本：
    - 若存在 EM/main.py 则将 'python main.py' 替换为 'python EM/main.py'
    - 否则保留原样
    """
    entry_em = os.path.join("EM", "main.py")
    if os.path.exists(entry_em):
        return base_str.replace("python main.py", "python EM/main.py")
    return base_str

def normalize_text(s: str) -> str:
    return (s or "").strip()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_summary_file(fp: str) -> dict:
    """
    解析 result_summary_*.txt，尽量提取：
    - run_name、encoder_type、fusion_type、feature_type、augment、augment_mode、queue_warmup_steps
    - AUC/AUROC、AUPRC、ACC、Precision、Recall、F1
    """
    data = {
        "run_name": None,
        "encoder_type": None,
        "fusion_type": None,
        "feature_type": None,
        "augment": None,
        "augment_mode": None,
        "queue_warmup_steps": None,
        "AUC": None,
        "AUPRC": None,
        "ACC": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "_file": fp
    }
    if not os.path.isfile(fp):
        return data

    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取失败: {fp} -> {e}")
        return data

    key_map = {
        "run_name": "run_name",
        "encoder": "encoder_type",
        "encoder_type": "encoder_type",
        "fusion": "fusion_type",
        "fusion_type": "fusion_type",
        "feature_type": "feature_type",
        "feature": "feature_type",
        "augment": "augment",
        "augment_mode": "augment_mode",
        "queue_warmup_steps": "queue_warmup_steps",
        "warmup_steps": "queue_warmup_steps",
        "auc": "AUC",
        "auroc": "AUC",
        "auprc": "AUPRC",
        "acc": "ACC",
        "accuracy": "ACC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "f1_score": "F1",
        "f1-score": "F1",
    }
    num_pattern = re.compile(r"(-?\d+(?:\.\d+)?)")

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        parts = re.split(r"[:=|\t]", line, maxsplit=1)
        if len(parts) == 2:
            k, v = parts[0].strip().lower(), parts[1].strip()
            for cand, norm in key_map.items():
                if cand == k or k.endswith(cand) or cand in k:
                    if norm in ["AUC", "AUPRC", "ACC", "Precision", "Recall", "F1"]:
                        m = num_pattern.search(v)
                        if m:
                            data[norm] = m.group(1)
                    elif norm in ["queue_warmup_steps"]:
                        m = num_pattern.search(v)
                        if m:
                            data[norm] = m.group(1)
                    else:
                        data[norm] = v
        else:
            low = line.lower()
            for cand, norm in key_map.items():
                if cand in low:
                    if norm in ["AUC", "AUPRC", "ACC", "Precision", "Recall", "F1", "queue_warmup_steps"]:
                        m = num_pattern.search(line)
                        if m:
                            data[norm] = m.group(1)
                    else:
                        m2 = re.search(rf"{cand}\s*[:=]\s*(.+)", low)
                        if m2:
                            data[norm] = m2.group(1).strip()

    # 规范化
    for k in list(data.keys()):
        if isinstance(data.get(k), str):
            data[k] = normalize_text(data[k])

    return data

def collect_all_summaries() -> List[dict]:
    """
    遍历 OUTPUT/result，收集每个目录下的 result_summary_*.txt，
    并尝试从目录/运行名补全关键参数。
    """
    collected = []
    if not os.path.isdir(RESULT_ROOT):
        return collected

    for sub in sorted(os.listdir(RESULT_ROOT)):
        subdir = os.path.join(RESULT_ROOT, sub)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if fname.startswith("result_summary_") and fname.endswith(".txt"):
                fp = os.path.join(subdir, fname)
                info = parse_summary_file(fp)
                info["_dir"] = sub
                try:
                    # 目录名或 run_name 解析：encoder-fusion__ft-XXX__aug-XXX__am-XXX__qw-10
                    rn_hint = info.get("run_name") or sub.split("_data")[0]
                    if rn_hint and not info.get("run_name"):
                        info["run_name"] = rn_hint

                    if rn_hint:
                        parts = rn_hint.split("__")
                        head = parts[0] if parts else rn_hint
                        if "-" in head:
                            enc, fus = head.split("-", 1)
                            info.setdefault("encoder_type", enc)
                            info.setdefault("fusion_type", fus)
                        else:
                            info.setdefault("encoder_type", head)
                        for p in parts[1:]:
                            if p.startswith("ft-"):
                                info.setdefault("feature_type", p[3:])
                            elif p.startswith("aug-"):
                                info.setdefault("augment", p[4:])
                            elif p.startswith("am-"):
                                info.setdefault("augment_mode", p[3:])
                            elif p.startswith("qw-"):
                                info.setdefault("queue_warmup_steps", p[3:])
                except Exception:
                    pass
                collected.append(info)
    return collected

# ========== 组合与命令构建 ==========
def build_param_grid(
    feature_types: List[str],
    augments: List[str],
    augment_modes: List[str],
    encoder_types: List[str],
    fusion_types: List[str],
    queue_warmups: List[int]
) -> List[Dict[str, Any]]:
    combos = []
    for ft, aug, am, enc, fus, qw in product(feature_types, augments, augment_modes, encoder_types, fusion_types, queue_warmups):
        combos.append({
            "feature_type": ft,
            "augment": aug,
            "augment_mode": am,
            "encoder_type": enc,
            "fusion_type": fus,
            "queue_warmup_steps": int(qw),
        })
    return combos

def make_run_name(p: Dict[str, Any]) -> str:
    # 形如：encoder-fusion__ft-XXX__aug-XXX__am-XXX__qw-10
    head = p["encoder_type"] + "-" + p["fusion_type"]
    suffix = [
        f"ft-{p['feature_type']}",
        f"aug-{p['augment']}",
        f"am-{p['augment_mode']}",
        f"qw-{p['queue_warmup_steps']}",
    ]
    return head + "__" + "__".join(suffix)

def build_cmd(base_str: str, p: Dict[str, Any], extra_epochs: int = None) -> Tuple[str, str]:
    cmd = base_str
    # 覆盖 feature_type（即使 base_str 已有）
    if p.get("feature_type"):
        cmd = re.sub(r"--feature_type\s+\S+", f"--feature_type {p['feature_type']}", cmd)
    # augment
    if p.get("augment") is not None:
        if p["augment"] == "none":
            cmd += f" --augment none"
        else:
            cmd += f" --augment {p['augment']}"
    # augment_mode
    if p.get("augment_mode"):
        cmd += f" --augment_mode {p['augment_mode']}"
    # encoder_type
    if p.get("encoder_type"):
        cmd += f" --encoder_type {p['encoder_type']}"
    # fusion_type
    if p.get("fusion_type"):
        cmd += f" --fusion_type {p['fusion_type']}"
    # queue_warmup_steps
    if p.get("queue_warmup_steps") is not None:
        cmd += f" --queue_warmup_steps {int(p['queue_warmup_steps'])}"
    # 覆盖 epochs（可选）
    if extra_epochs is not None:
        if re.search(r"--epochs\s+\d+", cmd):
            cmd = re.sub(r"--epochs\s+\d+", f"--epochs {int(extra_epochs)}", cmd)
        else:
            cmd += f" --epochs {int(extra_epochs)}"
    # run_name
    rn = make_run_name(p)
    cmd += f" --run_name {rn}"
    return cmd, rn

# ========== 执行与报告 ==========
def run_command(cmd: str, timeout: int = 0) -> int:
    """
    执行命令并实时打印输出；返回进程退出码。timeout: 0 表示不限时。
    注意：请先激活 conda 环境：conda activate experiment
    """
    print("=" * 80)
    print(f"[RUN] {cmd}")
    print("-" * 80)
    try:
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc:
            try:
                for line in iter(proc.stdout.readline, ""):
                    if line == "" and proc.poll() is not None:
                        break
                    print(line, end="")
                ret = proc.wait(timeout=None if timeout <= 0 else timeout)
                print("-" * 80)
                print(f"[EXIT CODE] {ret}")
                return ret
            except subprocess.TimeoutExpired:
                proc.kill()
                print("\n[TIMEOUT] 命令超时，已终止。")
                return -9
    except FileNotFoundError:
        print("[ERROR] 找不到可执行命令，请确认 python 是否在当前环境可用（建议先运行：conda activate experiment）。")
        return -127
    except Exception as e:
        print(f"[ERROR] 运行失败：{e}")
        return -1

def pick_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")

def build_leaderboard(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    enriched = []
    for r in rows:
        val = pick_float(r.get(key))
        enriched.append((val, r))
    # 将 NaN 放到后面
    enriched.sort(key=lambda x: (x[0] != x[0], -x[0] if x[0] == x[0] else 0))
    return [r for _, r in enriched]

def table_row(cols: List[str]) -> str:
    return "| " + " | ".join(cols) + " |"

def to_markdown_table(title: str, header: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append(f"### {title}")
    out.append("")
    out.append(table_row(header))
    out.append(table_row(["---"] * len(header)))
    for r in rows:
        out.append(table_row([str(c) if c not in (None, "") else "-" for c in r]))
    out.append("")
    return "\n".join(out)

def plot_trends(all_rows: List[Dict[str, Any]]) -> List[str]:
    """
    生成关键参数影响趋势图（按 encoder/fusion/augment/feature_type/augment_mode 对 AUC/ACC/F1 的均值），
    保存 PNG，返回图片路径列表。若 matplotlib 不可用则跳过。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] 无法导入 matplotlib，跳过绘图：{e}")
        return []

    ensure_dir(ASSET_DIR)
    images = []
    dims = [
        ("encoder_type", "不同 Encoder 的表现"),
        ("fusion_type", "不同 Fusion 的表现"),
        ("augment", "不同 Augment 的表现"),
        ("feature_type", "不同 Feature Type 的表现"),
        ("augment_mode", "不同 Augment Mode 的表现"),
    ]
    metrics = ["AUC", "ACC", "F1"]

    for dim, title in dims:
        groups: Dict[str, Dict[str, List[float]]] = {}
        for r in all_rows:
            k = r.get(dim) or "-"
            groups.setdefault(k, {m: [] for m in metrics})
            for m in metrics:
                v = pick_float(r.get(m))
                if v == v:  # 非 NaN
                    groups[k][m].append(v)

        if not groups:
            continue

        labels = list(groups.keys())
        bar_vals = {m: [] for m in metrics}
        for label in labels:
            for m in metrics:
                arr = groups[label][m]
                mean_v = sum(arr)/len(arr) if arr else float("nan")
                bar_vals[m].append(mean_v)

        x = list(range(len(labels)))
        width = 0.25
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.figure(figsize=(max(6, len(labels)*0.9), 4))
            for i, m in enumerate(metrics):
                xs = [xi + (i-1)*width for xi in x]
                plt.bar(xs, bar_vals[m], width=width, label=m)
            plt.xticks(x, labels, rotation=20, ha="right")
            plt.ylabel("Score")
            plt.title(title + "（均值）")
            plt.legend()
            img_path = os.path.join(ASSET_DIR, f"trend_{dim}.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=150)
            plt.close()
            images.append(img_path)
        except Exception as e:
            print(f"[WARN] 绘图失败 {dim}: {e}")

    return images

def generate_report(base_str: str, planned_cmds: List[Tuple[str, str]]):
    summaries = collect_all_summaries()

    # 构建行数据
    rows = []
    for s in summaries:
        rows.append({
            "Run": s.get("run_name") or s.get("_dir"),
            "encoder_type": s.get("encoder_type"),
            "fusion_type": s.get("fusion_type"),
            "feature_type": s.get("feature_type"),
            "augment": s.get("augment"),
            "augment_mode": s.get("augment_mode"),
            "queue_warmup_steps": s.get("queue_warmup_steps"),
            "AUC": s.get("AUC"),
            "AUPRC": s.get("AUPRC"),
            "ACC": s.get("ACC"),
            "Precision": s.get("Precision"),
            "Recall": s.get("Recall"),
            "F1": s.get("F1"),
        })

    # 生成趋势图
    img_paths = plot_trends(rows)

    # 报告内容
    lines: List[str] = []
    lines.append("# 实验报告")
    lines.append("")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 公共参数（入口已自动探测）：`{base_str}`")
    lines.append("")
    lines.append("## 计划执行的命令")
    lines.append("")
    for rn, cmd in planned_cmds:
        lines.append(f"- {rn}")
        lines.append(f"  - `{cmd}`")
    lines.append("")

    # 全部组合表现表
    header = ["Run", "encoder_type", "fusion_type", "feature_type", "augment", "augment_mode", "queue_warmup_steps", "AUC", "AUPRC", "ACC", "Precision", "Recall", "F1"]
    table_rows = []
    for r in rows:
        table_rows.append([r.get(h, "-") for h in header])
    lines.append(to_markdown_table("全部组合表现", header, table_rows))

    # 最优配置推荐（AUC/ACC/F1）
    for key in ["AUC", "ACC", "F1"]:
        lb = build_leaderboard(rows, key)
        topk = lb[:5]
        header2 = ["Rank", key, "Run", "encoder_type", "fusion_type", "feature_type", "augment", "augment_mode", "queue_warmup_steps"]
        trows = []
        for i, r in enumerate(topk, 1):
            trows.append([
                str(i), str(r.get(key, "-")),
                r.get("Run", "-"),
                r.get("encoder_type", "-"),
                r.get("fusion_type", "-"),
                r.get("feature_type", "-"),
                r.get("augment", "-"),
                r.get("augment_mode", "-"),
                r.get("queue_warmup_steps", "-"),
            ])
        lines.append(to_markdown_table(f"最优配置推荐（按 {key}）", header2, trows))

    # 趋势图
    if img_paths:
        lines.append("## 趋势图")
        lines.append("")
        for p in img_paths:
            rel = os.path.relpath(p, start=os.path.dirname(REPORT_MD))
            lines.append(f"![{os.path.basename(p)}]({rel})")
            lines.append("")

    ensure_dir(os.path.dirname(REPORT_MD))
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告已生成：{REPORT_MD}")

# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="自动生成参数组合，顺序执行实验并汇总 OUTPUT/result 指标，输出结构化报告与趋势图。")
    parser.add_argument("--execute", action="store_true", help="是否按组合顺序执行实验（默认仅生成报告与命令清单，不执行）")
    parser.add_argument("--timeout", type=int, default=0, help="单条命令超时时间（秒，0 不限时）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖基础命令中的 epochs 数值（可选）")
    parser.add_argument("--limit", type=int, default=None, help="仅取前 N 个组合用于执行/清单（快速验证用）")

    # 参数空间（均可多值）
    parser.add_argument("--feature_types", nargs="+", default=DEFAULT_FEATURE_TYPES, help="可选：one_hot uniform normal position")
    parser.add_argument("--augments", nargs="+", default=DEFAULT_AUGMENTS, help="可选：none random_permute_features add_noise attribute_mask noise_then_mask")
    parser.add_argument("--augment_modes", nargs="+", default=DEFAULT_AUGMENT_MODES, help="可选：static online")
    parser.add_argument("--encoder_types", nargs="+", default=DEFAULT_ENCODER_TYPES, help="可选：csglmd csglmd_main mgacmda gat gt gat_gt_serial gat_gt_parallel")
    parser.add_argument("--fusion_types", nargs="+", default=DEFAULT_FUSION_TYPES, help="可选：basic dot additive self_attn gat_fusion gt_fusion")
    parser.add_argument("--queue_warmup_steps", nargs="+", type=int, default=DEFAULT_QUEUE_WARMUPS, help="如：0 10 50")

    args = parser.parse_args()

    base_str = detect_entry_command(BASE_STR_USER)

    # 生成笛卡尔积组合
    grid = build_param_grid(
        args.feature_types,
        args.augments,
        args.augment_modes,
        args.encoder_types,
        args.fusion_types,
        args.queue_warmup_steps,
    )
    if args.limit is not None and args.limit > 0:
        grid = grid[: args.limit]

    # 构建命令与 run_name
    planned: List[Tuple[str, str]] = []
    for p in grid:
        cmd, rn = build_cmd(base_str, p, extra_epochs=args.epochs)
        planned.append((rn, cmd))

    # 对最后一条命令追加 --shutdown
    if planned:
        last_rn, last_cmd = planned[-1]
        if "--shutdown" not in last_cmd:
            planned[-1] = (last_rn, f"{last_cmd} --shutdown")
            print(f"[提示] 已为最后一条任务 '{last_rn}' 追加 --shutdown")

    # 可选执行（控制台实时输出）
    if args.execute:
        print("\n将开始执行命令，并在控制台实时显示输出：")
        for rn, cmd in planned:
            print(f"\n[任务] {rn}")
            rc = run_command(cmd, timeout=args.timeout)
            print(f"[完成] {rn} -> 返回码 {rc}")

    # 生成报告（表格 + 最优推荐 + 趋势图）
    generate_report(base_str, planned)

if __name__ == "__main__":
    main()