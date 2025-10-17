import os
import re
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from itertools import product
import sys

# =========================
# 控制变量法版：仅改变 moco 与 augment
# 固定组合来自用户提供清单
# =========================

# 用户基础命令（入口会被自动替换为 EM/main.py）
BASE_STR_USER = "python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5"

RESULT_ROOT = os.path.join("OUTPUT", "result")
RESULT_ROOTS = [RESULT_ROOT, os.path.join("EM", "result")]
REPORT_MD = os.path.join("EM", "test.md")
ASSET_DIR = os.path.join(RESULT_ROOT, "_report_assets")
FAILED_CMDS_LOG = os.path.join("OUTPUT", "log", "failed_commands.txt")

# 固定：模型-融合组合（按截图清单）
FIXED_COMBOS: List[Tuple[str, str]] = [
    ("csglmd", "gt_fusion"),
    ("csglmd", "basic"),
    ("csglmd", "self_attn"),
    ("csglmd", "additive"),
    ("csglmd", "gat_fusion"),
    ("csglmd", "dot"),
    ("gt", "self_attn"),
    ("gt", "gt_fusion"),
    ("gat_gt_parallel", "gt_fusion"),
    ("gat", "self_attn"),
    ("gat", "gt_fusion"),
    ("gat_gt_serial", "gt_fusion"),
]

# 自变量：MoCo 与增强视图
DEFAULT_MOCO_STATES = ["off", "on"]  # 仅两值
DEFAULT_AUGMENTS = ["random_permute_features"]

# 常量（不进入对比维度；仍可用参数设置为单值）
CONST_FEATURE_TYPE = "one_hot"
CONST_AUGMENT_MODE = "static"
CONST_QUEUE_WARMUP = 5  # 若你的主程序需要，可作为常量传入

# ========== 通用工具 ==========
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

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_text(s: Optional[str]) -> str:
    return (s or "").strip()

def parse_summary_file(fp: str) -> Dict[str, Any]:
    """
    解析 result_summary_*.txt，尽量提取核心指标与参数，兼容大小写/符号差异。
    并从目录或 run_name 推断 moco 状态。
    """
    data = {
        "run_name": None,
        "encoder_type": None,
        "fusion_type": None,
        "feature_type": None,
        "augment": None,
        "augment_mode": None,
        "queue_warmup_steps": None,
        "moco": None,  # 新增：on/off
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
        "contrastive_type": "contrastive_type",
        "moco": "moco",
        "moco_multi": "moco",
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
                    elif norm == "queue_warmup_steps":
                        m = num_pattern.search(v)
                        if m:
                            data[norm] = m.group(1)
                    elif norm == "moco":
                        lowv = v.lower()
                        if any(x in lowv for x in ["true", "on", "yes", "1"]):
                            data["moco"] = "on"
                        elif any(x in lowv for x in ["false", "off", "no", "0"]):
                            data["moco"] = "off"
                        else:
                            data["moco"] = normalize_text(v)
                    elif norm == "contrastive_type":
                        data["contrastive_type"] = v
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
                    elif norm == "moco":
                        if "moco_multi" in low or "moco on" in low:
                            data["moco"] = "on"
                        elif "moco off" in low:
                            data["moco"] = "off"

    # 如果提供了 contrastive_type，则据此推断 moco on/off（若尚未确定）
    ct = normalize_text(data.get("contrastive_type"))
    if data.get("moco") is None and ct:
        if ct in ("moco_multi", "moco_single"):
            data["moco"] = "on"
        elif ct == "em":
            data["moco"] = "off"

    # 从路径/运行名补全
    data["_dir"] = os.path.basename(os.path.dirname(fp))
    rn_hint = data.get("run_name") or data["_dir"].split("_data")[0]
    if rn_hint and not data.get("run_name"):
        data["run_name"] = rn_hint

    if rn_hint:
        parts = rn_hint.split("__")
        head = parts[0]
        if "-" in head:
            enc, fus = head.split("-", 1)
            if not normalize_text(data.get("encoder_type")):
                data["encoder_type"] = enc
            if not normalize_text(data.get("fusion_type")):
                data["fusion_type"] = fus
        else:
            if not normalize_text(data.get("encoder_type")):
                data["encoder_type"] = head

        for p in parts[1:]:
            if p.startswith("ft-"):
                if not normalize_text(data.get("feature_type")):
                    data["feature_type"] = p[3:]
            elif p.startswith("aug-"):
                if not normalize_text(data.get("augment")):
                    data["augment"] = p[4:]
            elif p.startswith("am-"):
                if not normalize_text(data.get("augment_mode")):
                    data["augment_mode"] = p[3:]
            elif p.startswith("qw-"):
                if not normalize_text(data.get("queue_warmup_steps")):
                    data["queue_warmup_steps"] = p[3:]
            elif p.startswith("moco-"):
                if not normalize_text(data.get("moco")):
                    data["moco"] = p[5:]

    # 目录名提示 moco
    dir_low = data["_dir"].lower()
    if data.get("moco") is None:
        if "moco_multi" in dir_low or "__moco-on" in dir_low:
            data["moco"] = "on"
        elif "__moco-off" in dir_low:
            data["moco"] = "off"

    # 规范化
    for k in list(data.keys()):
        if isinstance(data.get(k), str):
            data[k] = normalize_text(data[k])

    return data

def collect_all_summaries() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    roots = RESULT_ROOTS if 'RESULT_ROOTS' in globals() else [RESULT_ROOT]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for sub in sorted(os.listdir(root)):
            subdir = os.path.join(root, sub)
            if not os.path.isdir(subdir):
                continue
            for fname in os.listdir(subdir):
                if fname.startswith("result_summary_") and fname.endswith(".txt"):
                    rows.append(parse_summary_file(os.path.join(subdir, fname)))
    return rows

# ========== 组合与命令构建 ==========
def make_run_name(enc: str, fus: str, moco: str, augment: str, feature_type: str, augment_mode: str, queue_warmup_steps: int) -> str:
    # 形如：encoder-fusion__moco-on__aug-XXX__ft-one_hot__am-static__qw-5
    parts = [
        f"{enc}-{fus}",
        f"moco-{moco}",
        f"aug-{augment}",
        f"ft-{feature_type}",
        f"am-{augment_mode}",
        f"qw-{queue_warmup_steps}",
    ]
    return "__".join(parts)

def build_cmd(base_str: str, enc: str, fus: str, moco: str, augment: str, feature_type: str, augment_mode: str, queue_warmup_steps: int, extra_epochs: Optional[int]) -> Tuple[str, str]:
    cmd = base_str

    # 覆盖入口已有参数，确保常量/变量按我们控制
    def upsert(flag: str, value: Optional[str], add_if_none: bool = True):
        nonlocal cmd
        pat = re.compile(rf"{re.escape(flag)}\s+\S+")
        if value is None:
            return
        if pat.search(cmd):
            cmd = pat.sub(f"{flag} {value}", cmd)
        else:
            if add_if_none:
                cmd += f" {flag} {value}"

    # 常量与维度
    upsert("--feature_type", feature_type)
    upsert("--augment_mode", augment_mode)
    upsert("--encoder_type", enc)
    upsert("--fusion_type", fus)
    upsert("--queue_warmup_steps", str(int(queue_warmup_steps)))
    # augment
    upsert("--augment", augment)

    # moco：通过 --contrastive_type 与 --num_views 控制
    if moco == "on":
        upsert("--contrastive_type", "moco_multi")
        upsert("--num_views", "2")
    else:
        upsert("--contrastive_type", "em")
        upsert("--num_views", "1")

    # epochs（可选）
    if extra_epochs is not None:
        if re.search(r"--epochs\s+\d+", cmd):
            cmd = re.sub(r"--epochs\s+\d+", f"--epochs {int(extra_epochs)}", cmd)
        else:
            cmd += f" --epochs {int(extra_epochs)}"

    run_name = make_run_name(enc, fus, moco, augment, feature_type, augment_mode, queue_warmup_steps)
    if "--run_name" in cmd:
        cmd = re.sub(r"--run_name\s+\S+", f"--run_name {run_name}", cmd)
    else:
        cmd += f" --run_name {run_name}"
    return cmd, run_name

# ========== 终端高亮与执行辅助 ==========
def _ansi_supported() -> bool:
    """
    检测是否支持 ANSI 显示：
    - 终端为 TTY 且未设置 NO_COLOR 则启用
    - 兼容 Windows PowerShell 7+ / Linux 常见终端
    """
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    except Exception:
        return False

def _c(text: str, code: str) -> str:
    if _ansi_supported():
        return f"{code}{text}\033[0m"
    return text

def _bold(text: str) -> str:
    return _c(text, "\033[1m")

def _cyan(text: str) -> str:
    return _c(text, "\033[96m")

def _yellow(text: str) -> str:
    return _c(text, "\033[93m")

def _green(text: str) -> str:
    return _c(text, "\033[92m")

def print_task_header(index: int, total: int, run_name: str, cmd: str):
    bar = "=" * 80
    sub = "-" * 80
    print(bar)
    head = f"[{index}/{total}] RUN {run_name}"
    print(_bold(_cyan(head)))
    print(sub)
    print(_yellow(cmd))
    print(sub)

# ========== 执行 ==========
def run_command(cmd: str, timeout: int = 0) -> int:
    print("=" * 80)
    print(f"[RUN] {cmd}")
    print("-" * 80)
    try:
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc:
            try:
                assert proc.stdout is not None
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
        print("[ERROR] 找不到 python 或入口，请先激活环境。")
        return -127
    except Exception as e:
        print(f"[ERROR] 运行失败：{e}")
        return -1

# ========== 报告工具 ==========
def pick_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")

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

def build_leaderboard(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    enriched = []
    for r in rows:
        val = pick_float(r.get(key))
        enriched.append((val, r))
    enriched.sort(key=lambda x: (x[0] != x[0], -x[0] if x[0] == x[0] else 0))
    return [r for _, r in enriched]

def plot_trends(all_rows: List[Dict[str, Any]]) -> List[str]:
    """
    绘制维度均值趋势（encoder/fusion/augment/moco），保存 PNG。
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
        ("encoder_type", "Encoder performance"),
        ("fusion_type", "Fusion performance"),
        ("augment", "Augmentation performance"),
        ("moco", "MoCo ON/OFF performance"),
    ]
    metrics = ["AUC", "ACC", "F1"]

    for dim, title in dims:
        groups: Dict[str, Dict[str, List[float]]] = {}
        for r in all_rows:
            k = r.get(dim) or "-"
            groups.setdefault(k, {m: [] for m in metrics})
            for m in metrics:
                v = pick_float(r.get(m))
                if v == v:
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
            plt.title(title + " (mean)")
            plt.legend()
            img_path = os.path.join(ASSET_DIR, f"trend_{dim}.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=150)
            plt.close()
            images.append(img_path)
        except Exception as e:
            print(f"[WARN] 绘图失败 {dim}: {e}")

    return images

def generate_controlled_tables(rows: List[Dict[str, Any]]) -> List[str]:
    """
    生成两类控制变量对照表：
    1) 固定 组合 + 增强，对比 moco(off vs on)
    2) 固定 组合 + moco，对比 不同增强
    """
    out: List[str] = []

    # 1) 按 (enc,fus,augment) 对比 moco
    header1 = ["encoder-fusion", "augment", "moco=off(AUC/ACC/F1)", "moco=on(AUC/ACC/F1)"]
    table1: List[List[str]] = []
    # 建索引
    idx: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for r in rows:
        key = (
            normalize_text(r.get("encoder_type")),
            normalize_text(r.get("fusion_type")),
            normalize_text(r.get("augment")),
            normalize_text(r.get("moco") or "off"),
        )
        idx[key] = r

    for enc, fus in FIXED_COMBOS:
        for aug in DEFAULT_AUGMENTS:
            r_off = idx.get((enc, fus, aug, "off"))
            r_on  = idx.get((enc, fus, aug, "on"))
            def fmt(x):
                if not x: return "-"
                return f"{x.get('AUC','-')}/{x.get('ACC','-')}/{x.get('F1','-')}"
            table1.append([f"{enc}-{fus}", aug, fmt(r_off), fmt(r_on)])
    out.append(to_markdown_table("对照表A：固定(组合+增强)，对比 moco(off/on)", header1, table1))

    # 2) 按 (enc,fus,moco) 对比不同增强
    header2 = ["encoder-fusion", "moco", "augment", "AUC", "ACC", "F1"]
    table2: List[List[str]] = []
    for enc, fus in FIXED_COMBOS:
        for moco in DEFAULT_MOCO_STATES:
            for aug in DEFAULT_AUGMENTS:
                r = idx.get((enc, fus, aug, moco))
                table2.append([
                    f"{enc}-{fus}", moco, aug,
                    (r or {}).get("AUC", "-"),
                    (r or {}).get("ACC", "-"),
                    (r or {}).get("F1", "-"),
                ])
    out.append(to_markdown_table("对照表B：固定(组合+MoCo)，对比不同增强", header2, table2))
    return out

def generate_report(base_str: str, planned_cmds: List[Tuple[str, str]]):
    summaries = collect_all_summaries()

    # 统一行
    rows: List[Dict[str, Any]] = []
    for s in summaries:
        rows.append({
            "Run": s.get("run_name") or s.get("_dir"),
            "encoder_type": s.get("encoder_type"),
            "fusion_type": s.get("fusion_type"),
            "feature_type": s.get("feature_type"),
            "augment": s.get("augment"),
            "augment_mode": s.get("augment_mode"),
            "queue_warmup_steps": s.get("queue_warmup_steps"),
            "moco": (s.get("moco") or "off"),
            "AUC": s.get("AUC"),
            "AUPRC": s.get("AUPRC"),
            "ACC": s.get("ACC"),
            "Precision": s.get("Precision"),
            "Recall": s.get("Recall"),
            "F1": s.get("F1"),
        })

    # 趋势图
    img_paths = plot_trends(rows)

    # 报告
    lines: List[str] = []
    lines.append("# 实验报告（控制变量法：moco × augment）")
    lines.append("")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 公共基础命令（入口已自动探测）：`{base_str}`")
    lines.append(f"- 固定组合数量：{len(FIXED_COMBOS)}")
    lines.append(f"- 自变量：moco ∈ {DEFAULT_MOCO_STATES}，augment ∈ {DEFAULT_AUGMENTS}")
    lines.append(f"- 常量：feature_type={CONST_FEATURE_TYPE}，augment_mode={CONST_AUGMENT_MODE}，queue_warmup_steps={CONST_QUEUE_WARMUP}")
    lines.append("")

    lines.append("## 计划执行的命令（顺序）")
    for rn, cmd in planned_cmds:
        lines.append(f"- {rn}")
        lines.append(f"  - `{cmd}`")
    lines.append("")

    # 控制变量对照表
    for block in generate_controlled_tables(rows):
        lines.append(block)

    # 全量明细表
    header_all = ["Run", "encoder_type", "fusion_type", "moco", "augment", "feature_type", "augment_mode", "queue_warmup_steps", "AUC", "AUPRC", "ACC", "Precision", "Recall", "F1"]
    table_rows = []
    for r in rows:
        table_rows.append([r.get(h, "-") for h in header_all])
    lines.append(to_markdown_table("明细：全部记录", header_all, table_rows))

    # 最优推荐
    for key in ["AUC", "ACC", "F1"]:
        lb = build_leaderboard(rows, key)
        topk = lb[:10]
        header2 = ["Rank", key, "Run", "encoder_type", "fusion_type", "moco", "augment"]
        trows = []
        for i, r in enumerate(topk, 1):
            trows.append([
                str(i), str(r.get(key, "-")),
                r.get("Run", "-"),
                r.get("encoder_type", "-"),
                r.get("fusion_type", "-"),
                r.get("moco", "-"),
                r.get("augment", "-"),
            ])
        lines.append(to_markdown_table(f"最优配置（按 {key}）", header2, trows))

    # 趋势图
    if img_paths:
        lines.append("## 趋势图")
        for p in img_paths:
            rel = os.path.relpath(p, start=os.path.dirname(REPORT_MD))
            lines.append(f"![{os.path.basename(p)}]({rel})")

    ensure_dir(os.path.dirname(REPORT_MD))
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告已生成：{REPORT_MD}")

# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="控制变量法：固定组合，仅改变 MoCo 与增强视图，自动运行与汇总。")
    parser.add_argument("--execute", action="store_true", help="是否实际执行命令（默认仅生成报告与命令清单）")
    parser.add_argument("--timeout", type=int, default=0, help="单条命令超时时间（秒，0 不限时）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖基础命令中的 epochs 数值（可选）")
    parser.add_argument("--limit", type=int, default=None, help="仅取前 N 条命令用于执行/清单")

    # 自变量集合（如需临时缩小搜索空间可在此调整）
    parser.add_argument("--moco_states", nargs="+", default=DEFAULT_MOCO_STATES, choices=["off", "on"], help="moco on/off")
    parser.add_argument("--augments", nargs="+", default=DEFAULT_AUGMENTS, help="增强视图：none/random_permute_features/add_noise/attribute_mask/noise_then_mask")

    # 常量（不形成维度）
    parser.add_argument("--feature_type", default=CONST_FEATURE_TYPE)
    parser.add_argument("--augment_mode", default=CONST_AUGMENT_MODE)
    parser.add_argument("--queue_warmup_steps", type=int, default=CONST_QUEUE_WARMUP)

    args = parser.parse_args()

    base_str = detect_entry_command(BASE_STR_USER)

    # 构建计划（笛卡尔积：固定组合 × moco × augment）
    plan: List[Tuple[str, str]] = []
    for enc, fus in FIXED_COMBOS:
        for moco in args.moco_states:
            for aug in args.augments:
                cmd, rn = build_cmd(
                    base_str=base_str,
                    enc=enc, fus=fus,
                    moco=moco,
                    augment=aug,
                    feature_type=args.feature_type,
                    augment_mode=args.augment_mode,
                    queue_warmup_steps=args.queue_warmup_steps,
                    extra_epochs=args.epochs
                )
                plan.append((rn, cmd))

    if args.limit and args.limit > 0:
        plan = plan[: args.limit]

    # 为最后一条追加 --shutdown
    if plan:
        last_rn, last_cmd = plan[-1]
        if "--shutdown" not in last_cmd:
            plan[-1] = (last_rn, f"{last_cmd} --shutdown")
            print(f"[提示] 已为最后一条任务 '{last_rn}' 追加 --shutdown")

    # 可选执行
    if args.execute:
        print("\n将开始执行命令，并在控制台实时显示输出：")
        total = len(plan)
        for i, (rn, cmd) in enumerate(plan, 1):
            print_task_header(i, total, rn, cmd)
            rc = run_command(cmd, timeout=args.timeout)
            status = "OK" if rc == 0 else f"ERR({rc})"
            print(_green(f"[Done {i}/{total}] {rn} -> {status}"))
            if rc != 0:
                try:
                    ensure_dir(os.path.dirname(FAILED_CMDS_LOG))
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(FAILED_CMDS_LOG, "a", encoding="utf-8") as f:
                        f.write(f"[{ts}] run_name={rn} exit_code={rc}\n{cmd}\n\n")
                except Exception as e:
                    print(f"[WARN] 记录失败命令到文件出错: {e}")

    # 生成报告
    generate_report(base_str, plan)

if __name__ == "__main__":
    main()