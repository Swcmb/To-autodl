import os
import re
import argparse
import subprocess
from datetime import datetime

# 配置区
A_LIST = ["csglmd", "mgacmda", "gat", "gt", "gat_gt_serial", "gat_gt_parallel"]
B_LIST = ["basic", "dot", "additive", "self_attn", "gat_fusion", "gt_fusion"]

# 用户提供的基础命令（入口由脚本自动探测后替换）
BASE_STR_USER = "python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5"

RESULT_ROOT = os.path.join("OUTPUT", "result")
REPORT_MD = "test.md"

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

def parse_summary_file(fp: str) -> dict:
    """
    解析 result_summary_*.txt，尽量提取：
    - run_name
    - encoder_type / fusion_type
    - AUC / ACC / Precision / Recall / F1
    兼容键值对、用冒号/等号/制表符、以及行内包含数值的情况。
    """
    data = {
        "run_name": None,
        "encoder_type": None,
        "fusion_type": None,
        "AUC": None,
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

    # 通用键名映射（大小写不敏感）
    key_map = {
        "run_name": "run_name",
        "encoder": "encoder_type",
        "encoder_type": "encoder_type",
        "fusion": "fusion_type",
        "fusion_type": "fusion_type",
        "auc": "AUC",
        "acc": "ACC",
        "accuracy": "ACC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "f1_score": "F1"
    }

    num_pattern = re.compile(r"(-?\d+(?:\.\d+)?)")

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # 尝试使用分隔符切割
        parts = re.split(r"[:=|\t]", line, maxsplit=1)
        if len(parts) == 2:
            k, v = parts[0].strip().lower(), parts[1].strip()
            for cand, norm in key_map.items():
                if cand == k or k.endswith(cand) or cand in k:
                    # 提取数值或直接记录
                    if norm in ["AUC", "ACC", "Precision", "Recall", "F1"]:
                        m = num_pattern.search(v)
                        if m:
                            data[norm] = m.group(1)
                    else:
                        data[norm] = v
        else:
            # 行内扫描可能的键与数值
            low = line.lower()
            for cand, norm in key_map.items():
                if cand in low:
                    if norm in ["AUC", "ACC", "Precision", "Recall", "F1"]:
                        m = num_pattern.search(line)
                        if m:
                            data[norm] = m.group(1)
                    else:
                        # 非数值字段尽量取等号/冒号后的文本
                        m2 = re.search(rf"{cand}\s*[:=]\s*(.+)", low)
                        if m2:
                            data[norm] = m2.group(1).strip()

    # 规范化
    for k in list(data.keys()):
        if isinstance(data.get(k), str):
            data[k] = normalize_text(data[k])

    return data

def collect_all_summaries() -> list:
    """
    遍历 OUTPUT/result，收集每个目录下的 result_summary_*.txt
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
                # 兜底：附加目录名线索
                info["_dir"] = sub
                collected.append(info)
    return collected

def to_markdown_commands(title: str, commands: list) -> str:
    out = []
    out.append(f"## {title}")
    out.append("")
    for name, cmd in commands:
        out.append(f"### {name}")
        out.append("")
        out.append(f"`{cmd}`")
        out.append("")
    return "\n".join(out)

def table_row(cols: list) -> str:
    return "| " + " | ".join(cols) + " |"

def to_markdown_results(title: str, rows: list) -> str:
    """
    rows: list of [Run, Encoder, Fusion, AUC, ACC, Precision, Recall, F1]
    """
    out = []
    out.append(f"### 实验结果 - {title}")
    out.append("")
    header = ["Run", "Encoder", "Fusion", "AUC", "ACC", "Precision", "Recall", "F1"]
    out.append(table_row(header))
    out.append(table_row(["---"] * len(header)))
    for r in rows:
        out.append(table_row([c if c is not None and c != "" else "-" for c in r]))
    out.append("")
    return "\n".join(out)

def best_match_summary(summaries: list, run_name: str = None, encoder: str = None, fusion: str = None) -> dict:
    """
    匹配顺序：
    1) run_name 精确匹配（忽略大小写）
    2) encoder + fusion 同时匹配（忽略大小写）
    3) encoder 或 fusion 单独匹配
    匹配失败返回空 dict
    """
    def eq(a, b):
        return a and b and a.strip().lower() == b.strip().lower()

    # 1) run_name
    if run_name:
        for s in summaries:
            if eq(s.get("run_name"), run_name):
                return s

    # 2) encoder + fusion
    if encoder and fusion:
        for s in summaries:
            if eq(s.get("encoder_type"), encoder) and eq(s.get("fusion_type"), fusion):
                return s

    # 3) 单独匹配
    if encoder:
        for s in summaries:
            if eq(s.get("encoder_type"), encoder):
                return s
    if fusion:
        for s in summaries:
            if eq(s.get("fusion_type"), fusion):
                return s

    return {}

def build_commands(base_str: str):
    encoder_cmds = []
    for enc in A_LIST:
        cmd = f"{base_str} --encoder_type {enc} --run_name {enc}"
        encoder_cmds.append((enc, cmd))

    fusion_cmds = []
    for fus in B_LIST:
        cmd = f"{base_str} --fusion_type {fus} --run_name {fus}"
        fusion_cmds.append((fus, cmd))

    cross_cmds = []
    for enc in A_LIST:
        for fus in B_LIST:
            rn = f"{enc}-{fus}"
            cmd = f"{base_str} --encoder_type {enc} --fusion_type {fus} --run_name {rn}"
            cross_cmds.append((f"{enc}+{fus}", cmd))

    return encoder_cmds, fusion_cmds, cross_cmds

def run_command(cmd: str, timeout: int = 0) -> int:
    """
    执行命令并实时打印输出；返回进程退出码。
    timeout: 单位秒，0 表示不限时。
    """
    print("=" * 80)
    print(f"[RUN] {cmd}")
    print("-" * 80)
    try:
        # 使用 shell=True 以兼容包含参数的整行命令；依赖用户已在外部激活环境
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
        print("[ERROR] 找不到可执行命令，请确认 python 是否在当前环境可用。")
        return -127
    except Exception as e:
        print(f"[ERROR] 运行失败：{e}")
        return -1

def generate_report(base_str: str):
    encoder_cmds, fusion_cmds, cross_cmds = build_commands(base_str)

    # 收集结果
    summaries = collect_all_summaries()

    # 顶部信息
    lines = []
    lines.append("# 实验报告")
    lines.append("")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 公共参数：`{base_str}`")
    lines.append("")

    # 编码器类型 - 命令
    lines.append(to_markdown_commands("编码器类型", encoder_cmds))

    # 编码器类型 - 结果
    rows = []
    for name, _cmd in encoder_cmds:
        s = best_match_summary(summaries, run_name=name, encoder=name, fusion=None)
        rows.append([
            name,
            name,
            "-",
            s.get("AUC", "-") if s else "-",
            s.get("ACC", "-") if s else "-",
            s.get("Precision", "-") if s else "-",
            s.get("Recall", "-") if s else "-",
            s.get("F1", "-") if s else "-"
        ])
    lines.append(to_markdown_results("编码器类型", rows))

    # 图融合类型 - 命令
    lines.append(to_markdown_commands("图融合类型", fusion_cmds))

    # 图融合类型 - 结果
    rows = []
    for name, _cmd in fusion_cmds:
        s = best_match_summary(summaries, run_name=name, encoder=None, fusion=name)
        rows.append([
            name,
            "-",
            name,
            s.get("AUC", "-") if s else "-",
            s.get("ACC", "-") if s else "-",
            s.get("Precision", "-") if s else "-",
            s.get("Recall", "-") if s else "-",
            s.get("F1", "-") if s else "-"
        ])
    lines.append(to_markdown_results("图融合类型", rows))

    # 交叉实验 - 命令
    lines.append("## 编码器类型+图融合类型（交叉实验）")
    lines.append("")
    enc_cmds, fus_cmds, cross_cmds2 = build_commands(base_str)
    for name, cmd in cross_cmds2:
        lines.append(f"### {name.replace('+', ' 和 ')}")
        lines.append("")
        lines.append(f"`{cmd}`")
        lines.append("")

    # 交叉实验 - 结果
    rows = []
    for name, _cmd in cross_cmds2:
        enc, fus = name.split("+")
        run_name = f"{enc}-{fus}"
        s = best_match_summary(summaries, run_name=run_name, encoder=enc, fusion=fus)
        rows.append([
            run_name,
            enc,
            fus,
            s.get("AUC", "-") if s else "-",
            s.get("ACC", "-") if s else "-",
            s.get("Precision", "-") if s else "-",
            s.get("Recall", "-") if s else "-",
            s.get("F1", "-") if s else "-"
        ])
    lines.append(to_markdown_results("编码器+图融合（交叉实验）", rows))

    # 写入报告（覆盖）
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"报告已生成：{REPORT_MD}")

def main():
    parser = argparse.ArgumentParser(description="生成实验报告，并可选择执行命令以在控制台显示输出。")
    parser.add_argument("--execute", choices=["none", "encoders", "fusions", "cross", "all"], default="none",
                        help="选择要执行的命令集合（默认 none，仅生成报告）")

    parser.add_argument("--timeout", type=int, default=0, help="单条命令超时时间（秒，0 表示不限时）")
    args = parser.parse_args()

    base_str = detect_entry_command(BASE_STR_USER)

    # 生成报告
    generate_report(base_str)

    # 执行命令（可选）
    if args.execute != "none":
        encoder_cmds, fusion_cmds, cross_cmds = build_commands(base_str)
        to_run = []
        if args.execute in ("encoders", "all"):
            to_run.extend(encoder_cmds)
        if args.execute in ("fusions", "all"):
            to_run.extend(fusion_cmds)
        if args.execute in ("cross", "all"):
            to_run.extend(cross_cmds)



        # 对最终将被执行的“最后一条命令”追加 --shutdown
        if to_run:
            last_name, last_cmd = to_run[-1]
            if "--shutdown" not in last_cmd:
                shutdown_cmd = f"{last_cmd} --shutdown"
                to_run[-1] = (last_name, shutdown_cmd)
                print(f"\n[提示] 已为最后一条任务 '{last_name}' 追加 --shutdown")

        print("\n将开始执行命令，并在控制台实时显示输出：")
        for name, cmd in to_run:
            print(f"\n[任务] {name}")
            rc = run_command(cmd, timeout=args.timeout)
            print(f"[完成] {name} -> 返回码 {rc}")

if __name__ == "__main__":
    main()