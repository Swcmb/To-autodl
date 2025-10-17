#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数优化（HPO）脚本：三阶段流程
- 阶段B：粗随机搜索（每任务 N=60，epochs=3），以 AUPRC 为主排序，输出 top10 与历史
- 阶段C：对阶段B的 top10 在邻域精调（epochs=20），启用 online 增强与 mgraph 对抗（入口预置）
- 最终复现：对每任务 top3 配置在 seeds=[0,1,2] 下复现（入口预置）

实现策略：
- 直接复用当前目录的 load_data → Create_model → train_model，进行 5 折评估
- 为每个 trial 覆盖 layer.args 的相关字段，确保 EM 内部读取的超参与 trial 一致
- 严格生成 experiments/hpo_{TIMESTAMP}/{TASK}/ 下的 CSV/JSON 报告与错误日志
"""

import argparse
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# 引入项目内部模块（与 main.py 同步）
# 安全包装：延迟导入并在导入前暂时清空 sys.argv，避免 settings() 解析 HPO 自定义 CLI
import importlib

def _safe_import(mod_name: str):
    import sys
    argv_backup = list(sys.argv)
    try:
        sys.argv = [sys.executable]
        return importlib.import_module(mod_name)
    finally:
        sys.argv = argv_backup

def _ps_settings():
    """安全调用 parms_setting.settings() 获取基础 args"""
    ps = _safe_import("parms_setting")
    import sys as _sys
    argv_backup = list(_sys.argv)
    try:
        _sys.argv = [_sys.executable]
        return ps.settings()
    finally:
        _sys.argv = argv_backup

def _init_autodl_env(args):
    return _safe_import("autodl").init_autodl_env(args)

def _init_logging(run_name=None):
    return _safe_import("log_output_manager").init_logging(run_name=run_name)

def _redirect_print(enable=True):
    return _safe_import("log_output_manager").redirect_print(enable)

def _make_result_run_dir(prefix="data"):
    return _safe_import("log_output_manager").make_result_run_dir(prefix)

def _get_run_paths():
    return _safe_import("log_output_manager").get_run_paths()

def _finalize_run():
    return _safe_import("log_output_manager").finalize_run()

def _load_data(args):
    return _safe_import("data_preprocess").load_data(args)

def _Create_model(args):
    return _safe_import("instantiation").Create_model(args)

def _train_model(model, optimizer, data_o, data_a, train_loader, test_loader, args, fold_idx=None):
    return _safe_import("train").train_model(model, optimizer, data_o, data_a, train_loader, test_loader, args, fold_idx=fold_idx)

# 注意：layer.py 在 import 时会执行 settings() 获取全局 args
# 我们需要在每个 trial 运行前，覆盖其中的字段以保持一致
# 延迟导入 layer，避免其在导入时调用 settings() 与 HPO CLI 冲突
_LAYER_MOD = None
def _get_layer_mod():
    """
    安全延迟导入 layer：
    - 在导入前暂时清空 sys.argv，防止 layer.settings() 解析 HPO 的 --stage 等参数
    - 导入完成后恢复 sys.argv
    - 结果缓存到全局，避免重复导入
    """
    global _LAYER_MOD
    if _LAYER_MOD is not None:
        return _LAYER_MOD
    import importlib, sys
    argv_backup = list(sys.argv)
    try:
        sys.argv = [sys.executable]
        _LAYER_MOD = importlib.import_module("layer")
    finally:
        sys.argv = argv_backup
    return _LAYER_MOD


# ============== 工具函数 ==============

def now_tag() -> str:
    """返回当前时间戳短标签"""
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    """确保目录存在"""
    p.mkdir(parents=True, exist_ok=True)

def update_layer_args(args_obj: Any) -> None:
    """将当前 trial 的关键参数写入 layer.args（全局），以便 EM 内部读取"""
    # 仅覆盖训练相关参数，避免污染不可用项
    la = getattr(_get_layer_mod(), "args", None)
    if la is None:
        return
    for key in [
        "gat_heads", "fusion_heads",
        "proj_dim", "moco_queue", "moco_momentum", "moco_t", "queue_warmup_steps",
        "noise_std", "mask_rate", "augment", "augment_mode", "augment_seed",
        "adv_mode", "adv_on_moco", "adv_eps", "adv_alpha", "adv_steps", "adv_rand_init", "adv_project",
        "adv_agg", "adv_budget", "adv_use_amp", "adv_clip_min", "adv_clip_max",
        "num_views",
    ]:
        if hasattr(args_obj, key):
            setattr(la, key, getattr(args_obj, key))


def derive_seed(*parts: int) -> int:
    """简易派生稳定种子（与 utils.derive_seed 保持一致思想）"""
    mod = 2**32
    acc = 0
    for p in parts:
        acc = (acc * 1000003 + int(p)) % mod
    return int(acc if acc != 0 else 1)


def sample_config(task: str, rng: np.random.Generator) -> Dict[str, Any]:
    """阶段B：随机采样一个配置（与用户确认的粗搜索空间一致）"""
    # 搜索空间
    embed_dim_choices = [32, 64, 128, 256]
    lr_choices = [1e-4, 5e-4, 1e-3]
    wd_choices = [0.0, 5e-5, 5e-4, 1e-3]
    dropout_choices = [0.0, 0.1, 0.2]
    batch_choices = [16, 32, 64]
    moco_queue_choices = [1024, 4096, 8192]
    moco_momentum_choices = [0.95, 0.99, 0.999]
    moco_t_choices = [0.07, 0.1, 0.2]
    proj_dim_choices = [32, 64, 128, None]
    augment_choices = ["none", "random_permute_features", "add_noise", "attribute_mask", "noise_then_mask"]
    noise_std_choices = [0.005, 0.01, 0.02]
    mask_rate_choices = [0.05, 0.1, 0.2]
    # 损失权重（alpha/beta/gamma）
    alpha_choices = [0.5, 1.0, 2.0]
    beta_choices = [0.1, 0.5, 1.0]
    gamma_choices = [0.0, 0.1, 0.5]
    # 视图数（MoCo）
    num_views_choices = [2, 3]

    embed_dim = rng.choice(embed_dim_choices)
    hidden1_choices = [max(32, embed_dim // 2), embed_dim, min(2 * embed_dim, 512)]
    hidden2_choices = [max(16, embed_dim // 4), max(32, embed_dim // 2), embed_dim]

    cfg = {
        "task_type": task,
        "dimensions": int(embed_dim),
        "hidden1": int(rng.choice(hidden1_choices)),
        "hidden2": int(rng.choice(hidden2_choices)),
        "decoder1": int(rng.choice([256, 512])),
        "lr": float(rng.choice(lr_choices)),
        "weight_decay": float(rng.choice(wd_choices)),
        "dropout": float(rng.choice(dropout_choices)),
        "batch": int(rng.choice(batch_choices)),
        "moco_queue": int(rng.choice(moco_queue_choices)),
        "moco_momentum": float(rng.choice(moco_momentum_choices)),
        "moco_t": float(rng.choice(moco_t_choices)),
        "proj_dim": (None if (c := rng.choice(proj_dim_choices)) is None else int(c)),
        "augment": str(rng.choice(augment_choices)),
        "augment_mode": "static",  # 阶段B固定 static
        "noise_std": float(rng.choice(noise_std_choices)),
        "mask_rate": float(rng.choice(mask_rate_choices)),
        "alpha": float(rng.choice(alpha_choices)),   # 别名到 loss_ratio1
        "beta": float(rng.choice(beta_choices)),     # 别名到 loss_ratio2
        "gamma": float(rng.choice(gamma_choices)),   # 别名到 loss_ratio3
        "num_views": int(rng.choice(num_views_choices)),
        # 阶段B关闭对抗
        "adv_mode": "none",
        "adv_on_moco": False,
        "queue_warmup_steps": int(rng.choice([0, 200])),
        "gat_heads": int(rng.choice([2, 4])),
        "fusion_heads": int(rng.choice([2, 4])),
    }
    # 约束：若 augment=none，则忽略 noise_std/mask_rate
    if cfg["augment"] in ("none", "null", ""):
        cfg["noise_std"] = 0.0
        cfg["mask_rate"] = 0.0
    # proj_dim 兜底：None 使用 hidden2
    if (cfg["proj_dim"] is None) or (int(cfg["proj_dim"]) <= 0 if cfg["proj_dim"] is not None else False):
        cfg["proj_dim"] = int(cfg["hidden2"])
    return cfg


def trial_to_args(base_args: Any, cfg: Dict[str, Any], seed: int, in_file: str, neg_file: str, epochs: int) -> Any:
    """将采样配置写入 args 对象并返回"""
    # 基本任务文件
    base_args.task_type = cfg["task_type"]
    base_args.in_file = in_file
    base_args.neg_sample = neg_file
    base_args.validation_type = "5_cv1"
    # 训练与结构
    base_args.lr = cfg["lr"]
    base_args.learning_rate = cfg["lr"]
    base_args.weight_decay = cfg["weight_decay"]
    base_args.dropout = cfg["dropout"]
    base_args.batch = cfg["batch"]
    base_args.epochs = int(epochs)
    base_args.dimensions = cfg["dimensions"]
    base_args.embed_dim = cfg["dimensions"]
    base_args.hidden1 = cfg["hidden1"]
    base_args.hidden2 = cfg["hidden2"]
    base_args.decoder1 = cfg["decoder1"]
    base_args.proj_dim = cfg["proj_dim"]
    base_args.gat_heads = cfg["gat_heads"]
    base_args.fusion_heads = cfg["fusion_heads"]
    base_args.num_views = cfg["num_views"]
    # 损失权重
    base_args.loss_ratio1 = cfg["alpha"]
    base_args.loss_ratio2 = cfg["beta"]
    base_args.loss_ratio3 = cfg["gamma"]
    base_args.alpha = cfg["alpha"]
    base_args.beta = cfg["beta"]
    base_args.gamma = cfg["gamma"]
    # 增强
    base_args.augment_mode = cfg["augment_mode"]
    base_args.augment = cfg["augment"]
    base_args.noise_std = cfg["noise_std"]
    base_args.mask_rate = cfg["mask_rate"]
    base_args.augment_seed = seed
    base_args.feature_type = "one_hot"  # 与用户基线一致
    base_args.similarity_threshold = 0.5
    # MoCo
    base_args.moco_queue = cfg["moco_queue"]
    base_args.moco_momentum = cfg["moco_momentum"]
    base_args.moco_t = cfg["moco_t"]
    base_args.queue_warmup_steps = cfg["queue_warmup_steps"]
    # 对抗
    base_args.adv_mode = cfg["adv_mode"]
    base_args.adv_on_moco = cfg["adv_on_moco"]
    # 其他
    base_args.seed = seed
    base_args.cuda = True
    base_args.run_name = None  # 由外层设置
    # 并行与数据保存
    base_args.num_workers = -1
    base_args.threads = 32
    base_args.save_datasets = False
    return base_args


def parse_epoch_csv_loss(last_csv_path: Path) -> Tuple[float, float]:
    """
    从每折的 CSV 中解析首/末 epoch 的 loss_train，返回 (first_loss, last_loss)
    若无法读取，返回 (math.nan, math.nan)
    """
    try:
        txt = last_csv_path.read_text(encoding="utf-8").strip().splitlines()
        if len(txt) <= 1:
            return (math.nan, math.nan)
        # 第一数据行与最后数据行
        first = txt[1].split(",")
        last = txt[-1].split(",")
        # CSV 列固定顺序：epoch,loss_train,...
        return (float(first[1]), float(last[1]))
    except Exception:
        return (math.nan, math.nan)


def run_one_trial(task: str, trial_id: int, cfg: Dict[str, Any], epochs: int, exp_task_dir: Path, in_file: str, neg_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    执行单个 trial：返回 (metrics, meta)
    metrics：包含 AUROC/AUPRC/F1/Loss 的 mean/std
    meta：包含 run_name、log 路径、时间、是否重试、错误信息等
    """
    # 运行名：task_trial_{trial_id}
    run_name = f"{task}_trial_{trial_id}"
    meta: Dict[str, Any] = {"run_name": run_name, "trial_id": trial_id, "retry": False, "error": None}
    start_t = time.time()

    # 构造 args
    args_obj = _ps_settings()
    # 派生种子（确保每 trial 稳定）
    seed = derive_seed(int(getattr(args_obj, "seed", 0)), trial_id)
    args_obj = trial_to_args(args_obj, cfg, seed, in_file, neg_file, epochs)
    args_obj.run_name = run_name

    # 覆盖到 layer.args
    update_layer_args(args_obj)

    # 初始化环境与日志
    _init_autodl_env(args_obj)
    logger = _init_logging(run_name=args_obj.run_name)
    _redirect_print(True)
    _make_result_run_dir("data")

    # 加载数据与 5 折评估
    data_o_folds, data_a_folds, train_loaders, test_loaders = _load_data(args_obj)

    # 逐折训练与测试
    all_fold_results: List[Dict[str, Any]] = []
    for fold in range(5):
        model, optimizer = _Create_model(args_obj)
        fold_res = _train_model(model, optimizer, data_o_folds[fold], data_a_folds[fold], train_loaders[fold], test_loaders[fold], args_obj, fold_idx=fold+1)
        all_fold_results.append(fold_res)

    # 计算汇总
    def _collect(key: str) -> Tuple[float, float]:
        xs = [float(r[key]) for r in all_fold_results]
        return (float(np.mean(xs)), float(np.std(xs)))

    auroc_mean, auroc_std = _collect("auroc")
    auprc_mean, auprc_std = _collect("auprc")
    f1_mean, f1_std = _collect("f1")
    loss_mean, loss_std = _collect("loss")

    # 故障检测：读取最后一个折的 per-epoch CSV 判断发散
    paths = _get_run_paths()
    run_id = paths.get("run_id") or ""
    last_csv = exp_task_dir / f"{run_name}_fold_last_metrics_{run_id}.marker"  # 我们在本地创建一个标记文件，同时解析 OUTPUT 中真实 CSV
    try:
        # 找到一个 CSV 文件（例如 fold_1）
        csv_dir = Path(paths.get("run_result_dir") or "")
        csv_sample = None
        for fold_idx in range(1, 6):
            p = csv_dir / "metrics" / f"train_epoch_metrics_fold_{fold_idx}_{run_id}.csv"
            if p.exists():
                csv_sample = p
                break
        if csv_sample is not None:
            first_loss, last_loss = parse_epoch_csv_loss(csv_sample)
            # 发散判定：首末比 > 10 且末值更大
            if (not math.isnan(first_loss)) and (not math.isnan(last_loss)) and (first_loss > 0.0) and (last_loss / first_loss > 10.0):
                meta["divergent"] = True
            else:
                meta["divergent"] = False
        else:
            meta["divergent"] = False
        # 标记文件（便于排查）
        ensure_dir(exp_task_dir)
        last_csv.write_text(f"csv_checked={csv_sample is not None}\n", encoding="utf-8")
    except Exception:
        meta["divergent"] = False

    # 输出本次 trial 的 CSV 行（历史文件由外层写）
    _finalize_run()
    meta["time_s"] = round(time.time() - start_t, 3)
    meta["log_path"] = str(Path(paths.get("log_dir")) / f"EM_{run_name}_{run_id}.log") if run_id else None

    metrics = {
        "AUROC_mean": auroc_mean, "AUROC_std": auroc_std,
        "AUPRC_mean": auprc_mean, "AUPRC_std": auprc_std,
        "F1_mean": f1_mean, "F1_std": f1_std,
        "Loss_mean": loss_mean, "Loss_std": loss_std,
        "run_id": run_id,
    }
    return metrics, meta


def run_trial_with_retry(task: str, trial_id: int, cfg: Dict[str, Any], epochs: int, exp_task_dir: Path, in_file: str, neg_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """包裹一次重试逻辑：NaN/Inf 或发散时将 lr*=0.5 重试一次；OOM/其他异常记录错误并返回"""
    try:
        metrics, meta = run_one_trial(task, trial_id, cfg, epochs, exp_task_dir, in_file, neg_file)
        # 简单 NaN/Inf 检查与发散检查
        def _bad(x: float) -> bool:
            return (x is None) or (not math.isfinite(x)) or (x < 0)
        if _bad(metrics["AUPRC_mean"]) or meta.get("divergent", False):
            # 重试一次：lr *= 0.5
            cfg_retry = dict(cfg)
            cfg_retry["lr"] = float(cfg["lr"]) * 0.5
            meta["retry"] = True
            metrics, meta2 = run_one_trial(task, trial_id, cfg_retry, epochs, exp_task_dir, in_file, neg_file)
            # 合并关键信息
            meta.update({"retry_run_name": meta2.get("run_name"), "retry_log_path": meta2.get("log_path")})
        return metrics, meta
    except RuntimeError as e:
        # 可能是 OOM
        msg = str(e)
        meta = {"run_name": f"{task}_trial_{trial_id}", "trial_id": trial_id, "retry": False, "error": "OOM" if "out of memory" in msg.lower() else "RuntimeError", "time_s": None}
        # 写 error report
        ensure_dir(exp_task_dir)
        (exp_task_dir / "error_report.txt").write_text(f"{meta['run_name']}, {meta['error']}, detail={msg}\n", encoding="utf-8")
        return {
            "AUROC_mean": 0.0, "AUROC_std": 0.0,
            "AUPRC_mean": 0.0, "AUPRC_std": 0.0,
            "F1_mean": 0.0, "F1_std": 0.0,
            "Loss_mean": 0.0, "Loss_std": 0.0,
            "run_id": None,
        }, meta
    except Exception as e:
        meta = {"run_name": f"{task}_trial_{trial_id}", "trial_id": trial_id, "retry": False, "error": "Exception", "time_s": None}
        ensure_dir(exp_task_dir)
        # 保存错误
        err_path = exp_task_dir / f"{task}_trial_{trial_id}.err"
        err_path.write_text("".join(traceback.format_exception(e)), encoding="utf-8")
        (exp_task_dir / "error_report.txt").write_text(f"{meta['run_name']}, Exception, err_path={err_path}\n", encoding="utf-8")
        return {
            "AUROC_mean": 0.0, "AUROC_std": 0.0,
            "AUPRC_mean": 0.0, "AUPRC_std": 0.0,
            "F1_mean": 0.0, "F1_std": 0.0,
            "Loss_mean": 0.0, "Loss_std": 0.0,
            "run_id": None,
        }, meta


def sort_and_top(trials: List[Dict[str, Any]], topn: int = 10) -> List[Dict[str, Any]]:
    """按 AUPRC_mean 降序，tie-breaker: AUROC_mean、F1_mean"""
    def _key(d: Dict[str, Any]) -> Tuple[float, float, float]:
        return (float(d["AUPRC_mean"]), float(d["AUROC_mean"]), float(d["F1_mean"]))
    trials_sorted = sorted(trials, key=_key, reverse=True)
    return trials_sorted[:topn]


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["LDA", "MDA", "LMI"], choices=["LDA", "MDA", "LMI"], help="选择需要优化的任务集合")
    parser.add_argument("--stage", type=str, default="B", choices=["B", "C", "final"], help="选择阶段：B 粗随机搜索；C 精调；final 最终复现")
    parser.add_argument("--trials", type=int, default=60, help="阶段B每任务试验数")
    parser.add_argument("--epochs", type=int, default=3, help="阶段B训练轮数（建议 3）")
    parser.add_argument("--seed_base", type=int, default=0, help="基础种子")
    # 阶段C/最终复现参数（入口预置）
    parser.add_argument("--epochs_refine", type=int, default=20, help="阶段C精调的训练轮数")
    parser.add_argument("--final_seeds", nargs="+", type=int, default=[0, 1, 2], help="最终复现的 seeds 列表")
    args_cli = parser.parse_args()

    # 根目录与 experiments 目录
    proj_root = Path(__file__).resolve().parent.parent
    exp_root = proj_root / "experiments" / f"hpo_{now_tag()}"
    ensure_dir(exp_root)

    # 基线文件路径（与用户命令一致的数据文件）
    dataset_dir = Path(__file__).resolve().parent / "dataset1"
    file_map = {
        "LDA": {"in_file": str(dataset_dir / "LDA.edgelist"), "neg_file": str(dataset_dir / "non_LDA.edgelist")},
        "MDA": {"in_file": str(dataset_dir / "MDA.edgelist"), "neg_file": str(dataset_dir / "non_MDA.edgelist")},
        "LMI": {"in_file": str(dataset_dir / "LMI.edgelist"), "neg_file": str(dataset_dir / "non_LMI.edgelist")},
    }

    # 阶段B：粗随机搜索
    if args_cli.stage == "B":
        for task in args_cli.tasks:
            exp_task_dir = exp_root / task
            ensure_dir(exp_task_dir)

            rng = np.random.default_rng(args_cli.seed_base + hash(task) % 10000)

            history_rows: List[List[Any]] = []
            trial_metrics: List[Dict[str, Any]] = []

            for trial_id in range(1, args_cli.trials + 1):
                cfg = sample_config(task, rng)
                metrics, meta = run_trial_with_retry(task, trial_id, cfg, args_cli.epochs, exp_task_dir, file_map[task]["in_file"], file_map[task]["neg_file"])

                # 记录历史行
                history_rows.append([
                    meta.get("run_name"),
                    json.dumps(cfg, ensure_ascii=False),
                    metrics["AUPRC_mean"], metrics["AUPRC_std"],
                    metrics["AUROC_mean"], metrics["AUROC_std"],
                    metrics["F1_mean"], metrics["F1_std"],
                    metrics["Loss_mean"], metrics["Loss_std"],
                    meta.get("time_s"), meta.get("log_path"), meta.get("retry"), meta.get("error")
                ])
                # trial_metrics 用于排序
                tm = dict(metrics)
                tm["run_name"] = meta.get("run_name")
                tm["config_json"] = json.dumps(cfg, ensure_ascii=False)
                tm["time_s"] = meta.get("time_s")
                tm["log_path"] = meta.get("log_path")
                trial_metrics.append(tm)

            # 写历史 CSV
            write_csv(
                exp_task_dir / f"opt_history_task_{task}.csv",
                header=[
                    "run_name", "config_json",
                    "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                    "time_s", "log_path", "retry", "error"
                ],
                rows=history_rows
            )

            # Top10
            top10 = sort_and_top(trial_metrics, topn=10)
            top_rows = [[
                m["run_name"],
                m["config_json"],
                m["AUPRC_mean"], m["AUPRC_std"],
                m["AUROC_mean"], m["AUROC_std"],
                m["F1_mean"], m["F1_std"],
                m["Loss_mean"], m["Loss_std"],
                m["time_s"], m["log_path"]
            ] for m in top10]
            write_csv(
                exp_task_dir / f"configs_top10_task_{task}.csv",
                header=[
                    "run_name", "config_json",
                    "AUPRC_mean", "AUPRC_std", "AUROC_mean", "AUROC_std", "F1_mean", "F1_std", "Loss_mean", "Loss_std",
                    "time_s", "log_path"
                ],
                rows=top_rows
            )

            # 初版 best_configs_final.json（取 top3）
            best3 = top10[:3]
            best_payload = {
                "task": task,
                "top3": [{
                    "run_name": m["run_name"],
                    "config": json.loads(m["config_json"]),
                    "metrics": {
                        "AUPRC_mean": m["AUPRC_mean"], "AUROC_mean": m["AUROC_mean"], "F1_mean": m["F1_mean"]
                    },
                    "repro_cli": f"python hyperparameter-tuning/main.py --file {file_map[task]['in_file']} --neg_sample {file_map[task]['neg_file']} --validation_type 5-cv1 --task_type {task} --feature_type one_hot --similarity_threshold 0.5 --embed_dim {json.loads(m['config_json'])['dimensions']} --learning_rate {json.loads(m['config_json'])['lr']} --weight_decay {json.loads(m['config_json'])['weight_decay']} --epochs {args_cli.epochs} --alpha {json.loads(m['config_json'])['alpha']} --beta {json.loads(m['config_json'])['beta']} --gamma {json.loads(m['config_json'])['gamma']}"
                } for m in best3],
                "note": "阶段B随机搜索结果，阶段C将围绕这些配置做精调（epochs=20，启用online增强与mgraph对抗）"
            }
            (exp_task_dir / "best_configs_final.json").write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            # 预留报告文件
            (exp_task_dir / f"summary_task_{task}.md").write_text("# 阶段B结果\n\n- 已生成 top10 与历史CSV；待阶段C与最终复现补充。\n", encoding="utf-8")

        print(f"[HPO] 阶段B完成。输出位于: {exp_root}")

    elif args_cli.stage == "C":
        # 阶段C入口预置：读取阶段B top10，围绕邻域构造小网格/贝叶斯（可后续扩展）
        print("[HPO] 阶段C入口预置：请先运行阶段B以生成 top10，再在此基础上精调（epochs_refine，开启online与mgraph）。")
        print("暂未自动执行精调，以避免误覆盖实验文件。")

    elif args_cli.stage == "final":
        # 最终复现入口预置
        print("[HPO] 最终复现入口预置：按 seeds 列表对每任务 top3 配置运行完整 5-fold，写入 summary 与 artifacts。")
        print("暂未自动执行复现，以避免误覆盖实验文件。")


if __name__ == "__main__":
    main()