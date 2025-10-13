import torch  # 导入PyTorch
import numpy as np  # 导入NumPy库，用于进行科学计算，特别是数组操作
from parms_setting import settings  # 从本地的parms_setting.py文件导入settings函数，用于获取所有超参数
import os  # 导入os模块，用于与操作系统交互，如此处设置环境变量
from data_preprocess import load_data  # 从本地的data_preprocess.py文件导入load_data函数，用于加载和预处理数据
from instantiation import Create_model  # 从本地的instantiation.py文件导入Create_model函数，用于创建模型和优化器
from train import train_model  # 从本地的train.py文件导入train_model函数，用于执行模型的训练和评估流程
from data_preprocess import get_fold_data  # 导入get_fold_data函数，用于获取指定折的数据
import os  # 导入os模块，用于与操作系统交互，如此处设置环境变量
import numpy as np
# 集中日志与结果管理
import argparse
import platform  # 检测操作系统平台
try:
    import psutil  # CPU亲和设置（可选）
except Exception:
    psutil = None
from log_output_manager import (
    init_logging,
    redirect_print,
    make_result_run_dir,
    finalize_run,
    perform_shutdown_if_linux,
    get_logger,
    save_result_text,
    get_run_paths
)

# 参数改由 EM/parms_setting.py 统一解析（包含 --run_name 与 --shutdown）

def _detect_linux_numa_node0_cpus():
    """
    返回Linux下NUMA node0的CPU列表；若不可用则返回None。
    """
    try:
        nodes_path = "/sys/devices/system/node"
        if not os.path.isdir(nodes_path):
            return None
        node0 = os.path.join(nodes_path, "node0")
        if not os.path.isdir(node0):
            return None
        cpu_list = []
        for name in os.listdir(node0):
            if name.startswith("cpu") and name[3:].isdigit():
                cpu_list.append(int(name[3:]))
        return sorted(cpu_list) if cpu_list else None
    except Exception:
        return None

def _set_cpu_affinity_linux(cpus):
    """
    将当前进程的CPU亲和性绑定到给定cpus列表。
    使用psutil优先；无psutil则尝试os.sched_setaffinity。
    """
    try:
        if psutil is not None:
            p = psutil.Process(os.getpid())
            p.cpu_affinity(cpus)
            return True
    except Exception:
        pass
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cpus))
            return True
    except Exception:
        pass
    return False

def setup_parallelism(threads: int) -> None:
    """
    统一设置数值计算后端线程数，避免线程风暴。
    不调节 torch.set_num_threads（保持GPU训练不受影响）。
    同时可选启用Linux下的CPU亲和/NUMA绑定（由环境变量控制）。
    """
    t = int(max(1, min(32, threads)))
    # 统一设置底层库线程数
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
        os.environ[k] = str(t)

    # 可选：CPU亲和/NUMA绑定（仅Linux）
    try:
        if platform.system().lower() == "linux":
            use_aff = os.environ.get("EM_USE_NUMA") == "1" or os.environ.get("EM_CPU_AFFINITY") == "1"
            if use_aff:
                cpus = _detect_linux_numa_node0_cpus()
                if not cpus:
                    total = os.cpu_count() or 32
                    cpus = list(range(min(t, total)))
                ok = _set_cpu_affinity_linux(cpus)
                # 简要记录亲和结果（不会影响GPU训练）
                print(f"[AFFINITY] enabled={ok} cpus={cpus[:8]}... total={len(cpus)}")
    except Exception:
        # 忽略亲和设置异常，避免影响主流程
        pass

# 初始化集中日志（文件+控制台），日志开头记录完整命令
# 先解析全部参数（含 run_name、shutdown）
args = settings()
# Linux 默认启用 NUMA/亲和开关（仅影响CPU亲和，不影响GPU）
try:
    import platform as _plat
    if _plat.system().lower() == "linux":
        if os.environ.get("EM_USE_NUMA") is None and os.environ.get("EM_CPU_AFFINITY") is None:
            os.environ["EM_USE_NUMA"] = "1"
except Exception:
    pass

# 统一并行线程设置（自动探测并裁剪至32）
setup_parallelism(getattr(args, "threads", 32))

# 将关键并行参数同步到环境变量，供下游CPU并行计算读取（固化默认：workers=8, chunk=20000）
try:
    _threads = int(getattr(args, "threads", 32))
    _workers = int(getattr(args, "num_workers", -1))
    if _workers == -1:
        _workers = min(8, max(1, _threads))
    _chunk = int(getattr(args, "chunk_size", 0))
    if _chunk in (0, None):
        _chunk = 20000
    os.environ["EM_THREADS"] = str(min(32, max(1, _threads)))
    os.environ["EM_WORKERS"] = str(min(32, max(0, _workers)))
    os.environ["EM_CHUNK_SIZE"] = str(max(1, _chunk))
except Exception:
    # 防御性处理，避免启动失败
    os.environ.setdefault("EM_THREADS", "32")
    os.environ.setdefault("EM_WORKERS", "8")
    os.environ.setdefault("EM_CHUNK_SIZE", "20000")
# 初始化集中日志（文件+控制台），带 run_name
logger = init_logging(run_name=args.run_name)
# 重定向所有 print 到日志，同时保留控制台输出
redirect_print(True)
# 创建当前运行结果目录（data_时间戳）并记录
make_result_run_dir("data")
logger.info("Initialized logging and result directory.")

# 将后续 print 重定向到 logger.info，避免控制台重复输出
def _print_to_logger(*args, **kwargs):
    try:
        msg = " ".join(str(x) for x in args)
    except Exception:
        msg = " ".join(map(str, args))
    logger.info(msg)
print = _print_to_logger

# 设置程序使用的GPU设备
# "CUDA_VISIBLE_DEVICES"是一个环境变量，用于指定哪些GPU可以被CUDA应用程序看到
# "0"表示只使用系统中编号为0的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# parameters setting  # 注释：参数设置（已提前解析，此处无需重复）
# args 已在日志初始化前由 settings() 获取

# 检查CUDA（GPU计算）是否可用，并根据args.no_cuda标志来决定是否使用GPU
args.cuda = not args.no_cuda and torch.cuda.is_available()
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Using CUDA: {args.cuda}")
if torch.cuda.is_available():
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
else:
    logger.info("CUDA not available, using CPU")
np.random.seed(args.seed)  # 为NumPy设置随机种子，以确保随机数生成是可复现的
torch.manual_seed(args.seed)  # 为PyTorch在CPU上的操作设置随机种子，保证结果一致性
# 修复第31行的语法错误
if args.cuda:  # 如果确定使用CUDA
    torch.cuda.manual_seed(args.seed)  # 也为PyTorch在GPU上的操作设置随机种子
    # GPU 性能优化：启用 cuDNN benchmark 与 TF32；PyTorch>=2.0 设定更高的 float32 matmul 精度
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# load data  # 注释：加载数据
# 调用load_data函数，传入参数对象args
# 该函数会返回处理好的图数据对象（原始图和对抗图）以及所有折的训练和测试数据加载器
data_o_folds, data_a_folds, train_loaders, test_loaders = load_data(args)

# 存储每一折的结果
all_fold_results = []
logger.info("Starting 5-fold cross validation...")

for fold in range(5):
    # 按折使用对应的图数据与加载器
    data_o = data_o_folds[fold]
    data_a = data_a_folds[fold]
    logger.info(f"=== Fold {fold + 1}/5 ===")
    
    # 为每一折创建新的模型和优化器
    model, optimizer = Create_model(args)
    
    # 获取当前折的数据加载器
    train_loader = train_loaders[fold]
    test_loader = test_loaders[fold]
    
    # 训练和测试当前折的模型
    fold_results = train_model(model, optimizer, data_o, data_a, train_loader, test_loader, args)
    all_fold_results.append(fold_results)
    
    logger.info(f"Fold {fold + 1} completed.")

# 计算所有折的平均结果
logger.info("=== 5-Fold Cross Validation Results ===")
if all_fold_results:
    aurocs = [result['auroc'] for result in all_fold_results]
    auprcs = [result['auprc'] for result in all_fold_results]
    f1s = [result['f1'] for result in all_fold_results]
    losses = [result['loss'] for result in all_fold_results]
    
    logger.info(f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    logger.info(f"AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    logger.info(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logger.info(f"Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    # 保存最终 5-fold 汇总指标到 result_summary_{run_id}.txt（与日志后缀一致）
    _paths = get_run_paths()
    _run_id = _paths.get("run_id") or ""
    _summary_lines = [
        "5-Fold Cross Validation Summary",
        f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}",
        f"AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}",
        f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
        f"Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}"
    ]
    _fname = f"result_summary_{_run_id}.txt" if _run_id else "result_summary.txt"
    save_result_text("\n".join(_summary_lines), filename=_fname)
    logger.info("Detailed Results:")
    _per_fold_lines = []
    for i, result in enumerate(all_fold_results):
        logger.info(f"Fold {i+1}: AUROC={result['auroc']:.4f}, AUPRC={result['auprc']:.4f}, F1={result['f1']:.4f}")
        _per_fold_lines.append(f"Fold {i+1}: AUROC={result['auroc']:.4f}, AUPRC={result['auprc']:.4f}, F1={result['f1']:.4f}")
    _pfname = f"per_fold_{_run_id}.txt" if _run_id else "per_fold.txt"
    if len(_per_fold_lines) > 0:
        save_result_text("\n".join(_per_fold_lines), filename=_pfname)
else:
    logger.info("No results collected.")
logger.info("All folds completed!")
# 记录运行结束并（在 Linux 且命令指定时）执行关机
finalize_run()
perform_shutdown_if_linux(args.shutdown)