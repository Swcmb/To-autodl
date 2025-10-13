import logging
import os
import sys
import time
import platform
import subprocess
from pathlib import Path
from typing import Optional, Union, Iterable

# 全局状态
_LOGGER: Optional[logging.Logger] = None
_RUN_ID: Optional[str] = None
_RUN_NAME: Optional[str] = None
_BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根 d:/Paper-code
_OUTPUT_DIR = _BASE_DIR / "OUTPUT"
_LOG_DIR = _OUTPUT_DIR / "log"
_RESULT_DIR = _OUTPUT_DIR / "result"
_RUN_RESULT_DIR: Optional[Path] = None

# 默认格式
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _ensure_dirs() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _compose_log_filename(run_name: Optional[str], run_id: str) -> str:
    # 日志命名：EM_{run_name}_YYYYMMDD_HHMMSS.log 或 EM_YYYYMMDD_HHMMSS.log
    prefix = "EM"
    if run_name:
        return f"{prefix}_{run_name}_{run_id}.log"
    return f"{prefix}_{run_id}.log"


def init_logging(run_name: Optional[str] = None,
                 level: Union[int, str] = logging.INFO,
                 to_console: bool = True) -> logging.Logger:
    """
    初始化集中日志（文件+控制台），在日志开头写入完整命令行。
    返回全局 logger，名称为 'EM'.
    """
    global _LOGGER, _RUN_ID, _RUN_NAME
    _ensure_dirs()

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # 防止重复添加 handler
    if _LOGGER is not None:
        return _LOGGER

    _RUN_ID = _timestamp()
    _RUN_NAME = run_name or None
    log_file_name = _compose_log_filename(_RUN_NAME, _RUN_ID)
    log_file_path = _LOG_DIR / log_file_name

    logger = logging.getLogger("EM")
    logger.setLevel(level)
    logger.propagate = False

    # 文件 Handler
    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台 Handler（始终输出到控制台）
    if to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # 在日志开头记录完整命令行与运行边界
    cmd_line = f"{sys.executable} " + " ".join(sys.argv)
    border = "=" * 80
    logger.info(border)
    logger.info(f"RUN START | run_id={_RUN_ID} | run_name={_RUN_NAME or '-'}")
    logger.info(f"COMMAND: {cmd_line}")
    logger.info(border)

    _LOGGER = logger
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取模块 logger；若未初始化则执行默认初始化。
    """
    if _LOGGER is None:
        init_logging()
    if name:
        return _LOGGER.getChild(name)
    return _LOGGER


class _StdToLogger:
    """
    将写入的文本仅写入 logger（不再直写原stdout/stderr），避免控制台重复行。
    """
    def __init__(self, logger: logging.Logger, level: int, original_stream):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream
        self._buffer = ""

    def write(self, msg):
        # 将完整行写入日志
        self._buffer += msg
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                # 避免日志格式破坏，逐行写入
                if self.level == logging.ERROR:
                    self.logger.error(line)
                else:
                    self.logger.info(line)

    def flush(self):
        pass


def redirect_print(enable: bool = True) -> None:
    """
    重定向 print 到 logger，同时保留控制台输出。
    """
    if not enable:
        return
    logger = get_logger("print")
    sys.stdout = _StdToLogger(logger, logging.INFO, sys.__stdout__)
    sys.stderr = _StdToLogger(logger, logging.ERROR, sys.__stderr__)


def make_result_run_dir(prefix: str = "data") -> Path:
    """
    创建运行结果子目录：OUTPUT/result/{prefix}_YYYYMMDD_HHMMSS 或含 run-name 前缀。
    """
    global _RUN_RESULT_DIR
    run_id = _RUN_ID or _timestamp()
    name_prefix = prefix
    if _RUN_NAME:
        name_prefix = f"{_RUN_NAME}_{prefix}"
    run_dir = _RESULT_DIR / f"{name_prefix}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _RUN_RESULT_DIR = run_dir
    # 在日志中记录结果目录
    get_logger("result").info(f"Result directory: {run_dir}")
    return run_dir


def set_global_run_dir(path: Union[str, Path]) -> None:
    """
    设置全局当前运行结果目录。
    """
    global _RUN_RESULT_DIR
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    _RUN_RESULT_DIR = p
    get_logger("result").info(f"Global run dir set: {p}")


def get_run_paths() -> dict:
    """
    获取当前运行的路径信息。
    """
    return {
        "output_dir": str(_OUTPUT_DIR),
        "log_dir": str(_LOG_DIR),
        "result_dir": str(_RESULT_DIR),
        "run_result_dir": str(_RUN_RESULT_DIR) if _RUN_RESULT_DIR else None,
        "run_id": _RUN_ID,
        "run_name": _RUN_NAME
    }


def _ensure_run_dir() -> Path:
    if _RUN_RESULT_DIR is None:
        return make_result_run_dir("data")
    return _RUN_RESULT_DIR


def save_result_text(content: Union[str, Iterable[str]],
                     filename: Optional[str] = None,
                     subdir: Optional[str] = None) -> Path:
    """
    保存结果为 .txt 到 OUTPUT/result 下。
    - content: 字符串或可迭代的行
    - filename: 自定义文件名（不含路径），默认 'result_YYYYMMDD_HHMMSS.txt'
    - subdir: 可选子目录名（会在当前运行目录下创建）
    返回写入的文件路径。
    """
    logger = get_logger("result.save")
    base_dir = _ensure_run_dir()
    target_dir = base_dir

    if subdir:
        target_dir = base_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

    fname = filename or f"result_{_timestamp()}.txt"
    if not fname.endswith(".txt"):
        fname += ".txt"

    fpath = target_dir / fname

    try:
        if isinstance(content, str):
            fpath.write_text(content, encoding="utf-8")
        else:
            with fpath.open("w", encoding="utf-8") as f:
                for line in content:
                    f.write(str(line))
                    if not str(line).endswith("\n"):
                        f.write("\n")
        logger.info(f"Saved result text: {fpath}")
    except Exception as e:
        logger.error(f"Failed to save result text to {fpath}: {e}")
        raise
    return fpath


def finalize_run() -> None:
    """
    记录运行结束边界。
    """
    logger = get_logger()
    border = "=" * 80
    paths = get_run_paths()
    logger.info(border)
    logger.info(f"RUN END   | run_id={_RUN_ID} | run_name={_RUN_NAME or '-'}")
    logger.info(f"RESULT DIR: {paths.get('run_result_dir')}")
    logger.info(border)


def perform_shutdown_if_linux(shutdown: bool) -> bool:
    """
    辅助执行 Linux 关机（安全保护：默认不执行）。
    - 若 shutdown 为 True 且平台为 Linux：
        1) 记录提示信息
        2) 仅当环境变量 ENABLE_SHUTDOWN=1 时，尝试执行 'sudo shutdown -h now'
    返回是否已触发执行。
    """
    logger = get_logger("shutdown")
    if not shutdown:
        return False
    if platform.system() != "Linux":
        logger.info("Shutdown requested but current OS is not Linux. Skipped.")
        return False

    logger.warning("Shutdown requested on Linux. Safety guard enabled.")
    if os.environ.get("ENABLE_SHUTDOWN") == "1":
        cmd = ["sudo", "shutdown", "-h", "now"]
        logger.warning(f"Executing: {' '.join(cmd)}")
        try:
            # 执行命令（可能需要sudo权限）
            subprocess.run(cmd, check=False)
            logger.warning("Shutdown command issued.")
            return True
        except Exception as e:
            logger.error(f"Shutdown execution failed: {e}")
            return False
    else:
        logger.warning("Shutdown NOT executed. Set environment ENABLE_SHUTDOWN=1 to allow.")
        logger.warning("Manual command: sudo shutdown -h now")
        return False