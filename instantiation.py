"""编码器实验.md"""
import torch
import os
from torch.optim import Adam
from encoders import EncoderRegistry
from layer import CSGLMD  # 保留原始 CSGLMD 以兼容默认

def Create_model(args):
    encoder_type = getattr(args, 'encoder_type', 'csglmd')  # 默认使用本项目 CSGLMD
    if encoder_type == 'csglmd':
        model = CSGLMD(feature=args.dimensions, hidden1=args.hidden1, hidden2=args.hidden2, decoder1=args.decoder1, dropout=args.dropout)
    else:
        EncoderClass = EncoderRegistry.get(encoder_type)
        if EncoderClass is None:
            raise ValueError(f'Unknown encoder_type: {encoder_type}')
        model = EncoderClass(args)

    # 可选：开启 torch.compile 以融合算子（需 PyTorch >= 2.0）
    if getattr(args, 'compile', False):
        # 读取可选配置（若CLI未提供则采用安全默认）
        compile_mode = getattr(args, 'compile_mode', 'reduce-overhead')  # default/reduce-overhead/max-autotune
        gemm_backend = getattr(args, 'gemm_backend', 'auto')  # auto/cublas/triton
        max_autotune = int(getattr(args, 'max_autotune', 0))  # 0/1

        # 按需限制 GEMM autotune 后端与规模（仅在 compile 路径生效）
        if gemm_backend in ('cublas', 'triton'):
            os.environ['TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS'] = gemm_backend
        if max_autotune in (0, 1):
            os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = str(max_autotune)

        try:
            model = torch.compile(model, backend="inductor", mode=compile_mode)
        except Exception as _e:
            # 不影响主流程，保持可回退
            print(f"[compile] torch.compile failed: {_e}. Fallback to eager.")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer