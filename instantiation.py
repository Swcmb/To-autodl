"""编码器实验.md"""
import torch
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
        try:
            # 使用较稳健的模式；若你更追求极致可改为 "max-autotune"
            model = torch.compile(model, mode="max-autotune")
        except Exception as _e:
            # 不影响主流程，保持可回退
            print(f"[compile] torch.compile failed: {_e}. Fallback to eager.")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer