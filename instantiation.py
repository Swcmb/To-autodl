"""编码器实验.md"""
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

    # 仅优化需要训练的参数（显式排除 requires_grad=False，例如 MoCo 动量分支/缓冲）
    trainable_params = (p for p in model.parameters() if getattr(p, "requires_grad", True))
    optimizer = Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer