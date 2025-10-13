import argparse  # 导入argparse库，用于解析命令行参数

def settings():  # 定义一个名为settings的函数，用于设置和返回所有实验参数
    # 创建一个ArgumentParser对象，用于后续添加和解析参数
    parser = argparse.ArgumentParser()

    # public parameters  # 注释：公共参数
    # 添加'--seed'参数，用于设置随机种子，以保证实验的可复现性
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed. Default is 0.')

    # 添加'--no-cuda'标志，如果使用该标志，则禁用CUDA（即不使用GPU）
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    # 添加'--in_file'参数，指定阳性样本（已知的关联）文件路径
    parser.add_argument('--in_file', default="dataset1/LDA.edgelist",    # positive sample
                        help='Path to data. e.g., data/LDA.edgelist')
    # 添加'--file'参数作为'--in_file'的别名
    parser.add_argument('--file', dest='in_file', 
                        help='Alias for --in_file')

    # 添加'--neg_sample'参数，指定阴性样本（未知的关联）文件路径
    parser.add_argument('--neg_sample', default="dataset1/non_LDA.edgelist",     # negative sample
                        help='Path to data. e.g., data/LDA.edgelist')

    # 添加'--validation_type'参数，选择交叉验证的类型
    parser.add_argument('--validation_type', default="5_cv1", choices=['5_cv1', '5_cv2', '5-cv1', '5-cv2'],
                        help='Initial cross_validation type. Default is 5_cv1.')

    # 添加'--task_type'参数，选择要执行的预测任务类型（如lncRNA-disease, miRNA-disease等）
    parser.add_argument('--task_type', default="LDA", choices=['LDA', 'MDA','LMI'],
                        help='Initial prediction task type. Default is LDA.')

    # 添加'--feature_type'参数，选择节点初始特征的生成方式
    parser.add_argument('--feature_type', type=str, default='normal', choices=['one_hot', 'uniform', 'normal', 'position'],
                        help='Initial node feature type. Default is position.')

    # Training settings  # 注释：训练设置
    # 添加'--lr'参数，设置优化器的初始学习率
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate. Default is 5e-4.')
    # 添加'--learning_rate'参数作为'--lr'的别名
    parser.add_argument('--learning_rate', dest='lr', type=float,
                        help='Alias for --lr')

    # 添加'--dropout'参数，设置模型中dropout层的比率，用于防止过拟合
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate. Default is 0.5.')

    # 添加'--weight_decay'参数，设置权重衰减（L2正则化）的系数
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters) Default is 5e-4.')

    # 添加'--batch'参数，设置训练时每个批次的大小
    parser.add_argument('--batch', type=int, default=25,
                        help='Batch size. Default is 25.')

    # 添加'--epochs'参数，设置模型训练的总轮数
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train. Default is 50.')

    # 添加'--loss_ratio1'参数，设置第一个损失函数（任务损失）的权重
    parser.add_argument('--loss_ratio1', type=float, default=1,
                        help='Ratio of task1. Default is 1')

    # 添加'--loss_ratio2'参数，设置第二个损失函数（对比损失1）的权重
    parser.add_argument('--loss_ratio2', type=float, default=0.5,
                        help='Ratio of task2. Default is 0.5')

    # 添加'--loss_ratio3'参数，设置第三个损失函数（对比损失2）的权重
    parser.add_argument('--loss_ratio3', type=float, default=0.5,
                        help='Ratio of task3. Default is 0.5')

    # 添加'--loss_ratio4'参数，设置第四个损失函数（对抗损失）的权重
    parser.add_argument('--loss_ratio4', type=float, default=0.5,
                        help='Ratio of task4. Default is 0.5')

    # model parameter setting  # 注释：模型参数设置
    # 添加'--dimensions'参数，设置节点初始特征的维度
    parser.add_argument('--dimensions', type=int, default=256,
                        help='dimensions of feature d. Default is 256 (LDA, MDA tasks) 512 (LMI task).')
    # 添加'--embed_dim'参数作为'--dimensions'的别名
    parser.add_argument('--embed_dim', dest='dimensions', type=int,
                        help='Alias for --dimensions')

    # 添加'--hidden1'参数，设置编码器第一层的隐藏层维度
    parser.add_argument('--hidden1', type=int, default=128,
                        help='Embedding dimension of encoder layer 1 for CSGLMD. Default is d/2.')

    # 添加'--hidden2'参数，设置编码器第二层的隐藏层维度
    parser.add_argument('--hidden2', type=int, default=64,
                        help='Embedding dimension of encoder layer 2 for CSGLMD. Default is d/4.')

    # 添加'--decoder1'参数，设置解码器第一层的隐藏层维度
    parser.add_argument('--decoder1', type=int, default=512,
                        help='NEmbedding dimension of decoder layer 1 for CSGLMD. Default is 512.')

    # 编码器选择与图注意力参数
    parser.add_argument('--encoder_type', type=str, default='csglmd',
                        choices=['csglmd','csglmd_main','mgacmda','gat','gt','gat_gt_serial','gat_gt_parallel'],
                        help='Encoder type to use.')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='Number of attention heads for GAT-based encoders.')
    parser.add_argument('--gt_heads', type=int, default=4,
                        help='Number of attention heads for Graph Transformer encoders.')

    # 融合策略参数
    parser.add_argument('--fusion_type', type=str, default='basic',
                        choices=['basic','dot','additive','self_attn','gat_fusion','gt_fusion'],
                        help='Fusion strategy for pairwise interaction.')
    parser.add_argument('--fusion_heads', type=int, default=4,
                        help='Number of attention heads for self-attention/GAT/GT fusion (pairwise).')

    # CPU 并行与数据加载参数（与 To-autodl 对齐）
    parser.add_argument('--threads', type=int, default=32,
                        help='Backend threads cap. Use -1 for auto detect (capped at 32).')
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='DataLoader workers. -1 means auto detect with cap=32 (defaults to min(8, threads)).')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='DataLoader prefetch factor (only valid when num_workers>0).')
    parser.add_argument('--chunk_size', type=int, default=0,
                        help='Generic chunk size for CPU tasks (0 means auto, default 20000).')

    # 添加新参数以支持您的命令行需求
    # 添加相似度阈值参数
    parser.add_argument('--similarity_threshold', type=float, default=0.5,
                        help='Similarity threshold for graph construction. Default is 0.5.')
    
    # 添加多任务学习的权重参数别名
    parser.add_argument('--alpha', dest='loss_ratio2', type=float, default=0.5,
                        help='Weight for contrastive learning task. Alias for --loss_ratio2.')
    parser.add_argument('--beta', dest='loss_ratio3', type=float, default=0.5,
                        help='Weight for second contrastive learning task. Alias for --loss_ratio3.')
    parser.add_argument('--gamma', dest='loss_ratio4', type=float, default=0.5,
                        help='Weight for adversarial learning task. Alias for --loss_ratio4.')

    # 保存与折数相关参数（主程序接入）
    parser.add_argument('--save_datasets', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Whether to save constructed datasets. Use true/false. Default true.')
    parser.add_argument('--save_format', type=str, default='npy', choices=['npy', 'txt'],
                        help='Save format for datasets. Default npy.')
    parser.add_argument('--save_dir_prefix', type=str, default='result/data',
                        help='Save directory prefix relative to EM/. Default result/data')

    # 运行名称与关机控制（供主程序日志展示与收尾动作）
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name to show in logs and result folder prefix.')
    parser.add_argument('--shutdown', action='store_true',
                        help='Linux only: shutdown after run')


    # 解析所有添加的参数，并将它们存储在一个命名空间对象中
    args = parser.parse_args()
    
    # 处理验证类型的格式统一
    if args.validation_type == '5-cv1':
        args.validation_type = '5_cv1'
    elif args.validation_type == '5-cv2':
        args.validation_type = '5_cv2'

    # 返回包含所有参数设置的对象
    return args