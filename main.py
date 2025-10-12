import torch  # 导入PyTorch深度学习框架
import numpy as np  # 导入NumPy库，用于进行科学计算，特别是数组操作
from parms_setting import settings  # 从本地的parms_setting.py文件导入settings函数，用于获取所有超参数
from data_preprocess import load_data  # 从本地的data_preprocess.py文件导入load_data函数，用于加载和预处理数据
from instantiation import Create_model  # 从本地的instantiation.py文件导入Create_model函数，用于创建模型和优化器
from train import train_model  # 从本地的train.py文件导入train_model函数，用于执行模型的训练和评估流程
from data_preprocess import get_fold_data  # 导入get_fold_data函数，用于获取指定折的数据
import os  # 导入os模块，用于与操作系统交互，如此处设置环境变量

# 设置程序使用的GPU设备
# "CUDA_VISIBLE_DEVICES"是一个环境变量，用于指定哪些GPU可以被CUDA应用程序看到
# "0"表示只使用系统中编号为0的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# parameters setting  # 注释：参数设置
args = settings()  # 调用settings()函数，获取一个包含所有实验参数（如学习率、隐藏层维度等）的对象

# 检查CUDA（GPU计算）是否可用，并根据args.no_cuda标志来决定是否使用GPU
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using CUDA: {args.cuda}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")
else:
    print("CUDA not available, using CPU")
np.random.seed(args.seed)  # 为NumPy设置随机种子，以确保随机数生成是可复现的
torch.manual_seed(args.seed)  # 为PyTorch在CPU上的操作设置随机种子，保证结果一致性
# 修复第31行的语法错误
if args.cuda:  # 如果确定使用CUDA
    torch.cuda.manual_seed(args.seed)  # 也为PyTorch在GPU上的操作设置随机种子


# load data  # 注释：加载数据
# 调用load_data函数，传入参数对象args
# 该函数会返回处理好的图数据对象（原始图和对抗图）以及所有折的训练和测试数据加载器
data_o, data_a, train_loaders, test_loaders = load_data(args)

# 存储每一折的结果
all_fold_results = []
print("Starting 5-fold cross validation...")

for fold in range(5):
    print(f"\n=== Fold {fold + 1}/5 ===")
    
    # 为每一折创建新的模型和优化器
    model, optimizer = Create_model(args)
    
    # 获取当前折的数据加载器
    train_loader = train_loaders[fold]
    test_loader = test_loaders[fold]
    
    # 训练和测试当前折的模型
    fold_results = train_model(model, optimizer, data_o, data_a, train_loader, test_loader, args)
    all_fold_results.append(fold_results)
    
    print(f"Fold {fold + 1} completed.")

# 计算所有折的平均结果
print("\n=== 5-Fold Cross Validation Results ===")
if all_fold_results:
    import numpy as np
    aurocs = [result['auroc'] for result in all_fold_results]
    auprcs = [result['auprc'] for result in all_fold_results]
    f1s = [result['f1'] for result in all_fold_results]
    losses = [result['loss'] for result in all_fold_results]
    
    print(f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print("\nDetailed Results:")
    for i, result in enumerate(all_fold_results):
        print(f"Fold {i+1}: AUROC={result['auroc']:.4f}, AUPRC={result['auprc']:.4f}, F1={result['f1']:.4f}")
else:
    print("No results collected.")
print("All folds completed!")