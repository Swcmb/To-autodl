from torch_geometric.data import Data  # 从PyTorch Geometric库中导入Data类，用于封装图数据
from torch.utils.data import Dataset, DataLoader  # 从PyTorch中导入Dataset和DataLoader，用于创建和加载数据集
from utils import *  # 从本地的utils.py文件中导入所有函数
import numpy as np  # 导入numpy库，用于高效的数值计算
import torch  # 导入PyTorch库，用于深度学习
import scipy.sparse as sp  # 导入scipy的稀疏矩阵模块，用于处理稀疏数据
import os  # 基于文件目录解析数据路径

# 统一路径解析：若为相对路径，则相对 EM 目录解析
BASE_DIR = os.path.dirname(__file__)
def _p(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

"This code uses five-fold cross-validation"  # 字符串注释：此代码使用五折交叉验证

class Data_class(Dataset):  # 定义一个自定义的数据集类，继承自PyTorch的Dataset类

    def __init__(self, triple):  # 类的初始化方法，接收一个三元组数组（实体1, 实体2, 标签）
        self.entity1 = triple[:, 0]  # 提取所有样本的第一个实体
        self.entity2 = triple[:, 1]  # 提取所有样本的第二个实体
        self.label = triple[:, 2]  # 提取所有样本的标签（0或1）

    def __len__(self):  # 定义获取数据集大小的方法
        return len(self.label)  # 数据集的大小即为标签的数量

    def __getitem__(self, index):  # 定义通过索引获取单个样本的方法

        return self.label[index], (self.entity1[index], self.entity2[index])  # 返回指定索引的标签和实体对


def get_fold_data(data_o, data_a, train_loaders, test_loaders, fold_index):
    """获取指定折的数据加载器"""
    if fold_index >= len(train_loaders) or fold_index < 0:
        raise ValueError(f"Fold index {fold_index} is out of range. Available folds: 0-{len(train_loaders)-1}")
    
    return data_o, data_a, train_loaders[fold_index], test_loaders[fold_index]


def load_data(args, k_fold=5):  # 定义加载数据的主函数，接收命令行参数和折数
    """Read data from path, convert data into k-fold cross validation loaders, return features and adjacency"""  # 函数文档字符串：从路径读取数据，转换为k折交叉验证加载器，返回特征和邻接矩阵
    # read data  # 注释：读取数据
    print('Loading {0} seed{1} dataset...'.format(args.in_file, args.seed))  # 打印正在加载的数据集信息和随机种子
    positive = np.loadtxt(_p(args.in_file), dtype=np.int64)  # 从指定文件加载阳性样本（已知的关联）

    # postive sample  # 注释：阳性样本处理
    link_size = int(positive.shape[0])  # 获取阳性样本的总数
    np.random.seed(args.seed)  # 设置随机种子以保证结果可复现
    np.random.shuffle(positive)  # 随机打乱阳性样本的顺序
    positive = positive[:link_size]  # 确保使用所有阳性样本（这一步实际上没有改变样本数量）

    # negative sample  # 注释：阴性样本处理
    negative_all = np.loadtxt(_p(args.neg_sample), dtype=np.int64)  # 从指定文件加载所有可能的阴性样本
    np.random.shuffle(negative_all)  # 随机打乱所有阴性样本
    negative = np.asarray(negative_all[:positive.shape[0]])  # 选取与阳性样本数量相等的阴性样本

    # 计算每折的大小
    fold_size = positive.shape[0] // k_fold  # 计算每折的样本数量
    
    positive = np.concatenate([positive, np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)], axis=1)  # 给阳性样本添加标签1
    negative = np.concatenate([negative, np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)], axis=1)  # 给选出的阴性样本添加标签0
    negative_all = np.concatenate([negative_all, np.zeros(negative_all.shape[0], dtype=np.int64).reshape(negative_all.shape[0], 1)], axis=1) # 给所有阴性样本添加标签0

    print('Selected cross_validation type: 5_cv1')  # 打印选择了五折交叉验证类型
    
    # 五折交叉验证数据划分
    train_loaders = []
    test_loaders = []
    train_data_folds = []
    test_data_folds = []
    
    for fold in range(k_fold):
        print(f'Preparing fold {fold + 1}/{k_fold}...')
        
        # 计算当前折的测试集索引范围
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < k_fold - 1 else positive.shape[0]
        
        # 划分阳性样本
        test_positive = positive[start_idx:end_idx]
        train_positive = np.vstack((positive[:start_idx], positive[end_idx:]))
        
        # 划分阴性样本
        test_negative = negative[start_idx:end_idx]
        train_negative = np.vstack((negative[:start_idx], negative[end_idx:]))
        
        # 构建训练集和测试集
        train_data = np.vstack((train_positive, train_negative))
        test_data = np.vstack((test_positive, test_negative))
        
        train_data_folds.append(train_data)
        test_data_folds.append(test_data)
        
        print(f"Fold {fold + 1} - Train: {train_data.shape}, Test: {test_data.shape}")
    
    # 使用第一折的训练数据构建图（可以根据需要修改）
    train_data = train_data_folds[0]
    test_data = test_data_folds[0]


    # construct adjacency  # 注释：构建邻接矩阵
    train_positive = train_data[train_data[:, 2] == 1]  # 提取用于构建图的训练集中的阳性样本
    # print("train_positive: ",train_positive)  # 被注释掉的调试语句
    print('Selected task type...')  # 打印选择了任务类型
    "Note: node similarity need to recomputed, (1)your can save train/test id, " \
    "(2) according to calculating_similarity.py. "  # 字符串注释：提醒节点相似度需要根据训练/测试划分重新计算

    if args.task_type == 'LDA':  # 如果任务类型是 'LDA' (lncRNA-Disease Association)
        l_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                            shape=(240, 405), dtype=np.float32)  # 根据训练集中的阳性关联构建lncRNA-disease稀疏关联矩阵
        l_d = l_d.toarray()  # 将稀疏矩阵转换为密集矩阵
        m_d = np.loadtxt(_p("dataset1/mi_dis.txt"))   # 加载miRNA-disease关联矩阵
        m_l = np.loadtxt(_p("dataset1/lnc_mi.txt")).T  # 加载并转置lncRNA-miRNA关联矩阵，得到miRNA-lncRNA矩阵
        l_sim = np.loadtxt(_p("dataset1/one_hot_lnc_sim.txt")) # 加载lncRNA的相似度矩阵
        d_sim = np.loadtxt(_p("dataset1/one_hot_dis_sim.txt"))  # 加载disease的相似度矩阵
        m_sim = np.loadtxt(_p("dataset1/one_hot_mi_sim.txt"))   # 加载miRNA的相似度矩阵

    if args.task_type == 'MDA':  # 如果任务类型是 'MDA' (miRNA-Disease Association)
        m_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                            shape=(495, 405), dtype=np.float32)  # 根据训练集中的阳性关联构建miRNA-disease稀疏关联矩阵
        m_d = m_d.toarray()  # 将稀疏矩阵转换为密集矩阵
        l_d = np.loadtxt(_p("dataset1/lnc_dis.txt"))  # 加载lncRNA-disease关联矩阵
        m_l = np.loadtxt(_p("dataset1/lnc_mi.txt")).T  # 加载并转置lncRNA-miRNA关联矩阵

        l_sim = np.loadtxt(_p("dataset1/lnc_fuse_sim_0.8.txt"))  # 加载融合后的lncRNA相似度矩阵
        d_sim = np.loadtxt(_p("dataset1/dis_fuse_sim_0.8.txt"))  # 加载融合后的disease相似度矩阵
        m_sim = np.loadtxt(_p("dataset1/mi_fuse_sim_0.8.txt"))  # 加载融合后的miRNA相似度矩阵

    if args.task_type == 'LMI':  # 如果任务类型是 'LMI' (lncRNA-miRNA Interaction)
        l_m = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                            shape=(240, 495), dtype=np.float32)  # 根据训练集中的阳性关联构建lncRNA-miRNA稀疏关联矩阵
        m_l = l_m.toarray().T  # 将其转换为密集矩阵并转置，得到miRNA-lncRNA矩阵
        l_d = np.loadtxt(_p("dataset1/lnc_dis.txt"))  # 加载lncRNA-disease关联矩阵
        m_d = np.loadtxt(_p("dataset1/mi_dis.txt"))   # 加载miRNA-disease关联矩阵
        l_sim = np.loadtxt(_p("dataset1/one_hot_lnc_sim.txt")) # 加载lncRNA的相似度矩阵
        d_sim = np.loadtxt(_p("dataset1/one_hot_dis_sim.txt"))  # 加载disease的相似度矩阵
        m_sim = np.loadtxt(_p("dataset1/one_hot_mi_sim.txt"))  # 加载miRNA的相似度矩阵
    # print(l_d.shape, m_d.shape, l_m.shape, l_sim.shape, d_sim.shape, m_sim.shape) # 被注释掉的调试语句

    adj = construct_graph(l_d, m_d, m_l, l_sim, m_sim, d_sim)  # 调用utils中的函数，将所有矩阵拼接成一个大的异构图邻接矩阵
    adj = lalacians_norm(adj)  # 对邻接矩阵进行拉普拉斯归一化 (注: 原文可能有拼写错误，应为laplacians_norm)

    # construct edges  # 注释：构建边索引
    edges_o = adj.nonzero()  # 获取归一化后邻接矩阵中非零元素的索引
    edge_index_o = torch.tensor(np.vstack((edges_o[0], edges_o[1])), dtype=torch.long)  # 将索引转换为torch_geometric所需的边索引格式

    # build data loaders for all folds  # 注释：为所有折构建数据加载器
    params = {'batch_size': args.batch, 'shuffle': True,  'drop_last': True}  # 设置DataLoader的参数，如批量大小、是否打乱等
    
    for fold in range(k_fold):
        training_set = Data_class(train_data_folds[fold])  # 使用前面定义的Data_class类实例化训练数据集
        train_loader = DataLoader(training_set, **params)  # 创建训练数据的DataLoader
        train_loaders.append(train_loader)
        
        test_set = Data_class(test_data_folds[fold])  # 实例化测试数据集
        test_loader = DataLoader(test_set, **params)  # 创建测试数据的DataLoader
        test_loaders.append(test_loader)

    # extract features  # 注释：提取节点特征
    print('Extracting features...')  # 打印正在提取特征
    if args.feature_type == 'one_hot':  # 如果特征类型是 'one_hot'
        features = np.eye(adj.shape[0])  # 创建一个单位矩阵作为节点的独热编码特征

    elif args.feature_type == 'uniform':  # 如果特征类型是 'uniform'
        np.random.seed(args.seed)  # 设置随机种子
        features = np.random.uniform(low=0, high=1, size=(adj.shape[0], args.dimensions))  # 生成均匀分布的随机特征

    elif args.feature_type == 'normal':  # 如果特征类型是 'normal'
        np.random.seed(args.seed)  # 设置随机种子
        features = np.random.normal(loc=0, scale=1, size=(adj.shape[0], args.dimensions))  # 生成正态分布的随机特征

    elif args.feature_type == 'position':  # 如果特征类型是 'position'
        features = sp.coo_matrix(adj).todense()  # 使用归一化邻接矩阵的行作为节点的位置特征

    features_o = normalize(features)  # 对生成的原始特征进行归一化
    args.dimensions = features_o.shape[1]  # 更新参数中的特征维度

    # adversarial nodes  # 注释：生成对抗性节点特征
    np.random.seed(args.seed)  # 设置随机种子
    id = np.arange(features_o.shape[0])  # 生成节点ID序列
    id = np.random.permutation(id)  # 随机打乱节点ID
    features_a = features_o[id]  # 通过打乱的ID重新索引特征矩阵，生成“伪造”的对抗特征

    y_a = torch.cat((torch.ones(adj.shape[0], 1), torch.zeros(adj.shape[0], 1)), dim=1)  # 为对抗性任务创建标签（这里似乎不正确，通常对抗任务标签不同）

    x_o = torch.tensor(features_o, dtype=torch.float)  # 将原始节点特征转换为torch张量
    data_o = Data(x = x_o, edge_index=edge_index_o)  # 创建原始图的torch_geometric Data对象
    # print(data_o)  # 被注释掉的调试语句

    x_a = torch.tensor(features_a, dtype=torch.float)  # 将对抗节点特征转换为torch张量
    data_a = Data(x=x_a, y=y_a)  # 创建对抗图的Data对象
    # print(data_a)  # 被注释掉的调试语句

    print('Loading finished!')  # 打印数据加载完成
    return data_o, data_a, train_loaders, test_loaders  # 返回原始图数据、对抗图数据、所有折的训练加载器和测试加载器