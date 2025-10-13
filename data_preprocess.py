from torch_geometric.data import Data  # 从PyTorch Geometric库中导入Data类，用于封装图数据
from torch.utils.data import Dataset, DataLoader  # 从PyTorch中导入Dataset和DataLoader，用于创建和加载数据集
from utils import *  # 从本地的utils.py文件中导入所有函数
import numpy as np  # 导入numpy库，用于高效的数值计算
import torch  # 导入PyTorch库，用于深度学习
import scipy.sparse as sp  # 导入scipy的稀疏矩阵模块，用于处理稀疏数据
import os  # 基于文件目录解析数据路径
import time
from datetime import datetime
from label_annotation import load_positive, load_negative_all, sample_negative, attach_labels, save_dataset
from calculating_similarity import calculate_GaussianKernel_sim, getRNA_functional_sim, RNA_fusion_sim, dis_fusion_sim
from log_output_manager import get_logger

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
    _logger = get_logger()
    _logger.info('Loading {0} seed{1} dataset...'.format(args.in_file, args.seed))
    _logger.info(f"Selected cross_validation type: {args.validation_type}")
    _logger.info(f"Selected task_type: {args.task_type}")
    _logger.info(f"Selected feature_type: {args.feature_type}")
    _logger.info(f"Selected embed_dim: {getattr(args, 'embed_dim', 'N/A')}")
    _logger.info(f"Selected learning_rate: {getattr(args, 'learning_rate', 'N/A')}")
    _logger.info(f"Selected epochs: {getattr(args, 'epochs', 'N/A')}")
    t_global_start = time.perf_counter()

    # 根据 validation_type 实现两种折分割策略
    # 加载正样本与负样本全集，并附加标签
    positive = load_positive(args.in_file, args.seed)  # shape=(P,2)
    negative_all = load_negative_all(args.neg_sample, args.seed)  # shape=(N,2)

    # 统一正样本添加标签
    pos_lbl = np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)
    positive_labeled = np.concatenate([positive, pos_lbl], axis=1)

    train_data_folds = []
    test_data_folds = []
    train_loaders = []
    test_loaders = []

    if args.validation_type == '5_cv2':
        # 5-cv2:
        # - 正样本分5折；训练用4折正样本 + 等量随机负样本；测试用1折正样本 + 全部负样本
        fold_size = positive.shape[0] // 5

        # 全负样本添加标签（测试集使用全部负样本）
        neg_all_lbl = np.zeros(negative_all.shape[0], dtype=np.int64).reshape(negative_all.shape[0], 1)
        negative_all_labeled = np.concatenate([negative_all, neg_all_lbl], axis=1)

        for fold in range(5):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < 4 else positive.shape[0]

            test_positive = positive_labeled[start_idx:end_idx]
            train_positive = np.vstack((positive_labeled[:start_idx], positive_labeled[end_idx:]))

            # 训练负样本：等量随机选取
            # 为保证不同折的随机性，这里使用每折不同的种子派生
            np.random.seed(args.seed + fold)
            neg_shuffled = negative_all.copy()
            np.random.shuffle(neg_shuffled)
            train_neg_sampled = np.asarray(neg_shuffled[:train_positive.shape[0]])
            train_neg_lbl = np.zeros(train_neg_sampled.shape[0], dtype=np.int64).reshape(train_neg_sampled.shape[0], 1)
            train_negative = np.concatenate([train_neg_sampled, train_neg_lbl], axis=1)

            # 测试负样本：全部负样本
            test_negative = negative_all_labeled

            # 构建训练集与测试集
            train_data = np.vstack((train_positive, train_negative))
            test_data = np.vstack((test_positive, test_negative))

            train_data_folds.append(train_data)
            test_data_folds.append(test_data)

        total_data = np.vstack((positive_labeled, negative_all_labeled))
        # 选取第一折数据用于图构建
        train_data = train_data_folds[0]
        test_data = test_data_folds[0]

    else:
        # 默认 5_cv1：
        # - 采样与正样本等量的负样本；正负样本按相同索引区间切分；每折内拼接为 train/test
        neg_sampled = sample_negative(negative_all, positive.shape[0])
        neg_lbl = np.zeros(neg_sampled.shape[0], dtype=np.int64).reshape(neg_sampled.shape[0], 1)
        negative_labeled = np.concatenate([neg_sampled, neg_lbl], axis=1)

        fold_size = positive.shape[0] // 5

        for fold in range(5):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < 4 else positive.shape[0]

            # 划分阳性
            test_positive = positive_labeled[start_idx:end_idx]
            train_positive = np.vstack((positive_labeled[:start_idx], positive_labeled[end_idx:]))

            # 划分阴性（与正样本使用相同索引区间）
            test_negative = negative_labeled[start_idx:end_idx]
            train_negative = np.vstack((negative_labeled[:start_idx], negative_labeled[end_idx:]))

            # 构建训练集与测试集
            train_data = np.vstack((train_positive, train_negative))
            test_data = np.vstack((test_positive, test_negative))

            train_data_folds.append(train_data)
            test_data_folds.append(test_data)

        total_data = np.vstack((positive_labeled, negative_labeled))
        # 选取第一折数据用于图构建
        train_data = train_data_folds[0]
        test_data = test_data_folds[0]

    # 保存（可选）
    if getattr(args, 'save_datasets', True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_dir = os.path.join(BASE_DIR, args.save_dir_prefix)
        out_dir = f"{prefix_dir}_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)
        save_dataset(total_data, os.path.join(out_dir, f"total_data.{args.save_format}"), fmt=args.save_format)
        for idx, (train_data, test_data) in enumerate(zip(train_data_folds, test_data_folds), start=1):
            save_dataset(train_data, os.path.join(out_dir, f"train_fold_{idx}.{args.save_format}"), fmt=args.save_format)
            save_dataset(test_data, os.path.join(out_dir, f"test_fold_{idx}.{args.save_format}"), fmt=args.save_format)
        _logger.info(f"Saved datasets to: {out_dir}")


    _logger.info('Selected task type...')
    # 每折输出容器
    data_o_folds = []
    data_a_folds = []

    # 疾病语义相似度（固定来源文件）
    dis_sem_sim = np.loadtxt(_p("dataset1/dis_sem_sim.txt"))

    def mask_pairs(mat, pairs):
        # 将测试集关联位置置 0（临时掩码）
        for i, j in pairs:
            if 0 <= i < mat.shape[0] and 0 <= j < mat.shape[1]:
                mat[i, j] = 0

    for fold in range(5):
        t_fold_start = time.perf_counter()
        train_data = train_data_folds[fold]
        test_data = test_data_folds[fold]
        train_positive = train_data[train_data[:, 2] == 1]
        test_positive = test_data[test_data[:, 2] == 1]

        # 基于训练集构建 inter-layer，并对测试位置掩码
        if args.task_type == 'LDA':
            # lncRNA-disease
            l_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                                shape=(240, 405), dtype=np.float32).toarray()
            mask_pairs(l_d, test_positive[:, :2].astype(int))

            # 其他关联来源原始数据
            m_d = np.loadtxt(_p("dataset1/mi_dis.txt"))
            m_l = np.loadtxt(_p("dataset1/lnc_mi.txt")).T

            # 训练集重算融合相似性
            lnc_gau_1 = calculate_GaussianKernel_sim(l_d)
            lnc_gau_2 = calculate_GaussianKernel_sim(m_l.T)
            lnc_fun = getRNA_functional_sim(RNAlen=l_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=l_d.copy())
            l_sim = RNA_fusion_sim(lnc_gau_1, lnc_gau_2, lnc_fun)

            mi_gau_1 = calculate_GaussianKernel_sim(m_d)
            mi_gau_2 = calculate_GaussianKernel_sim(m_l)
            mi_fun = getRNA_functional_sim(RNAlen=m_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=m_d.copy())
            m_sim = RNA_fusion_sim(mi_gau_1, mi_gau_2, mi_fun)

            dis_gau_1 = calculate_GaussianKernel_sim(l_d.T)
            dis_gau_2 = calculate_GaussianKernel_sim(m_d.T)
            d_sim = dis_fusion_sim(dis_gau_1, dis_gau_2, dis_sem_sim)

        elif args.task_type == 'MDA':
            # miRNA-disease
            m_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                                shape=(495, 405), dtype=np.float32).toarray()
            mask_pairs(m_d, test_positive[:, :2].astype(int))

            l_d = np.loadtxt(_p("dataset1/lnc_dis.txt"))
            m_l = np.loadtxt(_p("dataset1/lnc_mi.txt")).T

            lnc_gau_1 = calculate_GaussianKernel_sim(l_d)
            lnc_gau_2 = calculate_GaussianKernel_sim(m_l.T)
            lnc_fun = getRNA_functional_sim(RNAlen=l_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=l_d.copy())
            l_sim = RNA_fusion_sim(lnc_gau_1, lnc_gau_2, lnc_fun)

            mi_gau_1 = calculate_GaussianKernel_sim(m_d)
            mi_gau_2 = calculate_GaussianKernel_sim(m_l)
            mi_fun = getRNA_functional_sim(RNAlen=m_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=m_d.copy())
            m_sim = RNA_fusion_sim(mi_gau_1, mi_gau_2, mi_fun)

            dis_gau_1 = calculate_GaussianKernel_sim(l_d.T)
            dis_gau_2 = calculate_GaussianKernel_sim(m_d.T)
            d_sim = dis_fusion_sim(dis_gau_1, dis_gau_2, dis_sem_sim)

        elif args.task_type == 'LMI':
            # lncRNA-miRNA
            l_m = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                                shape=(240, 495), dtype=np.float32).toarray()
            # miRNA-lncRNA
            m_l = l_m.T
            # 掩码时索引反转
            mask_pairs(m_l, np.ascontiguousarray(test_positive[:, :2][:, ::-1]).astype(int))

            l_d = np.loadtxt(_p("dataset1/lnc_dis.txt"))
            m_d = np.loadtxt(_p("dataset1/mi_dis.txt"))

            lnc_gau_1 = calculate_GaussianKernel_sim(l_d)
            lnc_gau_2 = calculate_GaussianKernel_sim(m_l.T)
            lnc_fun = getRNA_functional_sim(RNAlen=l_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=l_d.copy())
            l_sim = RNA_fusion_sim(lnc_gau_1, lnc_gau_2, lnc_fun)

            mi_gau_1 = calculate_GaussianKernel_sim(m_d)
            mi_gau_2 = calculate_GaussianKernel_sim(m_l)
            mi_fun = getRNA_functional_sim(RNAlen=m_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=m_d.copy())
            m_sim = RNA_fusion_sim(mi_gau_1, mi_gau_2, mi_fun)

            dis_gau_1 = calculate_GaussianKernel_sim(l_d.T)
            dis_gau_2 = calculate_GaussianKernel_sim(m_d.T)
            d_sim = dis_fusion_sim(dis_gau_1, dis_gau_2, dis_sem_sim)

        else:
            raise ValueError(f"Unknown task_type: {args.task_type}")

        # 记录相似度阶段耗时，并开始图构建计时
        t_sim_end = time.perf_counter()
        _logger.info(f"[TIMING] Fold {fold + 1} similarity stage: {(t_sim_end - t_fold_start):.3f}s")
        t_graph_start = time.perf_counter()

        # 构建邻接并归一化
        adj = construct_graph(l_d, m_d, m_l, l_sim, m_sim, d_sim)
        adj = lalacians_norm(adj)
        t_graph_end = time.perf_counter()

        # 边索引
        edges_o = adj.nonzero()
        edge_index_o = torch.tensor(np.vstack((edges_o[0], edges_o[1])), dtype=torch.long)

        # 特征
        if args.feature_type == 'one_hot':
            features = np.eye(adj.shape[0])
        elif args.feature_type == 'uniform':
            np.random.seed(args.seed)
            features = np.random.uniform(low=0, high=1, size=(adj.shape[0], args.dimensions))
        elif args.feature_type == 'normal':
            np.random.seed(args.seed)
            features = np.random.normal(loc=0, scale=1, size=(adj.shape[0], args.dimensions))
        elif args.feature_type == 'position':
            features = sp.coo_matrix(adj).todense()
        else:
            features = np.eye(adj.shape[0])

        features_o = normalize(features)
        if fold == 0:
            args.dimensions = features_o.shape[1]

        # 对抗特征
        np.random.seed(args.seed)
        id_perm = np.random.permutation(np.arange(features_o.shape[0]))
        features_a = features_o[id_perm]

        y_a = torch.cat((torch.ones(adj.shape[0], 1), torch.zeros(adj.shape[0], 1)), dim=1)

        x_o = torch.tensor(features_o, dtype=torch.float)
        data_o = Data(x=x_o, edge_index=edge_index_o)

        x_a = torch.tensor(features_a, dtype=torch.float)
        data_a = Data(x=x_a, y=y_a)

        data_o_folds.append(data_o)
        data_a_folds.append(data_a)

        # 特征阶段与整折耗时
        t_feat_end = time.perf_counter()
        _logger.info(f"[TIMING] Fold {fold + 1} graph: {(t_graph_end - t_graph_start):.3f}s, features: {(t_feat_end - t_graph_end):.3f}s, fold total: {(t_feat_end - t_fold_start):.3f}s")

    # 为所有折构建 DataLoader（优化CPU侧并行）
    t_loader_start = time.perf_counter()
    num_workers = int(getattr(args, "num_workers", 0) or 0)
    prefetch_factor = int(getattr(args, "prefetch_factor", 4) or 4)
    base_params = {'batch_size': args.batch, 'shuffle': True, 'drop_last': True}
    if num_workers > 0:
        base_params.update({
            'num_workers': num_workers,
            'persistent_workers': True,
            'pin_memory': False
        })
        # prefetch_factor 仅在 num_workers>0 时有效
        if prefetch_factor and prefetch_factor > 0:
            base_params['prefetch_factor'] = prefetch_factor
    train_loaders = []
    test_loaders = []
    for fold in range(5):
        training_set = Data_class(train_data_folds[fold])
        train_loaders.append(DataLoader(training_set, **base_params))
        test_set = Data_class(test_data_folds[fold])
        test_loaders.append(DataLoader(test_set, **base_params))
    t_loader_end = time.perf_counter()
    _logger.info(f"[TIMING] DataLoader build: {(t_loader_end - t_loader_start):.3f}s")
    _logger.info(f"[TIMING] Preprocess total: {(t_loader_end - t_global_start):.3f}s")

    _logger.info('Loading finished!')
    return data_o_folds, data_a_folds, train_loaders, test_loaders