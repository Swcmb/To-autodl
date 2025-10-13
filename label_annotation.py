import os
import numpy as np

# 统一路径解析：与 EM/data_preprocess.py 保持一致（相对路径相对 EM 目录解析）
BASE_DIR = os.path.dirname(__file__)
def _p(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def load_positive(in_file: str, seed: int):
    """
    读取并打乱正样本（已知关联），返回数组 shape=(N, 2)
    与 CSGLMD-main/data_preprocess.py 保持一致：保留全部样本，使用随机种子打乱
    """
    positive = np.loadtxt(_p(in_file), dtype=np.int64)
    link_size = int(positive.shape[0])  # 保留全部
    np.random.seed(seed)
    np.random.shuffle(positive)
    positive = positive[:link_size]
    return positive


def load_negative_all(neg_file: str, seed: int):
    """
    读取并打乱负样本全集（未知关联），返回数组 shape=(M, 2)
    """
    negative_all = np.loadtxt(_p(neg_file), dtype=np.int64)
    np.random.seed(seed)
    np.random.shuffle(negative_all)
    return negative_all


def sample_negative(negative_all: np.ndarray, pos_count: int):
    """
    采样与正样本等量的负样本（与参考实现完全一致）
    """
    if negative_all.shape[0] < pos_count:
        raise ValueError(f"负样本全集数量不足：需要 {pos_count}，实际 {negative_all.shape[0]}")
    negative = np.asarray(negative_all[:pos_count])
    return negative


def attach_labels(positive: np.ndarray, negative: np.ndarray, negative_all: np.ndarray):
    """
    为正/负样本分别附加标签列，输出：
    - positive_labeled: [i, j, 1]
    - negative_labeled: [i, j, 0]（采样得到，用于训练/测试）
    - negative_all_labeled: [i, j, 0]（全集，供需要时参考）
    """
    pos_lbl = np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)
    neg_lbl = np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)
    neg_all_lbl = np.zeros(negative_all.shape[0], dtype=np.int64).reshape(negative_all.shape[0], 1)

    positive_labeled = np.concatenate([positive, pos_lbl], axis=1)
    negative_labeled = np.concatenate([negative, neg_lbl], axis=1)
    negative_all_labeled = np.concatenate([negative_all, neg_all_lbl], axis=1)

    return positive_labeled, negative_labeled, negative_all_labeled


def kfold_split_triples(positive_labeled: np.ndarray,
                        negative_labeled: np.ndarray,
                        k_fold: int = 5):
    """
    五折交叉划分，与参考实现一致：
    - 按正样本数量均分折；每折取对应区间为测试，其余为训练
    - 负样本按相同索引区间进行划分
    返回 train_data_folds, test_data_folds（列表，每项为 (num_samples, 3)）
    """
    if k_fold <= 0:
        raise ValueError("k_fold 必须为正整数")
    if positive_labeled.shape[0] != negative_labeled.shape[0]:
        raise ValueError("正负样本数量必须一致以进行等量划分")

    pos_num = positive_labeled.shape[0]
    fold_size = pos_num // k_fold
    train_data_folds = []
    test_data_folds = []

    for fold in range(k_fold):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < k_fold - 1 else pos_num

        # 划分阳性样本
        test_positive = positive_labeled[start_idx:end_idx]
        train_positive = np.vstack((positive_labeled[:start_idx], positive_labeled[end_idx:]))

        # 划分阴性样本
        test_negative = negative_labeled[start_idx:end_idx]
        train_negative = np.vstack((negative_labeled[:start_idx], negative_labeled[end_idx:]))

        # 构建训练集和测试集
        train_data = np.vstack((train_positive, train_negative))
        test_data = np.vstack((test_positive, test_negative))

        train_data_folds.append(train_data)
        test_data_folds.append(test_data)

    return train_data_folds, test_data_folds


def build_triples(in_file: str,
                  neg_file: str,
                  seed: int = 0,
                  k_fold: int = 5):
    """
    主流程：构建样本三元组并进行五折划分
    与 CSGLMD-main/data_preprocess.py 的样本处理逻辑完全一致
    返回：
    - train_data_folds: list[np.ndarray], 每折训练三元组
    - test_data_folds: list[np.ndarray], 每折测试三元组
    - total_data: np.ndarray, 所有三元组（正负合并，仅供需要时使用）
    - meta: dict, 简要信息
    """
    # 正样本
    positive = load_positive(in_file, seed)
    # 负样本全集
    negative_all = load_negative_all(neg_file, seed)
    # 采样与正样本等量的负样本
    negative = sample_negative(negative_all, positive.shape[0])
    # 附加标签
    pos_l, neg_l, neg_all_l = attach_labels(positive, negative, negative_all)
    # 五折划分
    train_folds, test_folds = kfold_split_triples(pos_l, neg_l, k_fold=k_fold)

    total_data = np.vstack((pos_l, neg_l))
    meta = {
        "pos_count": int(pos_l.shape[0]),
        "neg_count": int(neg_l.shape[0]),
        "neg_all_count": int(neg_all_l.shape[0]),
        "folds": int(k_fold),
        "fold_size": int(pos_l.shape[0] // k_fold) if k_fold > 0 else int(pos_l.shape[0])
    }
    return train_folds, test_folds, total_data, meta


def save_dataset(array: np.ndarray, out_path: str, fmt: str = "npy"):
    """
    可选：保存构建的总数据或某折数据
    fmt 支持 'npy' 或 'txt'（空格分隔）
    """
    out_full = _p(out_path)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    if fmt == "npy":
        np.save(out_full, array)
    elif fmt == "txt":
        np.savetxt(out_full, array, fmt="%d")
    else:
        raise ValueError("不支持的保存格式，仅支持 'npy' 或 'txt'")