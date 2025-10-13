import numpy as np  # 导入numpy库，用于进行数值计算，特别是数组和矩阵操作
import copy  # 导入copy库，用于创建对象的副本
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def _pbpa_pair_idx(args):
    """
    顶层可pickle的worker：使用预计算索引计算单对(i,j)的PBPA值
    args: (i, j, di_sim, nz_idx)
    返回: (i, j, value)
    """
    i, j, di_sim, nz_idx = args
    idx_i = nz_idx[i]
    idx_j = nz_idx[j]
    if len(idx_i) == 0 or len(idx_j) == 0:
        return (i, j, 0.0)
    sub = di_sim[np.ix_(idx_i, idx_j)]
    # (sum max by columns + sum max by rows) / (rows + cols)
    v = (np.max(sub, axis=0).sum() + np.max(sub, axis=1).sum()) / (sub.shape[0] + sub.shape[1])
    return (i, j, v)

"positive sample in test set to 0"  # 这是一个字符串注释，说明下面的函数功能是将测试集中的阳性样本置为0
def Preproces_Data(A, test_id):  # 定义数据预处理函数，用于在计算相似度前，将测试集中的已知关联暂时移除
    copy_A = A / 1  # 创建关联矩阵A的一个副本，避免修改原始数据
    for i in range(test_id.shape[0]):  # 遍历测试集中的每一个样本ID
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0  # 将测试样本在关联矩阵副本中对应位置的值设为0
    return copy_A  # 返回处理后的矩阵副本

"Gaussiankernel similarity"  # 字符串注释，说明下面是关于高斯核相似度计算的函数
def calculate_kernel_bandwidth(A):  # 定义函数，用于计算高斯核的带宽参数(gamma)
    # 向量化：按行求范数平方并求和，避免Python层循环
    IP_0 = np.sum(np.sum(A * A, axis=1))
    lambd = 1.0 / ((1.0 / A.shape[0]) * IP_0 + 1e-12)
    return lambd  # 返回计算得到的带宽参数

def calculate_GaussianKernel_sim(A):  # 定义函数，用于计算基于关联谱A的高斯核相似度矩阵
    # 向量化：Gram矩阵与行范数计算所有成对距离，再做RBF
    gamma = calculate_kernel_bandwidth(A)
    row_norm_sq = np.sum(A * A, axis=1)  # (n,)
    G = A @ A.T  # BLAS多线程
    D = row_norm_sq[:, None] + row_norm_sq[None, :] - 2.0 * G
    D = np.maximum(D, 0.0)
    return np.exp(-gamma * D)  # 返回完整的高斯核相似度矩阵

"Functional similarity"  # 字符串注释，说明下面是关于功能相似度计算的函数
def PBPA(RNA_i, RNA_j, di_sim, rna_di):  # 定义PBPA函数，计算两个RNA（i和j）之间的功能相似度
    diseaseSet_i = rna_di[RNA_i] > 0  # 获取与RNA_i相关联的疾病集合的布尔索引
    diseaseSet_j = rna_di[RNA_j] > 0  # 获取与RNA_j相关联的疾病集合的布尔索引
    diseaseSim_ij = di_sim[diseaseSet_i][:, diseaseSet_j]  # 提取这两个疾病集合之间的相似度子矩阵
    ijshape = diseaseSim_ij.shape  # 获取该子矩阵的形状
    if ijshape[0] == 0 or ijshape[1] == 0:  # 如果任一RNA没有关联的疾病，则子矩阵为空
        return 0  # 在这种情况下，它们的功能相似度为0
    # 计算功能相似度：对两个疾病集合，分别计算一个集合中每个疾病与另一个集合中疾病的最大相似度，然后求和并归一化
    return (sum(np.max(diseaseSim_ij, axis=0)) + sum(np.max(diseaseSim_ij, axis=1))) / (ijshape[0] + ijshape[1])

def getRNA_functional_sim(RNAlen, diSiNet, rna_di):  # 定义函数，用于计算所有RNA对之间的功能相似度
    """
    支持并行（多进程）计算PBPA(i,j)，受环境变量控制：
      - EM_WORKERS: 进程数（>1启用并行），默认从CPU探测并上限32
      - EM_CHUNK_SIZE: 任务切片大小，默认10000
    当进程数<=1时，退化为原始串行实现以保持可复现。
    """
    workers = int(os.environ.get("EM_WORKERS", "1"))
    chunk_size = int(os.environ.get("EM_CHUNK_SIZE", "10000"))
    RNASiNet = np.zeros((RNAlen, RNAlen), dtype=float)

    if workers <= 1 or RNAlen <= 2:
        # 串行回退（使用预计算索引以降低循环内开销）
        nz_idx = [np.flatnonzero(rna_di[row] > 0) for row in range(RNAlen)]
        for i in range(RNAlen):
            idx_i = nz_idx[i]
            for j in range(i + 1, RNAlen):
                idx_j = nz_idx[j]
                if len(idx_i) == 0 or len(idx_j) == 0:
                    val = 0.0
                else:
                    sub = diSiNet[np.ix_(idx_i, idx_j)]
                    val = (np.max(sub, axis=0).sum() + np.max(sub, axis=1).sum()) / (sub.shape[0] + sub.shape[1])
                RNASiNet[i, j] = RNASiNet[j, i] = val
        np.fill_diagonal(RNASiNet, 1.0)
        return RNASiNet

    # 构造(i,j)对列表（上三角）
    pairs = [(i, j) for i in range(RNAlen) for j in range(i + 1, RNAlen)]
    # 预计算每行非零索引（小结构，易序列化）
    nz_idx = [np.flatnonzero(rna_di[row] > 0) for row in range(RNAlen)]
    # 迭代器打包参数，避免闭包捕获
    args_iter = ((i, j, diSiNet, nz_idx) for (i, j) in pairs)
    max_workers = min(32, max(1, workers))
    # 为executor.map设置合理chunksize：按总任务数/进程数粗略切分
    total_tasks = len(pairs)
    chunk = max(1, min(chunk_size, (total_tasks // max_workers) or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i, j, v in ex.map(_pbpa_pair_idx, args_iter, chunksize=chunk):
            RNASiNet[i, j] = v
            RNASiNet[j, i] = v

    np.fill_diagonal(RNASiNet, 1.0)
    return RNASiNet  # 返回RNA功能相似度网络

"label instantiation"  # 字符串注释，意为“标签实例化”，指将相似度值二值化
def label_preprocess(sim_matrix):  # 定义一个函数，用于对相似度矩阵进行阈值处理
    new_sim_matrix = np.zeros(shape=sim_matrix.shape)  # 创建一个与输入矩阵形状相同、元素全为0的新矩阵
    # print(lnc_sim_matrix.shape)  # 这是一个被注释掉的调试语句
    for i in range(sim_matrix.shape[0]):  # 遍历矩阵的行
        for j in range(sim_matrix.shape[1]):  # 遍历矩阵的列
            if sim_matrix[i][j] >= 0.8:  # 如果原始相似度值大于或等于0.8
                new_sim_matrix[i][j] = 1  # 在新矩阵的对应位置设为1
    return new_sim_matrix  # 返回处理后的二值化矩阵


def RNA_fusion_sim (G1, G2, F):  # 定义函数，用于融合两种高斯相似度(G1, G2)和功能相似度(F)
    fusion_sim = np.zeros((len(G1),len(G2)))  # 初始化融合后的相似度矩阵
    G = (G1+G2)/2  # 计算两种高斯相似度矩阵的平均值
    for i in range (len(G1)):  # 遍历矩阵的行
        for j in range(len(G1)):  # 遍历矩阵的列
            if F[i][j] > 0 :  # 如果功能相似度F中对应元素大于0
                fusion_sim[i][j] = F[i][j]  # 则优先使用功能相似度
            else:  # 否则
                fusion_sim[i][j] = G[i][j]  # 使用平均后的高斯相似度
    fusion_sim = label_preprocess(fusion_sim)  # 对融合后的相似度矩阵进行二值化处理
    return fusion_sim  # 返回最终的RNA融合相似度矩阵

def dis_fusion_sim (G1, G2, SD):  # 定义函数，用于融合两种疾病高斯相似度(G1, G2)和语义相似度(SD)
    fusion_sim = (SD+(G1+G2)/2)/2  # 计算语义相似度与平均高斯相似度的平均值
    fusion_sim = label_preprocess(fusion_sim)  # 对融合后的相似度矩阵进行二值化处理
    return fusion_sim  # 返回最终的疾病融合相似度矩阵


if __name__ == '__main__':  # Python主程序入口，当该脚本被直接运行时，以下代码将被执行

    'dataset1'  # 字符串注释，说明下面加载的是dataset1的数据
    lnc_dis = np.loadtxt("dataset1/lnc_dis_association.txt")  # 从文件加载lncRNA-disease关联矩阵
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")  # 从文件加载miRNA-disease关联矩阵
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")  # 从文件加载lncRNA-miRNA关联矩阵
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")  # 从文件加载疾病语义相似度矩阵
    from log_output_manager import get_logger
    _logger = get_logger()
    _logger.info(f"{lnc_dis.shape} {mi_dis.shape} {lnc_mi.shape} {dis_sem_sim.shape}")

    'dataset2'  # 字符串注释，说明下面加载的是dataset2的数据
    lnc_dis = np.loadtxt("dataset1/lnc_dis.txt")  # 从文件加载lncRNA-disease关联矩阵 (注意：路径原文为dataset1，可能是一个笔误)
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")  # 从文件加载miRNA-disease关联矩阵 (注意：路径原文为dataset1)
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")  # 从文件加载lncRNA-miRNA关联矩阵 (注意：路径原文为dataset1)
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")  # 从文件加载疾病语义相似度矩阵 (注意：路径原文为dataset1)
    _logger = get_logger()
    _logger.info(f"{lnc_dis.shape} {mi_dis.shape} {lnc_mi.shape} {dis_sem_sim.shape}")

    "this example use all sample to calculate"  # 字符串注释，说明这个示例使用所有样本进行计算

    # lnc_dis_test_id = np.loadtxt("dataset1/lnc_dis_test_id1.txt")  # 被注释掉的代码：加载lncRNA-disease测试集ID
    # mi_dis_test_id = np.loadtxt("dataset1/mi_dis_test_id1.txt")  # 被注释掉的代码：加载miRNA-disease测试集ID
    # mi_lnc_test_id = np.loadtxt("dataset1/mi_lnc_test_id1.txt")  # 被注释掉的代码：加载miRNA-lncRNA测试集ID
    #
    # "Zeroing of the association matrix"  # 被注释掉的字符串注释
    # lnc_dis = Preproces_Data(lnc_dis,lnc_dis_test_id)  # 被注释掉的代码：对lncRNA-disease关联矩阵进行预处理
    # mi_dis = Preproces_Data(mi_dis,mi_dis_test_id)  # 被注释掉的代码：对miRNA-disease关联矩阵进行预处理
    # mi_lnc = Preproces_Data(lnc_mi.T,mi_lnc_test_id)   # 被注释掉的代码：对转置后的lncRNA-miRNA关联矩阵进行预处理
    # # print(mi_lnc.shape)  # 被注释掉的调试语句

    "lncRNA similarity"  # 字符串注释，说明下面开始计算lncRNA相似度
    lnc_gau_1 = calculate_GaussianKernel_sim(lnc_dis)  # 基于lncRNA-disease关联计算lncRNA的高斯核相似度
    lnc_gau_2 = calculate_GaussianKernel_sim(lnc_mi)   # 基于lncRNA-miRNA关联计算lncRNA的高斯核相似度
    lnc_fun = getRNA_functional_sim(RNAlen=len(lnc_dis), diSiNet=copy.copy(dis_sem_sim), rna_di=copy.copy(lnc_dis))  # 计算lncRNA的功能相似度
    lnc_sim = RNA_fusion_sim(lnc_gau_1,lnc_gau_2,lnc_fun)  # 融合三种相似度得到最终的lncRNA相似度

    "miRNA similarity"  # 字符串注释，说明下面开始计算miRNA相似度
    mi_gau_1 = calculate_GaussianKernel_sim(mi_dis)     # 基于miRNA-disease关联计算miRNA的高斯核相似度
    mi_gau_2 = calculate_GaussianKernel_sim(lnc_mi.T)     # 基于miRNA-lncRNA关联(lnc_mi转置)计算miRNA的高斯核相似度
    mi_fun = getRNA_functional_sim(RNAlen=len(mi_dis), diSiNet=copy.copy(dis_sem_sim), rna_di=copy.copy(mi_dis))  # 计算miRNA的功能相似度
    mi_sim = RNA_fusion_sim(mi_gau_1,mi_gau_2,mi_fun)  # 融合三种相似度得到最终的miRNA相似度

    "disease similarity"  # 字符串注释，说明下面开始计算disease相似度
    dis_gau_1 = calculate_GaussianKernel_sim(lnc_dis.T)  # 基于disease-lncRNA关联(lnc_dis转置)计算disease的高斯核相似度
    dis_gau_2 = calculate_GaussianKernel_sim(mi_dis.T)   # 基于disease-miRNA关联(mi_dis转置)计算disease的高斯核相似度
    dis_sim = dis_fusion_sim(dis_gau_1,dis_gau_2,dis_sem_sim)  # 融合两种高斯相似度和语义相似度得到最终的疾病相似度