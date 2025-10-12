from __future__ import division  # 确保'/'总是执行浮点数除法
from __future__ import print_function  # 确保print是一个函数，而不是一个语句（兼容Python 2和3）

import numpy as np  # 导入NumPy库，用于高效的数值和数组操作
import scipy.sparse as sp  # 导入SciPy的稀疏矩阵模块，用于处理和操作稀疏矩阵

def normalize(mx):  # 定义一个函数，用于对矩阵进行行归一化
    '''Row-normalize sparse matrix'''  # 函数文档字符串：行归一化稀疏矩阵
    rowsum = np.array(mx.sum(1))  # 计算矩阵每行的和
    r_inv = np.power(rowsum, -1).flatten()  # 计算每行和的倒数，并将其展平为一维数组
    r_inv[np.isinf(r_inv)] = 0.  # 将无穷大的值（如果某行和为0，则倒数为无穷大）替换为0
    r_mat_inv = sp.diags(r_inv)  # 创建一个对角矩阵，对角线上的元素是r_inv
    mx = r_mat_inv.dot(mx)  # 将对角矩阵与原矩阵相乘，完成行归一化
    return mx  # 返回归一化后的矩阵

def Preproces_Data (A, test_id):  # 定义一个函数，用于在关联矩阵A中将测试集中的已知关联置为0
    copy_A = A / 1  # 创建矩阵A的一个副本，避免修改原始数据
    for i in range(test_id.shape[0]):  # 遍历测试集中的每一个样本ID
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0  # 将测试样本在关联矩阵副本中对应位置的值设为0
    return copy_A  # 返回处理后的矩阵副本

'''construct_graph (lncRNA, disease, miRNA)'''  # 字符串注释：构建包含lncRNA, disease, miRNA的异构图
def construct_graph(lncRNA_disease,  miRNA_disease, miRNA_lncRNA, lncRNA_sim, miRNA_sim, disease_sim ):  # 定义构建异构图邻接矩阵的函数
    # 水平拼接矩阵，构建异构图的第一大部分（lncRNA视角）
    # [lncRNA-lncRNA相似度, lncRNA-disease关联, lncRNA-miRNA关联]
    lnc_dis_sim = np.hstack((lncRNA_sim, lncRNA_disease, miRNA_lncRNA.T))
    # 水平拼接矩阵，构建异构图的第二大部分（disease视角）
    # [disease-lncRNA关联, disease-disease相似度, disease-miRNA关联]
    dis_lnc_sim = np.hstack((lncRNA_disease.T, disease_sim, miRNA_disease.T))
    # 水平拼接矩阵，构建异构图的第三大部分（miRNA视角）
    # [miRNA-lncRNA关联, miRNA-disease关联, miRNA-miRNA相似度]
    mi_lnc_dis = np.hstack((miRNA_lncRNA, miRNA_disease, miRNA_sim))

    # 垂直拼接上述三个大部分，形成一个完整的、大的邻接矩阵
    matrix_A = np.vstack((lnc_dis_sim, dis_lnc_sim, mi_lnc_dis))
    return matrix_A  # 返回构建好的异构图邻接矩阵

'''Norm'''  # 字符串注释：归一化
def lalacians_norm(adj):  # 定义一个函数，用于对邻接矩阵进行对称拉普拉斯归一化
    # adj += np.eye(adj.shape[0]) # 注释掉的代码：添加自环（self-loop），这里没有使用
    degree = np.array(adj.sum(1))  # 计算邻接矩阵每个节点的度（每行的和）
    D = []  # 初始化一个列表，用于存储度矩阵的-0.5次方
    for i in range(len(degree)):  # 遍历每个节点的度
        if degree[i] != 0:  # 如果节点的度不为0
            de = np.power(degree[i], -0.5)  # 计算度的-0.5次方
            D.append(de)  # 添加到列表中
        else:  # 如果节点的度为0
            D.append(0)  # 添加0，避免除以0的错误
    degree = np.diag(np.array(D))  # 将列表D转换为一个对角矩阵 D^(-0.5)
    # 执行对称归一化公式: D^(-0.5) * A * D^(-0.5)
    norm_A = degree.dot(adj).dot(degree)
    # norm_A = degree.dot(adj) # 注释掉的代码：这是另一种归一化方式（左归一化）
    return norm_A  # 返回归一化后的邻接矩阵