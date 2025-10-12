import numpy as np  # 导入numpy库，用于进行数值计算，特别是数组和矩阵操作
import copy  # 导入copy库，用于创建对象的副本

"positive sample in test set to 0"  # 这是一个字符串注释，说明下面的函数功能是将测试集中的阳性样本置为0
def Preproces_Data(A, test_id):  # 定义数据预处理函数，用于在计算相似度前，将测试集中的已知关联暂时移除
    copy_A = A / 1  # 创建关联矩阵A的一个副本，避免修改原始数据
    for i in range(test_id.shape[0]):  # 遍历测试集中的每一个样本ID
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0  # 将测试样本在关联矩阵副本中对应位置的值设为0
    return copy_A  # 返回处理后的矩阵副本

"Gaussiankernel similarity"  # 字符串注释，说明下面是关于高斯核相似度计算的函数
def calculate_kernel_bandwidth(A):  # 定义函数，用于计算高斯核的带宽参数(gamma)
    IP_0 = 0  # 初始化变量IP_0，用于累加每个向量的范数平方
    for i in range(A.shape[0]):  # 遍历矩阵A的每一行（每个实体的关联谱）
        IP = np.square(np.linalg.norm(A[i]))  # 计算当前行向量的L2范数的平方
        # print(IP)  # 这是一个被注释掉的调试语句，用于打印每个向量的范数平方
        IP_0 += IP  # 将计算出的范数平方累加到IP_0
    lambd = 1/((1/A.shape[0]) * IP_0)  # 根据公式计算带宽参数lambda，通常是基于平均范数平方的倒数
    return lambd  # 返回计算得到的带宽参数

def calculate_GaussianKernel_sim(A):  # 定义函数，用于计算基于关联谱A的高斯核相似度矩阵
    kernel_bandwidth = calculate_kernel_bandwidth(A)  # 调用前面的函数计算高斯核的带宽
    gauss_kernel_sim = np.zeros((A.shape[0],A.shape[0]))  # 初始化一个方阵，用于存储最终的高斯核相似度
    for i in range(A.shape[0]):  # 遍历矩阵A的每一行
        for j in range(A.shape[0]):  # 再次遍历矩阵A的每一行，以计算两两之间的相似度
            gaussianKernel = np.exp(-kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j])))  # 计算向量A[i]和A[j]之间的高斯核相似度
            gauss_kernel_sim[i][j] = gaussianKernel  # 将计算出的相似度值存入相似度矩阵
            # print("gau",gauss_kernel_sim)  # 这是一个被注释掉的调试语句

    return gauss_kernel_sim  # 返回完整的高斯核相似度矩阵

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
    RNASiNet = np.zeros((RNAlen, RNAlen))  # 初始化一个方阵来存储RNA功能相似度
    for i in range(RNAlen):  # 遍历所有RNA
        for j in range(i + 1, RNAlen):  # 遍历i之后的RNA，避免重复计算（因为矩阵是对称的）
            RNASiNet[i, j] = RNASiNet[j, i] = PBPA(i, j, diSiNet, rna_di)  # 调用PBPA函数计算并填充对称矩阵
    RNASiNet = RNASiNet + np.eye(RNAlen)  # 将矩阵对角线元素设为1（每个RNA与自身的相似度为1）
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
    print(lnc_dis.shape,mi_dis.shape,lnc_mi.shape,dis_sem_sim.shape)  # 打印加载的dataset1各个矩阵的形状

    'dataset2'  # 字符串注释，说明下面加载的是dataset2的数据
    lnc_dis = np.loadtxt("dataset1/lnc_dis.txt")  # 从文件加载lncRNA-disease关联矩阵 (注意：路径原文为dataset1，可能是一个笔误)
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")  # 从文件加载miRNA-disease关联矩阵 (注意：路径原文为dataset1)
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")  # 从文件加载lncRNA-miRNA关联矩阵 (注意：路径原文为dataset1)
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")  # 从文件加载疾病语义相似度矩阵 (注意：路径原文为dataset1)
    print(lnc_dis.shape,mi_dis.shape,lnc_mi.shape,dis_sem_sim.shape)  # 打印加载的dataset2各个矩阵的形状

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