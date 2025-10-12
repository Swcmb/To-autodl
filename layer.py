import math  # 导入数学库，用于后续的参数初始化
import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入PyTorch的神经网络模块，用于构建网络层
import torch.nn.functional as F  # 导入PyTorch的函数模块，用于激活函数、dropout等
from torch_geometric.nn import GCNConv  # 从PyTorch Geometric库导入图卷积网络层
from parms_setting import settings  # 从本地的parms_setting.py文件导入settings函数
args = settings()  # 调用settings函数获取并存储所有超参数


def reset_parameters(w):  # 定义一个函数来重置权重参数
    stdv = 1. / math.sqrt(w.size(0))  # 计算标准差，用于均匀分布的范围
    w.data.uniform_(-stdv, stdv)  # 在[-stdv, stdv]范围内均匀地初始化权重数据

class Discriminator(nn.Module):  # 定义一个判别器类，用于对比学习
    def __init__(self, n_h):  # 初始化方法，n_h是隐藏层维度
        super(Discriminator, self).__init__()  # 调用父类的初始化方法
        self.f_k = nn.Bilinear(n_h, n_h, 1)  # 定义一个双线性层，用于计算两个输入的相似度得分

        for m in self.modules():  # 遍历模型的所有模块
            self.weights_init(m)  # 对每个模块进行权重初始化

    def weights_init(self, m):  # 定义权重初始化方法
        if isinstance(m, nn.Bilinear):  # 如果模块是双线性层
            torch.nn.init.xavier_uniform_(m.weight.data)  # 使用Xavier均匀分布初始化权重
            if m.bias is not None:  # 如果有偏置项
                m.bias.data.fill_(0.0)  # 将偏置项填充为0

    def forward(self, c, h_pl, h_mi, s_bias1 = None, s_bias2 = None):  # 定义前向传播
        c_x = c.expand_as(h_pl)  # 将全局图表示c扩展成与局部节点表示h_pl相同的形状

        sc_1 = self.f_k(h_pl, c_x)  # 计算正样本（原始图节点）与全局表示的相似度得分
        sc_2 = self.f_k(h_mi, c_x)  # 计算负样本（损坏图节点）与全局表示的相似度得分

        if s_bias1 is not None:  # 如果有额外的偏置项1
            sc_1 += s_bias1  # 添加到正样本得分上
        if s_bias2 is not None:  # 如果有额外的偏置项2
            sc_2 += s_bias2  # 添加到负样本得分上

        logits = torch.cat((sc_1, sc_2), 1)  # 将正负样本的得分拼接在一起

        return logits  # 返回拼接后的得分


class AvgReadout(nn.Module):  # 定义一个平均读出（Readout）层，用于从节点表示生成图级别的表示
    def __init__(self):  # 初始化方法
        super(AvgReadout, self).__init__()  # 调用父类初始化

    def forward(self, seq, msk=None):  # 定义前向传播，seq是节点表示序列
        if msk is None:  # 如果没有提供掩码（mask）
            return torch.mean(seq, 0)  # 直接计算所有节点表示的平均值
        else:  # 如果提供了掩码
            msk = torch.unsqueeze(msk, -1)  # 扩展掩码的维度以进行广播
            return torch.sum(seq * msk, 0) / torch.sum(msk)  # 计算加权平均

class MLP(nn.Module):  # 定义一个简单的多层感知机（MLP）
    def __init__(self, in_channels, out_channels):  # 初始化方法，输入和输出通道数
        super(MLP, self).__init__()  # 调用父类初始化

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)  # 第一个线性层
        self.linear2 = nn.Linear(2 * out_channels, out_channels)  # 第二个线性层

    def forward(self, x):  # 定义前向传播
        x = F.relu(self.linear1(x))  # 通过第一个线性层并使用ReLU激活函数
        x = self.linear2(x)  # 通过第二个线性层

        return x  # 返回结果


class CSGLMD(nn.Module):    # 定义主模型CSGLMD
    def __init__(self, feature, hidden1, hidden2, decoder1, dropout):  # 初始化方法，接收各层维度和dropout率
        super(CSGLMD, self).__init__()  # 调用父类初始化

        self.encoder_o1 = GCNConv(feature, hidden1)  # 定义第一个图卷积编码器层
        self.prelu1_o1 = nn.PReLU(hidden1)  # 为第一层定义PReLU激活函数
        self.encoder_o2 = GCNConv(hidden1, hidden2)  # 定义第二个图卷积编码器层
        self.prelu_o2 = nn.PReLU(hidden2)  # 为第二层定义PReLU激活函数

        self.mlp = torch.nn.ModuleList()  # 创建一个模块列表来存储MLP层
        for i in range(1):  # 循环1次
            self.mlp.append(nn.Linear(hidden2, hidden2))  # 添加一个线性层到列表中（用于对抗学习）

        self.mlp1 = nn.Linear(hidden2, hidden2)  # 另一个独立的线性层（用于图级别表示）
        self.decoder1 = nn.Linear(hidden2 * 4, decoder1)  # 解码器的第一个线性层
        self.decoder2 = nn.Linear(decoder1, 1)  # 解码器的第二个线性层，输出最终预测值


        self.disc = Discriminator(hidden2)  # 实例化之前定义的判别器

        self.dropout = dropout  # 存储dropout率
        self.sigm = nn.Sigmoid()  # 定义Sigmoid激活函数
        self.read = AvgReadout()  # 实例化平均读出层

    def forward(self, data_o, data_a, idx):  # 定义模型的主前向传播逻辑
        x_o, adj = data_o.x, data_o.edge_index  # 从原始图数据中解包节点特征和邻接信息

        x_a = data_a.x  # 获取损坏图的节点特征

        # original graph encoder  # 注释：原始图编码器部分
        x1_o = self.prelu1_o1(self.encoder_o1(x_o, adj))  # 第一层GCN编码和PReLU激活
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)  # 应用dropout

        x2_o = self.encoder_o2(x1_o, adj)  # 第二层GCN编码
        x2_o = self.prelu_o2(x2_o)  # PReLU激活，得到原始图的最终节点表示

        # corrupt graph encoder  # 注释：损坏图编码器部分
        x1_o_a = self.prelu1_o1(self.encoder_o1(x_a, adj))  # 对损坏图进行第一层GCN编码和激活
        x1_o_a = F.dropout(x1_o_a, self.dropout, training=self.training)  # 应用dropout

        x2_o_a = self.encoder_o2(x1_o_a, adj)  # 第二层GCN编码
        x2_o_a = self.prelu_o2(x2_o_a)  # PReLU激活，得到损坏图的最终节点表示

        # graph level representation  # 注释：图级别表示生成
        h_os = self.read(x2_o)  # 对原始图节点表示进行读出操作，得到图级别表示
        h_os = self.sigm(h_os)  # 通过Sigmoid激活
        h_os = self.mlp1(h_os)  # 通过一个线性层

        h_os_a = self.read(x2_o_a)  # 对损坏图节点表示进行读出操作
        h_os_a = self.sigm(h_os_a)  # 通过Sigmoid激活
        h_os_a = self.mlp1(h_os_a)  # 通过一个线性层
        # print("h: ",h_os.shape, h_os_a.shape)  # 被注释掉的调试语句

        # Adversarial learning  # 注释：对抗学习部分
        sc_1 = x2_o.squeeze(0)  # 原始图节点表示
        sc_2 = x2_o_a.squeeze(0)  # 损坏图节点表示
        for i, lin in enumerate(self.mlp):  # 通过MLP层
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)
        sc_1 = sc_1.sum(1).unsqueeze(0)  # 对特征求和，得到一个分数
        sc_2 = sc_2.sum(1).unsqueeze(0)  # 对特征求和，得到一个分数
        logits = torch.cat((sc_1, sc_2),1)  # 拼接两个分数，用于对抗性损失

        # contrastive learning  # 注释：对比学习部分
        ret_os = self.disc(h_os, x2_o, x2_o_a)  # 计算原始图的对比损失（正样本：原始节点，负样本：损坏节点）
        ret_os_a = self.disc(h_os_a, x2_o_a, x2_o)  # 计算损坏图的对比损失（正样本：损坏节点，负样本：原始节点）
        
        # 根据任务类型，从批次索引idx中提取要预测的实体对的节点嵌入
        if args.task_type == 'LDA':  # 如果是lncRNA-Disease任务
            entity1 = x2_o[idx[0]]  # 获取lncRNA节点的嵌入
            entity2 = x2_o[idx[1] + 240]  # 获取disease节点的嵌入（240是lncRNA节点数量的偏移）
            # dataset2
            # entity1 = x2_o[idx[0]]
            # entity2 = x2_o[idx[1] + 665]

        if args.task_type == 'MDA':  # 如果是miRNA-Disease任务
            #dataset1
            entity1 = x2_o[idx[0] + 645]  # 获取miRNA节点的嵌入（645是lncRNA+disease节点数量的偏移）
            entity2 = x2_o[idx[1] + 240]  # 获取disease节点的嵌入

            # dataset2
            # entity1 = x2_o[idx[0] + 981]
            # entity2 = x2_o[idx[1] + 665]

        if args.task_type == 'LMI':  # 如果是lncRNA-miRNA任务
            # dataset1
            entity1 = x2_o[idx[0]]  # 获取lncRNA节点的嵌入
            entity2 = x2_o[idx[1] + 645]  # 获取miRNA节点的嵌入

        # multi-relationship modelling decoder  # 注释：多关系建模解码器
        add = entity1 + entity2  # 实体嵌入相加
        product = entity1 * entity2  # 实体嵌入逐元素相乘
        concatenate = torch.cat((entity1, entity2), dim=1)  # 实体嵌入拼接

        feature = torch.cat((add, product, concatenate), dim=1)  # 将三种组合特征拼接成最终的交互特征

        log1 = F.relu(self.decoder1(feature))  # 通过解码器的第一层并使用ReLU激活
        log = self.decoder2(log1)  # 通过解码器的第二层得到最终的预测得分

        return log, ret_os, ret_os_a, x2_o, logits, log1  # 返回多个结果，用于计算不同的损失和分析