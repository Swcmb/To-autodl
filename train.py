import torch  # 导入PyTorch主库
import numpy as np  # 导入NumPy库，用于数值运算
import torch.nn as nn  # 导入PyTorch的神经网络模块
import matplotlib.pyplot as plt  # 导入matplotlib的绘图库（在此文件中未使用）
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, f1_score, auc  # 从scikit-learn导入各种评估指标函数
from log_output_manager import get_logger
from torch.cuda.amp import autocast, GradScaler
# 使用 main.py 的 redirect_print 统一重定向输出，无需在此处绑定 logger


def train_model(model, optimizer, data_o, data_a, train_loader, test_loader, args):  # 定义主训练函数
    m = torch.nn.Sigmoid()  # 实例化Sigmoid函数，用于将模型输出转换为概率
    loss_fct = torch.nn.BCELoss()  # 实例化二元交叉熵损失函数（用于主任务）
    b_xent = nn.BCEWithLogitsLoss()  # 实例化带Logits的二元交叉熵损失，更稳定（用于对比和对抗损失）
    node_loss = nn.BCEWithLogitsLoss()  # 同上，用于节点级别的对抗损失
    loss_history = []  # 创建一个列表来记录每个批次的损失值

    if args.cuda:  # 检查是否使用GPU
        model.to('cuda')  # 将模型移动到GPU
        data_o.to('cuda')  # 将原始图数据移动到GPU
        data_a.to('cuda')  # 将对抗图数据移动到GPU
    scaler = GradScaler(enabled=bool(getattr(args, "cuda", False)))

    # Train model  # 注释：训练模型
    lbl = data_a.y  # 获取对抗数据的标签（用于对比学习）
    print('Start Training...')  # 打印开始训练的信息

    for epoch in range(args.epochs):  # 开始按设定的轮数进行训练循环
        print('-------- Epoch ' + str(epoch + 1) + ' --------')  # 打印当前轮数
        y_pred_train = []  # 初始化列表，用于存储当前轮次的预测值
        y_label_train = []  # 初始化列表，用于存储当前轮次的真实标签
        loss_train = torch.tensor(0.0) # 初始化训练损失，防止在训练加载器为空时引用错误

        # 为节点级别的对抗损失创建标签
        lbl_1 = torch.ones(1, 1140)  # 创建全为1的标签（对应原始图节点）
        lbl_2 = torch.zeros(1, 1140)  # 创建全为0的标签（对应损坏图节点）
        lbl2 = torch.cat((lbl_1, lbl_2),1).cuda(non_blocking=True)  # 拼接并移动到GPU

        for i, (label, inp) in enumerate(train_loader):  # 遍历训练数据加载器，获取每个批次的标签和输入

            if args.cuda:  # 如果使用GPU
                label = label.cuda(non_blocking=True)  # 将批次标签移动到GPU（非阻塞）

            model.train()  # 将模型设置为训练模式
            optimizer.zero_grad()  # 清除上一批次的梯度
            with autocast(enabled=bool(getattr(args, "cuda", False))):
                output, cla_os, cla_os_a, _, logits, log1 = model(data_o, data_a, inp)  # 将数据输入模型，获取多个输出

                log = torch.squeeze(m(output))  # 对主任务输出应用Sigmoid并压缩维度
                loss1 = loss_fct(log, label.float())  # 计算主任务的二元交叉熵损失
                loss2 = b_xent(cla_os, lbl.float())  # 计算第一个对比损失
                loss3 = b_xent(cla_os_a, lbl.float())  # 计算第二个对比损失
                loss4 = node_loss(logits, lbl2.float())  # 计算节点级别的对抗损失
                # 根据预设的权重，将四个损失加权求和得到总损失
                loss_train = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3 \
                             + args.loss_ratio4 * loss4
            # print("loss_train: ",loss_train)  # 被注释掉的调试语句

            loss_history.append(loss_train.item())  # 记录当前批次的总损失
            scaler.scale(loss_train).backward()  # AMP缩放反向传播
            scaler.step(optimizer)  # AMP优化器步进
            scaler.update()  # AMP缩放器更新

            label_ids = label.to('cpu').numpy()  # 将标签移回CPU并转为numpy数组
            y_label_train = y_label_train + label_ids.flatten().tolist()  # 收集真实标签
            y_pred_train = y_pred_train + log.flatten().tolist()  # 收集预测概率

            if i % 100 == 0:  # 每100个批次
                # 打印当前轮次、迭代次数和训练损失
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))

        # 在训练集非空时计算AUROC，否则为0，避免程序出错
        roc_train = roc_auc_score(y_label_train, y_pred_train) if y_label_train else 0.0

        # 打印当前轮次的总结信息
        print('epoch: {:04d}'.format(epoch + 1),'loss_train: {:.4f}'.format(loss_train.item()),
                'auroc_train: {:.4f}'.format(roc_train))

        if hasattr(torch.cuda, 'empty_cache'):  # 如果PyTorch版本支持
            torch.cuda.empty_cache()  # 清空GPU缓存，释放不必要的显存
    print("Optimization Finished!")  # 所有轮次训练完成后，打印优化完成

    # Testing  # 注释：测试阶段
    # 调用test函数，在测试集上评估最终模型
    auroc_test, prc_test, f1_test, loss_test = test(model, test_loader, data_o, data_a, args)
    # 打印测试集上的各项性能指标
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))
    
    # 返回测试结果
    return {'auroc': auroc_test, 'auprc': prc_test, 'f1': f1_test, 'loss': loss_test.item()}


def test(model, loader, data_o, data_a, args):  # 定义测试函数

    m = torch.nn.Sigmoid()  # 实例化Sigmoid
    loss_fct = torch.nn.BCELoss()  # 实例化损失函数
    b_xent = nn.BCEWithLogitsLoss()
    node_loss = nn.BCEWithLogitsLoss()


    model.eval()  # 将模型设置为评估模式（会关闭dropout等）
    y_pred = []  # 初始化列表，用于存储预测值
    y_label = []  # 初始化列表，用于存储真实标签
    loss = torch.tensor(0.0) # 初始化损失，防止在加载器为空时引用错误
    lbl = data_a.y  # 获取对抗数据的标签

    # 同样为对抗损失创建标签
    lbl_1 = torch.ones(1, 1140)
    lbl_2 = torch.zeros(1, 1140)
    lbl2 = torch.cat((lbl_1, lbl_2), 1).cuda(non_blocking=True)

    with torch.no_grad():  # 在此代码块中，不计算梯度，以节省计算资源
        for i, (label, inp) in enumerate(loader):  # 遍历测试数据加载器

            if args.cuda:  # 如果使用GPU
                label = label.cuda(non_blocking=True)  # 将标签移动到GPU（非阻塞）

            with autocast(enabled=bool(getattr(args, "cuda", False))):
                output, cla_os, cla_os_a, _, logits, log1 = model(data_o, data_a, inp)  # 前向传播
                log = torch.squeeze(m(output))  # 获取主任务预测概率

                # 计算测试集上的损失（尽管在测试阶段通常更关心指标而非损失值）
                loss1 = loss_fct(log, label.float())
                loss2 = b_xent(cla_os, lbl.float())
                loss3 = b_xent(cla_os_a, lbl.float())
                loss4 = node_loss(logits, lbl2.float())
                loss = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3 \
                       + args.loss_ratio4 * loss4

            label_ids = label.to('cpu').numpy()  # 将标签移回CPU
            y_label = y_label + label_ids.flatten().tolist()  # 收集真实标签
            y_pred = y_pred + log.flatten().tolist()  # 收集预测概率
    
    # 如果测试集为空，则返回0，避免程序崩溃
    if not y_label:
        return 0.0, 0.0, 0.0, loss

    # 在循环结束后，根据所有批次的预测概率计算硬预测（0或1）
    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    # 计算并返回测试集上的AUROC, AUPRC, F1分数和平均损失
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss