import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

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


class MoCoV2SingleView(nn.Module):
    """
    单视图 MoCo v2：原始图为 Query，增强图为 Key。
    - 使用投影头（MLP）将编码器输出映射到对比空间
    - 使用动量更新的 key 投影头
    - 维护一个特征队列（memory bank）作为负样本
    - 输出 InfoNCE 所需的 logits 和 targets（targets 恒为0）
    """
    def __init__(self, base_dim: int, proj_dim: int, K: int = 4096, m: float = 0.999, T: float = 0.2, queue_warmup_steps: int = 0, debug: bool = False):
        super().__init__()
        assert proj_dim is not None and proj_dim > 0, "proj_dim must be positive"
        assert K > 0, "queue size K must be positive"
        self.K = int(K)
        self.m = float(m)
        self.T = float(T)
        self.queue_warmup_steps = int(queue_warmup_steps)
        self.debug = bool(debug)
        self.global_step = 0
        self._filled = 0

        # query/key 投影头
        self.q_proj = nn.Sequential(
            nn.Linear(base_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(base_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

        # 初始化 key 投影头与 q 相同，并冻结梯度
        for qp, kp in zip(self.q_proj.parameters(), self.k_proj.parameters()):
            kp.data.copy_(qp.data)
            kp.requires_grad = False

        # 队列：形状 [proj_dim, K]
        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, self.K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        # 动量更新 key 投影头参数
        for param_q, param_k in zip(self.q_proj.parameters(), self.k_proj.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        # keys: [B, proj_dim]
        keys = keys.detach()
        batch_size = keys.shape[0]
        K = self.K
        ptr = int(self.queue_ptr.item())

        if batch_size <= 0:
            return

        # 分段写入，循环覆盖
        if ptr + batch_size <= K:
            self.queue[:, ptr:ptr + batch_size] = keys.t()
            ptr = (ptr + batch_size) % K
        else:
            first = K - ptr
            second = batch_size - first
            if first > 0:
                self.queue[:, ptr:] = keys[:first].t()
            if second > 0:
                self.queue[:, :second] = keys[first:].t()
            ptr = second % K

        self.queue_ptr[0] = ptr

    def forward(self, q_embed: torch.Tensor, k_embed: torch.Tensor):
        """
        输入:
          - q_embed: 原始图编码器输出 [N, base_dim]
          - k_embed: 增强图编码器输出 [N, base_dim]
        输出:
          - logits: [N, 1 + K]
          - targets: [N]，恒为0
        """
        if q_embed.dim() != 2 or k_embed.dim() != 2 or q_embed.shape != k_embed.shape:
            raise ValueError(f"q_embed and k_embed must be 2D and same shape, got {q_embed.shape} vs {k_embed.shape}")

        # 归一化的 query
        q = F.normalize(self.q_proj(q_embed), dim=1)
        # 步数自增
        self.global_step = int(self.global_step) + 1

        # 计算 key（动量分支，不反传）
        with torch.no_grad():
            self.momentum_update_key_encoder()
            k = F.normalize(self.k_proj(k_embed), dim=1)

        # 正样本相似度：逐样本点积
        l_pos = torch.sum(q * k, dim=1, keepdim=True)  # [N,1]

        # 冷启动：仅 batch 内负样本（不使用/不更新队列）
        warmup = self.global_step <= self.queue_warmup_steps
        if warmup:
            # 使用本 batch 的 k 作为负样本，排除对角
            sim = torch.matmul(q, k.t())  # [N,N]
            N = sim.size(0)
            if N > 1:
                mask = ~torch.eye(N, dtype=torch.bool, device=sim.device)
                l_neg = sim[mask].view(N, N - 1)
            else:
                # 若 batch 太小，退化为与自身队列（不会发生在正常训练）
                l_neg = torch.matmul(q, self.queue.clone().detach())
        else:
            # 正常：与队列全部向量点积
            l_neg = torch.matmul(q, self.queue.clone().detach())  # [N,K]

        logits = torch.cat([l_pos, l_neg], dim=1)  # [N, 1+K] 或 [N, 1+(N-1)]
        logits = logits / self.T
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # 轻量日志与断言（仅在 debug 时、早期输出）
        if self.debug and (self.global_step == 1 or self.global_step == self.queue_warmup_steps or self.global_step <= 3):
            with torch.no_grad():
                cos_stats = {
                    "mean": float((q * k).sum(dim=1).mean().item()),
                    "std": float((q * k).sum(dim=1).std(unbiased=False).item()) if q.size(0) > 1 else 0.0,
                    "min": float((q * k).sum(dim=1).min().item()),
                    "max": float((q * k).sum(dim=1).max().item()),
                }
                qsz = list(q.shape); ksz = list(k.shape)
                if hasattr(self, "queue"):
                    qshape = list(self.queue.shape)
                    # 填充率估计：warmup 阶段为 0，之后按累计 enqueue 次数近似
                    fill_ratio = (0.0 if warmup else float(min(self._filled, self.K)) / float(self.K))
                else:
                    qshape = []
                    fill_ratio = 0.0
                # 断言
                assert int(targets.sum().item()) == 0, "MoCo targets 应全为 0"
            print(f"[EM.moco][single] step={self.global_step} warmup={warmup} q={qsz} k={ksz} logits={list(logits.shape)} queue={qshape} fill_ratio={fill_ratio:.2f} cos={cos_stats}")

        # 更新队列（使用当前 batch 的 k）——仅在非 warmup 时启用
        if not warmup:
            self._dequeue_and_enqueue(k)
            # 更新填充估计
            self._filled = int(min(self.K, int(self._filled) + k.size(0)))

        return logits, targets

class MoCoV2MultiView(nn.Module):
    """
    多视图 MoCo v2：
    - 共享一个 q 投影头
    - 每个视图拥有独立的 k 投影头和队列
    - 返回每个视图的 (logits, targets)
    """
    def __init__(self, base_dim: int, proj_dim: int, num_views: int, K: int = 4096, m: float = 0.999, T: float = 0.2, queue_warmup_steps: int = 0, debug: bool = False):
        super().__init__()
        assert num_views >= 1, "num_views must be >= 1"
        self.num_views = int(num_views)
        self.K = int(K)
        self.m = float(m)
        self.T = float(T)
        self.queue_warmup_steps = int(queue_warmup_steps)
        self.debug = bool(debug)
        self.global_step = 0
        self._filled = [0 for _ in range(self.num_views)]

        # 共享 q 投影头
        self.q_proj = nn.Sequential(
            nn.Linear(base_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        # 独立 k 投影头
        self.k_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim),
            ) for _ in range(self.num_views)
        ])
        # 初始化各 k_proj 与 q_proj 相同，且冻结梯度
        for k_proj in self.k_projs:
            for qp, kp in zip(self.q_proj.parameters(), k_proj.parameters()):
                kp.data.copy_(qp.data)
                kp.requires_grad = False

        # 为每个视图注册独立队列与指针
        for i in range(self.num_views):
            self.register_buffer(f"queue_{i}", F.normalize(torch.randn(proj_dim, self.K), dim=0))
            self.register_buffer(f"queue_ptr_{i}", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoders(self):
        for k_proj in self.k_projs:
            for param_q, param_k in zip(self.q_proj.parameters(), k_proj.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, view_idx: int):
        keys = keys.detach()
        batch_size = keys.shape[0]
        if batch_size <= 0:
            return
        K = self.K

        queue = getattr(self, f"queue_{view_idx}")
        queue_ptr = getattr(self, f"queue_ptr_{view_idx}")

        ptr = int(queue_ptr.item())
        if ptr + batch_size <= K:
            queue[:, ptr:ptr + batch_size] = keys.t()
            ptr = (ptr + batch_size) % K
        else:
            first = K - ptr
            second = batch_size - first
            if first > 0:
                queue[:, ptr:] = keys[:first].t()
            if second > 0:
                queue[:, :second] = keys[first:].t()
            ptr = second % K

        queue_ptr[0] = ptr

    def forward(self, q_embed: torch.Tensor, k_embeds: List[torch.Tensor]):
        """
        输入:
          - q_embed: [N, base_dim]
          - k_embeds: 长度 = num_views 的列表，每项 [N, base_dim]
        输出:
          - logits_list: List[[N, 1+K]]
          - targets_list: List[[N]]
        """
        if len(k_embeds) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(k_embeds)}")
        for k in k_embeds:
            if k.dim() != 2 or k.shape != q_embed.shape:
                raise ValueError("Each k_embed must be 2D and same shape as q_embed")

        # 归一化 q
        q = F.normalize(self.q_proj(q_embed), dim=1)
        # 步数自增
        self.global_step = int(self.global_step) + 1
        warmup = self.global_step <= self.queue_warmup_steps

        logits_list = []
        targets_list = []
        if self.debug and not (len(k_embeds) == self.num_views):
            raise AssertionError(f"视图数不一致: got {len(k_embeds)} vs expected {self.num_views}")

        with torch.no_grad():
            self.momentum_update_key_encoders()

        for i, k_embed in enumerate(k_embeds):
            with torch.no_grad():
                k = F.normalize(self.k_projs[i](k_embed), dim=1)

            queue = getattr(self, f"queue_{i}")
            # 正样本
            l_pos = torch.sum(q * k, dim=1, keepdim=True)
            # 负样本：warmup 用 batch 内；否则用队列
            if warmup:
                sim = torch.matmul(q, k.t())
                N = sim.size(0)
                if N > 1:
                    mask = ~torch.eye(N, dtype=torch.bool, device=sim.device)
                    l_neg = sim[mask].view(N, N - 1)
                else:
                    l_neg = torch.matmul(q, queue.clone().detach())
            else:
                l_neg = torch.matmul(q, queue.clone().detach())
            logits = torch.cat([l_pos, l_neg], dim=1) / self.T
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

            if self.debug and (self.global_step == 1 or self.global_step == self.queue_warmup_steps or self.global_step <= 3):
                with torch.no_grad():
                    cos_stats = {
                        "mean": float((q * k).sum(dim=1).mean().item()),
                        "std": float((q * k).sum(dim=1).std(unbiased=False).item()) if q.size(0) > 1 else 0.0,
                        "min": float((q * k).sum(dim=1).min().item()),
                        "max": float((q * k).sum(dim=1).max().item()),
                    }
                    qshape = list(queue.shape)
                    fill_ratio = (0.0 if warmup else float(min(self._filled[i], self.K)) / float(self.K))
                    assert int(targets.sum().item()) == 0, "MoCo targets 应全为 0"
                print(f"[EM.moco][multi][v={i}] step={self.global_step} warmup={warmup} q={list(q.shape)} k={list(k.shape)} logits={list(logits.shape)} queue={qshape} fill_ratio={fill_ratio:.2f} cos={cos_stats}")

            logits_list.append(logits)
            targets_list.append(targets)

            # 更新队列（非 warmup）
            if not warmup:
                self._dequeue_and_enqueue(k, i)
                self._filled[i] = int(min(self.K, int(self._filled[i]) + k.size(0)))

        return logits_list, targets_list