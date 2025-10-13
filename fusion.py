import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionBase(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FusionBase.forward must be implemented by subclasses")

class BasicFusion(FusionBase):
    """
    复现现有行为：add, product, concat -> 4H
    """
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        # e1,e2: [B,H]
        add = e1 + e2                     # [B,H]
        product = e1 * e2                 # [B,H]
        concatenate = torch.cat([e1, e2], dim=1)  # [B,2H]
        feature = torch.cat([add, product, concatenate], dim=1)  # [B,4H]
        return feature

class DotProductAttentionFusion(FusionBase):
    """
    Scaled Dot-Product Attention over 2 tokens (e1,e2)
    输出将两个token的注意力更新后拼接 -> [B,2H]
    """
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        H = hidden_dim
        self.Wq = nn.Linear(H, H, bias=False)
        self.Wk = nn.Linear(H, H, bias=False)
        self.Wv = nn.Linear(H, H, bias=False)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        B, H = e1.size(0), e1.size(1)
        seq = torch.stack([e1, e2], dim=1)            # [B,2,H]
        Q = self.Wq(seq)                               # [B,2,H]
        K = self.Wk(seq)                               # [B,2,H]
        V = self.Wv(seq)                               # [B,2,H]
        attn = torch.matmul(Q, K.transpose(1,2)) / (H ** 0.5)  # [B,2,2]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)                    # [B,2,H]
        out = out.reshape(B, 2*H)                      # [B,2H]
        return out

class AdditiveAttentionFusion(FusionBase):
    """
    Bahdanau 加性注意力：对(e1,e2)打分并加权求和 -> [B,H]
    """
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        H = hidden_dim
        self.W = nn.Linear(H, H)
        self.v = nn.Parameter(torch.empty(H))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.uniform_(self.v, -1.0 / (H ** 0.5), 1.0 / (H ** 0.5))

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        # score_i = v^T tanh(W e_i)
        s1 = torch.tanh(self.W(e1)) @ self.v          # [B]
        s2 = torch.tanh(self.W(e2)) @ self.v          # [B]
        scores = torch.stack([s1, s2], dim=1)         # [B,2]
        alpha = F.softmax(scores, dim=1)              # [B,2]
        stacked = torch.stack([e1, e2], dim=1)        # [B,2,H]
        fused = torch.sum(alpha.unsqueeze(-1) * stacked, dim=1)  # [B,H]
        return fused

class SelfAttentionFusion(FusionBase):
    """
    传统 Transformer 多头自注意力（仅2 tokens）
    输出将两个token的输出拼接 -> [B,2H]
    """
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__(hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        B, H = e1.size(0), e1.size(1)
        x = torch.stack([e1, e2], dim=1)               # [B,2,H]
        attn_out, _ = self.mha(x, x, x)                # [B,2,H]
        x = self.norm1(x + self.dropout(attn_out))     # [B,2,H]
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))      # [B,2,H]
        x = x.reshape(B, 2*H)                           # [B,2H]
        return x

class GATStyleFusion(FusionBase):
    """
    GAT 风格的两节点注意力聚合（不依赖PyG，复刻GAT打分）
    输出拼接两个节点更新后的表示 -> [B,2H]
    """
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__(hidden_dim)
        H = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.W = nn.Linear(H, H * heads, bias=False)
        self.a = nn.Parameter(torch.empty(heads, 2 * H))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        B, H = e1.size(0), e1.size(1)
        X = torch.stack([e1, e2], dim=1)                     # [B,2,H]
        Wh = self.W(X)                                       # [B,2,H*heads]
        Wh = Wh.view(B, 2, self.heads, H)                    # [B,2,heads,H]
        Wh1 = Wh[:,0]                                        # [B,heads,H]
        Wh2 = Wh[:,1]                                        # [B,heads,H]
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        def edge_score(u, v, a):
            cat = torch.cat([u, v], dim=-1)                  # [B,heads,2H]
            return F.leaky_relu((cat * a).sum(dim=-1), 0.2)  # [B,heads]
        e11 = edge_score(Wh1, Wh1, self.a)                   # 自环
        e12 = edge_score(Wh1, Wh2, self.a)
        e21 = edge_score(Wh2, Wh1, self.a)
        e22 = edge_score(Wh2, Wh2, self.a)
        # softmax over neighbors (2 nodes)
        attn1 = torch.stack([e11, e12], dim=-1)              # [B,heads,2]
        attn2 = torch.stack([e21, e22], dim=-1)              # [B,heads,2]
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)
        attn1 = F.dropout(attn1, p=self.dropout, training=self.training)
        attn2 = F.dropout(attn2, p=self.dropout, training=self.training)
        # 聚合
        h1 = attn1[...,0].unsqueeze(-1) * Wh1 + attn1[...,1].unsqueeze(-1) * Wh2   # [B,heads,H]
        h2 = attn2[...,0].unsqueeze(-1) * Wh1 + attn2[...,1].unsqueeze(-1) * Wh2   # [B,heads,H]
        # 合并多头并拼接两个节点
        h1 = h1.reshape(B, self.heads * H)                   # [B,heads*H]
        h2 = h2.reshape(B, self.heads * H)                   # [B,heads*H]
        # 压回到每节点H维以便统一映射
        proj = nn.functional.linear
        # 为避免每次创建参数，使用 reshape 再线性映射时保持一致性：这里简单拼接后交给外层映射
        out = torch.cat([h1, h2], dim=1)                     # [B,2*heads*H]
        return out

class GraphTransformerStyleFusion(FusionBase):
    """
    Graph Transformer 风格（等价于两节点的多头自注意 + 前馈），与 SelfAttentionFusion 类似但独立参数集
    输出拼接两个token -> [B,2H]
    """
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__(hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        B, H = e1.size(0), e1.size(1)
        x = torch.stack([e1, e2], dim=1)               # [B,2,H]
        attn_out, _ = self.mha(x, x, x)                # [B,2,H]
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))      # [B,2,H]
        x = x.reshape(B, 2*H)
        return x

class FusionDecoder(nn.Module):
    """
    将不同策略的输出统一映射到 4H，再接 decoder1 -> 1，保持与现有训练兼容
    """
    def __init__(self, strategy: FusionBase, hidden_dim: int, decoder1_dim: int, out4h: bool = True):
        super().__init__()
        self.strategy = strategy
        self.hidden_dim = hidden_dim
        H = hidden_dim
        self._out4h = out4h
        # 映射到4H：默认 Identity，首次前传若维度不符再替换为 Linear
        self.proj4h: nn.Module = nn.Identity()
        self.fc1 = nn.Linear(4 * H, decoder1_dim)
        self.fc2 = nn.Linear(decoder1_dim, 1)

    def _ensure_proj(self, feat: torch.Tensor):
        H = self.hidden_dim
        in_dim = feat.size(1)
        # 若当前为 Identity 且维度不匹配，则替换为线性映射
        if isinstance(self.proj4h, nn.Identity) and in_dim != 4 * H:
            self.proj4h = nn.Linear(in_dim, 4 * H).to(feat.device)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor):
        feat = self.strategy(e1, e2)        # [B, D*]
        self._ensure_proj(feat)
        fused4h = self.proj4h(feat)         # [B,4H]
        log1 = F.relu(self.fc1(fused4h))    # [B,decoder1]
        log = self.fc2(log1)                # [B,1]
        return log, log1

def build_fusion_decoder(args, hidden_dim: int) -> FusionDecoder:
    """
    工厂：根据 args.fusion_type 构建 FusionDecoder
    可选：basic、dot、additive、self_attn、gat_fusion、gt_fusion
    """
    fusion_type = getattr(args, "fusion_type", "basic")
    heads = getattr(args, "fusion_heads", 4)
    dropout = getattr(args, "dropout", 0.1)
    decoder1 = getattr(args, "decoder1", 512)

    if fusion_type == "basic":
        strat = BasicFusion(hidden_dim)
    elif fusion_type == "dot":
        strat = DotProductAttentionFusion(hidden_dim)
    elif fusion_type == "additive":
        strat = AdditiveAttentionFusion(hidden_dim)
    elif fusion_type == "self_attn":
        strat = SelfAttentionFusion(hidden_dim, heads=heads, dropout=dropout)
    elif fusion_type == "gat_fusion":
        strat = GATStyleFusion(hidden_dim, heads=heads, dropout=dropout)
    elif fusion_type == "gt_fusion":
        strat = GraphTransformerStyleFusion(hidden_dim, heads=heads, dropout=dropout)
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")

    return FusionDecoder(strategy=strat, hidden_dim=hidden_dim, decoder1_dim=decoder1)