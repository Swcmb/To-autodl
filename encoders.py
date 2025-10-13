"""编码器实验.md"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from fusion import build_fusion_decoder
import importlib.util, os

# 统一的读出与判别器（对比学习），对齐 CSGLMD 行为
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits

# 统一的编码器基类与注册机制
class BaseEncoder(nn.Module):
    def __init__(self, args):
        super(BaseEncoder, self).__init__()
        self.args = args
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

    def forward(self, data_o, data_a, idx):
        raise NotImplementedError

class EncoderRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def wrapper(encoder_cls):
            cls._registry[name] = encoder_cls
            return encoder_cls
        return wrapper

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

# 辅助：根据任务类型抽取实体节点嵌入（对齐 CSGLMD 的偏移约定）（合并为单一定义）
def extract_entities(args, x2_o, idx):
    if args.task_type == 'LDA':
        entity1 = x2_o[idx[0]]
        entity2 = x2_o[idx[1] + 240]
    elif args.task_type == 'MDA':
        entity1 = x2_o[idx[0] + 645]
        entity2 = x2_o[idx[1] + 240]
    elif args.task_type == 'LMI':
        entity1 = x2_o[idx[0]]
        entity2 = x2_o[idx[1] + 645]
    else:
        entity1 = x2_o[idx[0]]
        entity2 = x2_o[idx[1]]
    return entity1, entity2
# 交互解码改为可插拔融合模块（见 EM/fusion.py）

# 通用：构造对抗项 logits（与 CSGLMD 接近）
def adversarial_logits(x2_o, x2_o_a, hidden_dim):
    mlp = nn.Linear(hidden_dim, hidden_dim).to(x2_o.device)
    sc_1 = mlp(x2_o).sum(1).unsqueeze(0)
    sc_2 = mlp(x2_o_a).sum(1).unsqueeze(0)
    logits = torch.cat((sc_1, sc_2), 1)
    return logits

# GAT 编码器
@EncoderRegistry.register("gat")
class GATEncoder(BaseEncoder):
    def __init__(self, args):
        super(GATEncoder, self).__init__(args)
        in_dim = args.dimensions
        hidden1 = getattr(args, 'hidden1', max(1, in_dim // 2))
        hidden2 = getattr(args, 'hidden2', max(1, in_dim // 4))
        heads = getattr(args, 'gat_heads', 4)
        dropout = getattr(args, 'dropout', 0.1)
        decoder1 = getattr(args, 'decoder1', 512)

        self.gat1 = GATConv(in_dim, hidden1, heads=heads, concat=True, dropout=dropout)
        self.prelu1 = nn.PReLU(hidden1 * heads)
        self.gat2 = GATConv(hidden1 * heads, hidden2, heads=1, concat=False, dropout=dropout)
        self.prelu2 = nn.PReLU(hidden2)

        self.mlp1 = nn.Linear(hidden2, hidden2)
        self.disc = Discriminator(hidden2)
        self.fusion = build_fusion_decoder(self.args, hidden2)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x1 = self.prelu1(self.gat1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gat2(x1, edge_index)
        x2 = self.prelu2(x2)
        return x2

    def forward(self, data_o, data_a, idx):
        x_o, edge_index = data_o.x, data_o.edge_index
        x_a = data_a.x

        x2_o = self.encode(x_o, edge_index)
        x2_o_a = self.encode(x_a, edge_index)

        h_os = self.read(x2_o)
        h_os = self.sigm(h_os)
        h_os = self.mlp1(h_os)

        h_os_a = self.read(x2_o_a)
        h_os_a = self.sigm(h_os_a)
        h_os_a = self.mlp1(h_os_a)

        ret_os = self.disc(h_os, x2_o, x2_o_a)
        ret_os_a = self.disc(h_os_a, x2_o_a, x2_o)

        entity1, entity2 = extract_entities(self.args, x2_o, idx)
        log, log1 = self.fusion(entity1, entity2)

        logits = adversarial_logits(x2_o, x2_o_a, x2_o.shape[1])

        return log, ret_os, ret_os_a, x2_o, logits, log1

# Graph Transformer 编码器
@EncoderRegistry.register("gt")
class GTEncoder(BaseEncoder):
    def __init__(self, args):
        super(GTEncoder, self).__init__(args)
        in_dim = args.dimensions
        hidden1 = getattr(args, 'hidden1', max(1, in_dim // 2))
        hidden2 = getattr(args, 'hidden2', max(1, in_dim // 4))
        heads = getattr(args, 'gt_heads', 4)
        dropout = getattr(args, 'dropout', 0.1)
        decoder1 = getattr(args, 'decoder1', 512)

        self.gt1 = TransformerConv(in_dim, hidden1, heads=heads, concat=True, dropout=dropout)
        self.prelu1 = nn.PReLU(hidden1 * heads)
        self.gt2 = TransformerConv(hidden1 * heads, hidden2, heads=1, concat=False, dropout=dropout)
        self.prelu2 = nn.PReLU(hidden2)

        self.mlp1 = nn.Linear(hidden2, hidden2)
        self.disc = Discriminator(hidden2)
        self.fusion = build_fusion_decoder(self.args, hidden2)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x1 = self.prelu1(self.gt1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gt2(x1, edge_index)
        x2 = self.prelu2(x2)
        return x2

    def forward(self, data_o, data_a, idx):
        x_o, edge_index = data_o.x, data_o.edge_index
        x_a = data_a.x

        x2_o = self.encode(x_o, edge_index)
        x2_o_a = self.encode(x_a, edge_index)

        h_os = self.read(x2_o)
        h_os = self.sigm(h_os)
        h_os = self.mlp1(h_os)

        h_os_a = self.read(x2_o_a)
        h_os_a = self.sigm(h_os_a)
        h_os_a = self.mlp1(h_os_a)

        ret_os = self.disc(h_os, x2_o, x2_o_a)
        ret_os_a = self.disc(h_os_a, x2_o_a, x2_o)

        entity1, entity2 = extract_entities(self.args, x2_o, idx)
        log, log1 = self.fusion(entity1, entity2)

        logits = adversarial_logits(x2_o, x2_o_a, x2_o.shape[1])

        return log, ret_os, ret_os_a, x2_o, logits, log1

# 串联：GAT -> GT
@EncoderRegistry.register("gat_gt_serial")
class GATGTSerial(BaseEncoder):
    def __init__(self, args):
        super(GATGTSerial, self).__init__(args)
        in_dim = args.dimensions
        hidden1 = getattr(args, 'hidden1', max(1, in_dim // 2))
        hidden2 = getattr(args, 'hidden2', max(1, in_dim // 4))
        gat_heads = getattr(args, 'gat_heads', 4)
        gt_heads = getattr(args, 'gt_heads', 4)
        dropout = getattr(args, 'dropout', 0.1)
        decoder1 = getattr(args, 'decoder1', 512)

        self.gat1 = GATConv(in_dim, hidden1, heads=gat_heads, concat=True, dropout=dropout)
        self.prelu_g1 = nn.PReLU(hidden1 * gat_heads)
        self.gt2 = TransformerConv(hidden1 * gat_heads, hidden2, heads=1, concat=False, dropout=dropout)
        self.prelu_t2 = nn.PReLU(hidden2)

        self.mlp1 = nn.Linear(hidden2, hidden2)
        self.disc = Discriminator(hidden2)
        self.fusion = build_fusion_decoder(self.args, hidden2)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x1 = self.prelu_g1(self.gat1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gt2(x1, edge_index)
        x2 = self.prelu_t2(x2)
        return x2

    def forward(self, data_o, data_a, idx):
        x_o, edge_index = data_o.x, data_o.edge_index
        x_a = data_a.x

        x2_o = self.encode(x_o, edge_index)
        x2_o_a = self.encode(x_a, edge_index)

        h_os = self.read(x2_o)
        h_os = self.sigm(h_os)
        h_os = self.mlp1(h_os)

        h_os_a = self.read(x2_o_a)
        h_os_a = self.sigm(h_os_a)
        h_os_a = self.mlp1(h_os_a)

        ret_os = self.disc(h_os, x2_o, x2_o_a)
        ret_os_a = self.disc(h_os_a, x2_o_a, x2_o)

        entity1, entity2 = extract_entities(self.args, x2_o, idx)
        log, log1 = self.fusion(entity1, entity2)

        logits = adversarial_logits(x2_o, x2_o_a, x2_o.shape[1])

        return log, ret_os, ret_os_a, x2_o, logits, log1

# 并联：GAT || GT，然后融合
@EncoderRegistry.register("gat_gt_parallel")
class GATGTParallel(BaseEncoder):
    def __init__(self, args):
        super(GATGTParallel, self).__init__(args)
        in_dim = args.dimensions
        hidden1 = getattr(args, 'hidden1', max(1, in_dim // 2))
        hidden2 = getattr(args, 'hidden2', max(1, in_dim // 4))
        gat_heads = getattr(args, 'gat_heads', 4)
        gt_heads = getattr(args, 'gt_heads', 4)
        dropout = getattr(args, 'dropout', 0.1)
        decoder1 = getattr(args, 'decoder1', 512)

        self.gat1 = GATConv(in_dim, hidden1, heads=gat_heads, concat=True, dropout=dropout)
        self.prelu_g1 = nn.PReLU(hidden1 * gat_heads)
        self.gat2 = GATConv(hidden1 * gat_heads, hidden2, heads=1, concat=False, dropout=dropout)
        self.prelu_g2 = nn.PReLU(hidden2)

        self.gt1 = TransformerConv(in_dim, hidden1, heads=gt_heads, concat=True, dropout=dropout)
        self.prelu_t1 = nn.PReLU(hidden1 * gt_heads)
        self.gt2 = TransformerConv(hidden1 * gt_heads, hidden2, heads=1, concat=False, dropout=dropout)
        self.prelu_t2 = nn.PReLU(hidden2)

        self.fuse = nn.Linear(hidden2 * 2, hidden2)

        self.mlp1 = nn.Linear(hidden2, hidden2)
        self.disc = Discriminator(hidden2)
        self.fusion = build_fusion_decoder(self.args, hidden2)
        self.dropout = dropout

    def branch_gat(self, x, edge_index):
        x1 = self.prelu_g1(self.gat1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gat2(x1, edge_index)
        x2 = self.prelu_g2(x2)
        return x2

    def branch_gt(self, x, edge_index):
        x1 = self.prelu_t1(self.gt1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gt2(x1, edge_index)
        x2 = self.prelu_t2(x2)
        return x2

    def forward(self, data_o, data_a, idx):
        x_o, edge_index = data_o.x, data_o.edge_index
        x_a = data_a.x

        gat_o = self.branch_gat(x_o, edge_index)
        gt_o = self.branch_gt(x_o, edge_index)
        x2_o = self.fuse(torch.cat([gat_o, gt_o], dim=1))

        gat_a = self.branch_gat(x_a, edge_index)
        gt_a = self.branch_gt(x_a, edge_index)
        x2_o_a = self.fuse(torch.cat([gat_a, gt_a], dim=1))

        h_os = self.read(x2_o)
        h_os = self.sigm(h_os)
        h_os = self.mlp1(h_os)

        h_os_a = self.read(x2_o_a)
        h_os_a = self.sigm(h_os_a)
        h_os_a = self.mlp1(h_os_a)

        ret_os = self.disc(h_os, x2_o, x2_o_a)
        ret_os_a = self.disc(h_os_a, x2_o_a, x2_o)

        entity1, entity2 = extract_entities(self.args, x2_o, idx)
        log, log1 = self.fusion(entity1, entity2)

        logits = adversarial_logits(x2_o, x2_o_a, x2_o.shape[1])

        return log, ret_os, ret_os_a, x2_o, logits, log1

# 适配：CSGLMD-main 项目中的编码器（直接委托其实现）
@EncoderRegistry.register("csglmd_main")
class CSGLMDAdapter(BaseEncoder):
    def __init__(self, args):
        super(CSGLMDAdapter, self).__init__(args)
        
        layer_path = os.path.join(os.path.dirname(__file__), '..', 'CSGLMD-main', 'layer.py')
        spec = importlib.util.spec_from_file_location('csglmd_layer_ext', layer_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load module spec from {layer_path}')
        module = importlib.util.module_from_spec(spec)
        # 断言加载器存在以满足类型检查
        assert spec.loader is not None
        spec.loader.exec_module(module)
        ExternalCSGLMD = module.CSGLMD
        self.model = ExternalCSGLMD(feature=args.dimensions, hidden1=args.hidden1, hidden2=args.hidden2, decoder1=args.decoder1, dropout=args.dropout)

    def forward(self, data_o, data_a, idx):
        return self.model(data_o, data_a, idx)

# 适配：MGACMDA 项目中的编码器（封装 GraphAttentionEncoder）
@EncoderRegistry.register("mgacmda")
class MGACMDAAdapter(BaseEncoder):
    def __init__(self, args):
        super(MGACMDAAdapter, self).__init__(args)
        in_dim = args.dimensions
        hidden1 = getattr(args, 'hidden1', max(1, in_dim // 2))
        hidden2 = getattr(args, 'hidden2', max(1, in_dim // 4))
        num_heads = getattr(args, 'gat_heads', 4)
        self.encoder = _InternalGraphAttentionEncoder(in_dim=in_dim, hidden_dims=[hidden1, hidden2], num_heads=num_heads, fc=None)

        self.mlp1 = nn.Linear(hidden2, hidden2)
        self.disc = Discriminator(hidden2)
        self.decoder = InteractionDecoder(hidden2, getattr(args, 'decoder1', 512))
        self.dropout = getattr(args, 'dropout', 0.1)

    def edge_index_to_dense_adj(self, edge_index, num_nodes):
        values = torch.ones(edge_index.size(1), device=edge_index.device)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()
        return adj.to_dense()

    def forward(self, data_o, data_a, idx):
        x_o, edge_index = data_o.x, data_o.edge_index
        x_a = data_a.x
        n = x_o.size(0)

        A_o = self.edge_index_to_dense_adj(edge_index, n)
        A_a = A_o

        x2_o = self.encoder(A_o, x_o)
        x2_o_a = self.encoder(A_a, x_a)

        h_os = self.read(x2_o); h_os = self.sigm(h_os); h_os = self.mlp1(h_os)
        h_os_a = self.read(x2_o_a); h_os_a = self.sigm(h_os_a); h_os_a = self.mlp1(h_os_a)

        ret_os = self.disc(h_os, x2_o, x2_o_a)
        ret_os_a = self.disc(h_os_a, x2_o_a, x2_o)

        entity1, entity2 = extract_entities(self.args, x2_o, idx)
        log, log1 = self.decoder(entity1, entity2)

        logits = adversarial_logits(x2_o, x2_o_a, x2_o.shape[1])
        return log, ret_os, ret_os_a, x2_o, logits, log1

# 内置版 GraphAttentionEncoder（参考 MGACMDA/gate.py）
class _GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = self.W(h)
        N = Wh.size(0)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        e = F.leaky_relu(torch.matmul(all_combinations_matrix, self.a).squeeze(1), 0.2).view(N, N)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, 0.1, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class _MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            _GraphAttentionLayer(in_features, out_features) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(num_heads * out_features, out_features, bias=False)

    def forward(self, h, adj):
        head_outputs = [attn_head(h, adj) for attn_head in self.attention_heads]
        h_prime = torch.cat(head_outputs, dim=1)
        h_prime = self.linear(h_prime)
        return F.elu(h_prime)

class _GraphAttentionLayerWithResidual(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4):
        super().__init__()
        self.attention = _MultiHeadGraphAttentionLayer(in_features, out_features, num_heads)
        self.residual = nn.Linear(in_features, out_features, bias=False)

    def forward(self, h, adj):
        h_prime = self.attention(h, adj)
        res = self.residual(h)
        return F.relu(h_prime + res)

class _InternalGraphAttentionEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, num_heads=3, fc=None):
        super().__init__()
        self.fc = fc
        self.layers = nn.ModuleList()
        prev_dim = in_dim
        for out_dim in hidden_dims:
            self.layers.append(_GraphAttentionLayerWithResidual(prev_dim, out_dim, num_heads))
            prev_dim = out_dim

    def forward(self, A, X):
        h = X
        for layer in self.layers:
            h = layer(h, A)
            h = F.relu(h)
            h = F.dropout(h, p=0.4, training=self.training)
        if self.fc:
            h = self.fc(h)
        embedding = h.clone()
        return embedding