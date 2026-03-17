import torch
import torch.nn as nn
import esm
from chroma.models.graph_backbone import GraphBackbone

# 加载预训练的 ESM-2 模型（3500万参数版本，适合显存较小的环境）
# alphabet 用于将氨基酸序列转换为模型可识别的索引（Token）
esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()

class Model(nn.Module):
    def __init__(self, encoder_dim=512, dropout=0.1, init_kwargs=None):
        """
        初始化模型架构
        :param encoder_dim: 编码器隐藏层维度
        :param dropout: 随机失活率
        :param init_kwargs: 传递给 GraphBackbone 的配置参数（字典格式）
        """
        super().__init__()
        self.dropout = dropout
        self.esm = esm_model  # 嵌入预训练的 ESM 模型
        
        # 线性层：将 ESM 的 480 维特征映射到指定的 encoder_dim (512)
        self.encode_head = nn.Linear(480, encoder_dim)

        # Chroma 的图神经网络骨架，用于处理蛋白质结构特征
        self.encoder = GraphBackbone(**init_kwargs)
        
        # 序列位置编码：支持最大 30000 个残基的位置信息
        self.position_emb = nn.Embedding(num_embeddings=30000, embedding_dim=encoder_dim)
        
        # 链编码：用于区分不同的蛋白质链（如 A 链、B 链），padding_idx=0 表示 0 是填充值
        self.chain_embed = nn.Embedding(num_embeddings=1000, embedding_dim=encoder_dim, padding_idx=0)

    def infer(self, X, chain_encoding, seqs=None, t=1.0):
        """
        推理函数：根据初始坐标和序列生成预测结构
        :param X: 初始坐标张量，形状通常为 [Batch, Residues, Atoms, 3]
        :param chain_encoding: 链标识编码
        :param seqs: 氨基酸序列列表
        :param t: 扩散模型的时间步参数（控制噪声水平或精修强度）
        """
        device = X.device
        
        # 将 t 转换为 Tensor 并移动到对应设备
        if isinstance(t, float):
            t = torch.FloatTensor([t]).to(device)
            
        # 生成序列的相对位置索引 [0, 1, 2, ...]
        seq_pos = torch.arange(chain_encoding.shape[0]).to(device)[None, :]
        
        # 确保链编码是长整型，并增加 Batch 维度
        chain_encoding = chain_encoding.long()[None,:].to(device)

        # 维度补齐逻辑：如果链编码长度与输入坐标 X 不一致（通常是 X 缺失部分原子）
        if chain_encoding.shape[1] != X.shape[1]:
            # 创建全零的填充张量，并以 X 的均值作为初始占位符（避免零值导致计算异常）
            imputed_X = torch.zeros(list(chain_encoding.shape) + [4, 3]).to(device) + X.mean(1, keepdim=True)
            # 将已有的 X 坐标覆盖到填充张量的前部
            imputed_X[:, :chain_encoding.shape[1]] = X[:, :chain_encoding.shape[1]]
        else:
            imputed_X = X

        # 推理阶段关闭梯度计算，节省显存
        with torch.no_grad():
            # 1. 提取序列特征（结合 ESM、位置和链编码）
            h_V = self.get_h_V(seq_pos=seq_pos, chain_encoding=chain_encoding, seqs=seqs)
            
            # 2. 将坐标和序列特征喂入图神经网络骨架进行结构预测
            X_pred = self.encoder(imputed_X, chain_encoding, t, h_V, if_infer=True)
            
        return X_pred
    
    def get_h_V(self, seq_pos, chain_encoding, seqs=None):
        """
        特征提取函数：获取残基级别的节点特征 h_V
        """
        device = chain_encoding.device
        length = len(seqs)
        
        # 使用 ESM 的 alphabet 对序列进行编码并转为索引 Tensor
        seqs_ids = torch.LongTensor(alphabet.encode(''.join(seqs))).to(device)[None,:]

        h_Vs = []
        # 分段处理超长序列（分段长度 3500）：防止一次性输入 ESM 导致显存溢出（OOM）
        for i in range(0, length, 3500):
            start, end = i, i+3500
            local_len = min(3500, length-start)
            
            # 滑动窗口边界处理：确保最后一段至少有 3500 长度（如果序列够长的话）
            if i + 3500 > length:
                start = max(length-3500, 0)

            # 调用 ESM 获取第 12 层的表示特征（Representations）
            results = self.esm(seqs_ids[:, start: end], repr_layers=[12])
            h_V_part = results['representations'][12] 
            
            # 仅截取当前窗口对应的有效长度部分
            h_Vs.append(h_V_part[:, -local_len:])

        # 将分段提取的特征拼接回完整序列
        h_V = torch.cat(h_Vs, dim=1)
        
        # 1. 将 ESM 特征降维/升维到 encoder_dim
        h_V = self.encode_head(h_V)
        
        # 2. 融合链信息编码
        # 3. 融合位置信息编码（加法融合是 Transformer 类模型的常用做法）
        h_V = h_V + self.chain_embed(chain_encoding) + self.position_emb(seq_pos)

        return h_V