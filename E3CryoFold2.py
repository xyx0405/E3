import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from cryo_module.AutoEM import CryoProcesser
from argparse import ArgumentParser # 重复导入，但保持原样
from cryo_module.Cryo_model import Cryo_Model
from chroma.layers.structure.conditioners import ShapeConditioner, SubsequenceConditioner, ComposedConditioner
from chroma import Chroma
from chroma.constants.sequence import AA20 as alphabet
import numpy as np
from chroma.data.protein import Protein


class E3CryoFold2(pl.LightningModule):
    def __init__(self, args=None, model_weights=None, struc_model_weight=None, **model_kwargs):
        """初始化 E3CryoFold2，加载冷冻电镜处理器和各种神经网络模型"""
        super().__init__()

        self.save_hyperparameters() # 自动保存输入参数
        self.args = args
        self.cryo_processer = CryoProcesser(args) # 负责密度图解析和 Cα 候选点提取
        model_weights = torch.load(model_weights) # 加载联合权重字典
        
        # 加载用于从密度图提取特征的 Cryo Model
        self.cryo_model = Cryo_Model().models
        self.cryo_model.load_state_dict(model_weights['cryo_model'])
        print('Cryo Model loaded successfully!')
        
        # 加载 Chroma 扩散模型（骨架网络 + 设计网络）
        self.model = Chroma(weights_backbone=model_weights['backbone_model'], weights_design=model_weights['design_model'])
        print('All models loaded successfully')

        # 如果是 denovo（从头预测）协议，加载额外的结构先验模型
        if args.protocol == 'denovo':
            from structure_model import Model
            struc_model_weight = torch.load(struc_model_weight)
            self.structure_model = Model(init_kwargs=struc_model_weight['init_kwargs'])
            self.structure_model.load_state_dict(struc_model_weight['model_state_dict'], strict=False)
            self.diffusion = self.structure_model.encoder.noise_perturb

    def infer(self, t=0.1, spatial_conditioner=False, sequence_conditioner = False, N=None, chain_lengths=[100]):
        """执行推理流程：密度图特征提取 -> 初始结构生成 -> 扩散采样精修"""
        # A. 提取空间特征：得到骨架、Cα 和氨基酸的候选概率分布
        self.cryo_processer.get_spatial_feature(self.cryo_model['BB_model'], self.cryo_model['CA_model'], self.cryo_model['AA_model'])
        spatial_features = self.cryo_processer.CA_cands.astype(np.float32)

        # B. 根据协议生成初始结构坐标 X, 链编码 C, 序列 S
        if self.args.protocol == 'pre_align' or self.args.protocol == 'seq_free':
            # 预对齐模式：基于密度图片段映射得到初始位置
            X, C, S = self.cryo_processer.mapping()
            X = torch.from_numpy(self.generate_backbone_atoms(X))[None,:].to(self.args.device)
        elif self.args.protocol == 'denovo':
            # 从头预测模式：由序列先验模型生成初始 backbone
            C, S = self.cryo_processer.get_seq()
            X = torch.from_numpy(self.generate_backbone_atoms(spatial_features)).to(self.args.device)
            X = self.structure_model.infer(X[None, :], C, S)

        # C. 处理序列和 Protein 对象封装
        if self.args.protocol != 'seq_free_denovo':
            # 将单字母序列转换为对应的 Token 索引
            S = torch.LongTensor([alphabet.index(AA) for AA in S])
            S, C = S.long()[None,:].to(self.args.device), C.long()[None,:].to(self.args.device)
            protein = Protein.from_XCS(X, C, S)
            design_method = None
        else:
            # 无序列模式：由模型在采样过程中设计序列
            protein = None
            t = 1.0 # 从纯噪声开始
            chain_lengths = [500] * (spatial_features.shape[0]//500) + [spatial_features.shape[0]%500]
            design_method = 'potts'

        spatial_features = torch.from_numpy(spatial_features)
        S_length = S.shape[1] if 'S' in locals() else spatial_features.shape[0]

        # D. 构造约束条件（Conditioners）来引导结构生成
        conditioners = []
        if spatial_conditioner:
            # 空间约束：引导生成的原子靠近电镜图密度中心
            spatial_constrain = ShapeConditioner(
                    spatial_features,
                    self.model.backbone_network.noise_schedule,
                    autoscale_num_residues=S_length).to(self.args.device)
            conditioners.append(spatial_constrain)

        if sequence_conditioner:
            # 序列约束：保持序列特征的一致性
            seq_constrain = SubsequenceConditioner(
                self.model.design_network,
                protein=protein
            )
            conditioners.append(seq_constrain)
        
        # 组合多个约束条件
        if conditioners:
            conditioners = ComposedConditioner(conditioners)
        else:
            return protein
        
        # 确定采样步数 N
        if N is None:
            N = int(t * 500 )if t<0.5 else 500
        
        # E. 调用 Chroma 扩散引擎进行采样，返回精修后的 Protein 对象
        protein = self.model.sample(tspan=(t, 0.001), protein_init=protein, steps=N, \
                                conditioner=conditioners, inverse_temperature=10.0, \
                                langevin_factor=2, langevin_isothermal=False, initialize_noise=False,
                                chain_lengths=chain_lengths, design_method=design_method
                                )
        return protein

    def generate_backbone_atoms(self, ca_coords):
        """基于 Cα 坐标，利用启发式几何算法推断并补全 N, C, O 骨架原子"""
        c_atoms = []
        n_atoms = []
        o_atoms = []

        for i in range(len(ca_coords)):
            ca = np.array(ca_coords[i])
            # 推断 C 原子位置
            if i < len(ca_coords)-1:
                ca_next = np.array(ca_coords[i + 1])
                c_atom = ca + 1.53 * (ca_next - ca) / np.linalg.norm(ca_next - ca)
            else:
                c_atom = ca + np.array([-1.5, 0, 0])

            # 推断 N 原子位置
            if i > 0:
                ca_prev = np.array(ca_coords[i - 1])
                n_atom = ca + 1.32 * (ca - ca_prev) / np.linalg.norm(ca - ca_prev)
            else:
                n_atom = ca + np.array([0, 1.32, 0])  # 假设第一个残基的初始位置

            # 推断 O 原子坐标
            o_atom = c_atom + np.array([0, 0, 1.24])  # 假设 O 始终在 C 的上方

            c_atoms.append(c_atom)
            n_atoms.append(n_atom)
            o_atoms.append(o_atom)
            
        # 堆叠原子坐标并重组成 (L, 4, 3) 的格式，顺序为 [C, CA, N, O]
        c_atoms, n_atoms, o_atoms = map(np.stack, [c_atoms, n_atoms, o_atoms])
        backbone_atoms = np.stack([c_atoms, ca_coords, n_atoms, o_atoms], axis=1)

        return backbone_atoms.astype(np.float32)


    @staticmethod
    def add_model_specific_args(parent_parser):
        """定义模型特有的命令行参数，包括电镜处理的各种评分和阈值"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        # 全原子重建相关参数
        parser.add_argument('--pulchra', default=True, help='whether to run pulchra for all_atom construction')
        parser.add_argument('--pulchra_path',type=str, default='cryo_module/pulchra304/src/pulchra', help='directory of pulchra')

        # 冷冻电镜前处理参数（聚类、NMS、评分阈值等）
        parser.add_argument('--seed', type=int, default=2022, help='set as default')
        parser.add_argument('--cluster_eps', type=int, default=10, help='set as default')
        parser.add_argument('--resolution', type=float, help='resolution of EM map, required when run_phenix_real_space_refine is open')
        parser.add_argument('--cluster_min_points', type=int, default=10, help='set as default')
        parser.add_argument('--nms_radius', type=int, default=9, help='set as default')
        parser.add_argument('--CA_score_thrh', type=float, default=0.35, help='set as default')
        parser.add_argument('--frags_len', type=int, default=150, help='set as defaul')
        parser.add_argument('--n_hop', type=int, default=6, help='set as default')
        parser.add_argument('--neigh_mat_thrh', type=float, default=0.7, help='set as default')
        parser.add_argument('--mul_proc_num', type=int, default=30, help='set as default')
        parser.add_argument('--score_thrh', type=float, default=2, help='set as default')
        parser.add_argument('--gap_len', type=int, default=3, help='set as default')
        parser.add_argument('--struct_len', type=int, default=5, help='set as default')
        parser.add_argument('--afdb_allow_seq_id', type=float, default=0.6, help='set as default')
        return parser