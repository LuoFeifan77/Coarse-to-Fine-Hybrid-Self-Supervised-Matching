import torch.nn as nn
from utils.registry import NETWORK_REGISTRY

import torch
import torch.nn as nn

# mask functional maps的操作
class LearnableNonDiagMask(nn.Module):
    def __init__(self, size=10):
        super(LearnableNonDiagMask, self).__init__()
        self.size = size
        
        # 1. 定义可学习的权重矩阵 W
        # 初始化可以采用正态分布或 Xavier 初始化
        self.learnable_weights = nn.Parameter(torch.randn(size, size))
        
        # 2. 注册一个固定不更新的 Buffer：单位矩阵 I
        self.register_buffer('identity', torch.eye(size))
        # 非对角线掩码 I_bar
        self.register_buffer('non_diag_mask', 1.0 - torch.eye(size))

    def forward(self, A):
        # A 是输入的 10x10 方阵
        
        # 提取主对角线部分 (保持原始 A 的对角线不变)
        diag_part = self.identity * A
        
        # 提取并学习非对角线部分
        # 注意：这里我们可以让 W 直接作用，或者让 W 与原 A 结合
        # 如果是“对非对角线元素进行可学习掩码”，通常是用 W 替换原非对角线
        non_diag_part = self.non_diag_mask * self.learnable_weights
        
        # 合并
        out = diag_part + non_diag_part
        return out





















@NETWORK_REGISTRY.register()
class MLP_filter(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        linear_module=None,
        non_linear_module=None,
        device="cpu",
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shape = (output_dim, input_dim)
        self.device = device

        # Linear Module
        self.linear_module = linear_module
        if self.linear_module is None:
            self.linear_module = nn.Linear(input_dim, output_dim, bias=False).to(
                self.device
            )

        # Non-linear MLP Module
        self.nonlinear_module = non_linear_module

        if self.nonlinear_module is None:
            self.nonlinear_module = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                depth=2,
                width=200, #特征向量的个数
                # act=nn.LeakyReLU(),  # 我的激活函数，应该考虑sigmod！
                act = nn.Sigmoid(), #将数值控制[0,1]直接，然后将特征值去掉，不需要特征值
            ).to(self.device)
        # Apply small scaling to MLP output for initialization
        self.mlp_scale = 0.01
        self._reset_parameters()

    def forward(self, x):

        x = x[:, : self.input_dim]
        fmap = self.linear_module(x)
        t = self.mlp_scale * self.nonlinear_module(x) # 将基函数放进去？
        x_out = fmap + t

        return x_out.squeeze()

    def _reset_parameters(self):
        """Initialize the model parameters using Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



class MLP(nn.Module):
    """
    A simple MLP (Multi-Layer Perceptron) module.

    Parameters
    ----------
    input_dim : int
        The dimension of the input data.
    output_dim : int
        The dimension of the output data.
    depth : int
        The number of layers in the MLP.
    width : int
        The width of each layer in the MLP.
    act : torch.nn.Module
        The activation function to be used in the MLP.
    """

    def __init__(
        self, input_dim, output_dim, depth=4, width=128, act=nn.LeakyReLU(), bias=True
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev_dim, width, bias=bias))
            layers.append(act)  # Add activation after each layer
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):  #我们也用简单的loss看看！
        """Forward pass through the MLP."""
        return self.mlp(x)
    
