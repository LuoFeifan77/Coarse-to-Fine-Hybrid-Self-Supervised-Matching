import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Iterable
from utils.registry import NETWORK_REGISTRY


@NETWORK_REGISTRY.register()
class Combination(nn.Module):  #
    '''
    A mod combination the bases of polynomial filters.
    Args:
        channels (int): number of feature channels.
        orders_basis (int): number of bases to combine.
        sole (bool): whether or not use the same filter for all output channels.
    '''
    def __init__(self, orders_basis: int = 8, output_channels: int =6, learn_coef: bool=True):
        super().__init__()
        self.learn_coef = learn_coef

        if self.learn_coef:
            self.comb_weight = nn.Parameter(torch.Tensor(1, output_channels, orders_basis), requires_grad=True)
            
        else:
            self.comb_weight = nn.Parameter(torch.Tensor(1, output_channels, orders_basis), requires_grad=False)

        nn.init.xavier_uniform_(self.comb_weight) #

    def forward(self, basis_x : Tensor, basis_y : Tensor):
        '''
        x: node features filtered by bases, of shape (number of nodes, depth, channels).
        '''

        gs_x = torch.bmm(self.comb_weight, basis_x)  # 1*O*K
        gs_y = torch.bmm(self.comb_weight, basis_y)

        return gs_x, gs_y



