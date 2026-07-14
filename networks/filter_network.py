import torch
import hashlib
import numpy as np
import scipy
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import gamma as gammaFunc
from torch import Tensor

from utils.registry import NETWORK_REGISTRY

EPS=1e-7

def get_norm_weight_formula_torch(a, b, K):   # 
    ret = []

    def gammaFunc_torch(x):
        return torch.exp(torch.special.gammaln(x))

    for i in range(K+1):
        term1 = torch.pow(2, a+b+1)/(2*i+a+b+1)
        term2 = gammaFunc_torch(i+a+1)/gammaFunc_torch(i+a+b+1)
        term3 = gammaFunc_torch(i+b+1)/gammaFunc_torch(torch.tensor(i+1))
        ret.append(torch.sqrt(term1*term2*term3))
    return torch.stack(ret)


def get_norm_weight_formula_np(a, b, K):  #
    ret = []
    for i in range(K+1):
        term1 = np.power(2, a+b+1)/(2*i+a+b+1)
        term2 = gammaFunc(i+a+1)/gammaFunc(i+a+b+1)
        term3 = gammaFunc(i+b+1)/gammaFunc(i+1)
        ret.append(np.sqrt(term1*term2*term3))
    ret = np.asarray(ret, dtype=np.float32)
    # print(ret, file=sys.stderr)
    return ret


####-------------------------Learning filters------------------------###########
from functools import partial

# 这个设计暂时不考虑
@NETWORK_REGISTRY.register()
class Learned_Fliters(torch.nn.Module):
    def __init__(self, 
                Poly_Type : str = 'JacobiConv',
                orders: int = 8, 
                alpha: float = 1.0, 
                learnable_bases: bool = True, 
                learnable_alphas: bool = True,
                normalized_bases: bool = True,
                **kwargs):
        super().__init__()
        
        self.Poly_Type = Poly_Type
        self.orders = orders  # default 30
        self.basealpha = alpha
        self._normalized_bases = normalized_bases
        self.learnable_alphas = learnable_alphas
        self.learnable_bases = learnable_bases
        
        #1 choose conv
        if self.Poly_Type == 'JacobiConv':
            conv_fn =  partial(JacobiConv, **kwargs) # 
            # conv_fn =  partial(JacobiConv, [jacobi_a, jacobi_b])
            jacobi_a=kwargs.get('a', 1.0),  # 
            jacobi_b=kwargs.get('b', 1.0),
            # whether learn a, b or not
            if learnable_bases:
                self._a = nn.Parameter(torch.tensor(jacobi_a), requires_grad=True)   #
                self._b = nn.Parameter(torch.tensor(jacobi_b), requires_grad=True)
            else:
                self._a = jacobi_a[0]
                self._b = jacobi_b[0]

        if self.Poly_Type == 'PowerConv':
            conv_fn = PowerConv
        if self.Poly_Type == 'LegendreConv':
            conv_fn = LegendreConv
        if self.Poly_Type == 'ChebyshevConv':
            conv_fn = ChebyshevConv

        self.conv_fn = conv_fn

        #2 whether learn alpha or not and apply restriction on alphas
        if self.learnable_alphas:
            self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(min(1 / alpha, 1))),
                         requires_grad=True) for i in range(self.orders)]) # 
        else:
            self.alphas = [torch.tensor(float(min(1 / alpha, 1))) for i in range(self.orders)]  # 
            

    def forward(self, evals_x :Tensor, evals_y : Tensor):   # 

        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas] if self.learnable_alphas else self.alphas

        num_eig_x = len(evals_x[0])  # 
        num_eig_y = len(evals_y[0])

        self.device = evals_x.device
        #1 yield sign == 1
        x= torch.ones(1, num_eig_x, device = self.device)
        y= torch.ones(1, num_eig_y, device = self.device)

        #2 learn a, b
        if self.learnable_bases:
            a = self._a if isinstance(self._a, float) else torch.clamp(self._a, min=-1.0+EPS, max=10.0)  #
            b = self._b if isinstance(self._b, float) else torch.clamp(self._b, min=-1.0+EPS, max=10.0)

            xs_x = [self.conv_fn(0, [x], evals_x, alphas, a=a, b=b)]  # 
            xs_y = [self.conv_fn(0, [y], evals_y, alphas, a=a, b=b)]  # 

            for L in range(1, self.orders):  # 

                tx = self.conv_fn(L, xs_x, evals_x, alphas, a=a, b=b)   # 
                ty = self.conv_fn(L, xs_y, evals_y, alphas, a=a, b=b) 

                xs_x.append(tx)
                xs_y.append(ty)

        else: 
            xs_x = [self.conv_fn(0, [x], evals_x, alphas)]  # 
            xs_y = [self.conv_fn(0, [y], evals_y, alphas)]  # 

            for L in range(1, self.orders):  # 

                tx = self.conv_fn(L, xs_x, evals_x, alphas)   # 
                ty = self.conv_fn(L, xs_y, evals_y, alphas) 

                xs_x.append(tx)
                xs_y.append(ty)

        xs_x = [x.unsqueeze(1) for x in xs_x]
        xs_y = [y.unsqueeze(1) for y in xs_y]

        basis_x = torch.cat(xs_x, dim=1)   #
        basis_y = torch.cat(xs_y, dim=1)   # 

        #3 normalized base
        if self.Poly_Type =='JacobiConv': 
            if self._normalized_bases and self.learnable_bases:
                norms = get_norm_weight_formula_torch(self._a, self._b, self.orders-1).reshape(1, self.orders, 1)
                basis_x = basis_x / (norms + EPS)
                basis_y = basis_y / (norms + EPS)

            if self._normalized_bases and not self.learnable_bases:
                norms = get_norm_weight_formula_np(self._a, self._b, self.orders-1).reshape(1, self.orders, 1)
                norms = torch.from_numpy(norms)
                norms = norms.to(self.device)
                basis_x = basis_x / (norms + EPS)
                basis_y = basis_y / (norms + EPS)

        return basis_x, basis_y



#----------------FilterConvolution--------------------#
def PowerConv(L, xs, adj, alphas):
    '''
    Monomial bases.
    '''
    if L == 0: return xs[0]
    return alphas[L] * (adj * xs[-1])


def LegendreConv(L, xs, adj, alphas):
    '''
    Legendre bases. Please refer to our paper for the form of the bases.
    '''
    adj = 2*adj/adj[-1]-1  #

    if L == 0: return xs[0]
    nx = (alphas[L - 1] * (2 - 1 / L)) * (adj * xs[-1])
    if L > 1:
        nx -= (alphas[L - 1] * alphas[L - 2] * (1 - 1 / L)) * xs[-2]
    return nx


def ChebyshevConv(L, xs, adj, alphas):
    '''
    Chebyshev Bases. Please refer to our paper for the form of the bases.
    '''
    adj = 2*adj/adj[-1]-1  # 
    
    if L == 0: return xs[0]
    nx = (2 * alphas[L - 1]) * (adj * xs[-1])
    if L > 1:
        nx -= (alphas[L - 1] * alphas[L - 2]) * xs[-2]   #
    return nx


def JacobiConv(L, xs, adj, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):  # 
    '''
    Jacobi Bases. Please refer to our paper for the form of the bases.
    '''
    adj = 2*adj/adj[0][-1]-1  # scale to [-1, 1];

    if L == 0: return xs[0]
    if L == 1:

        coef1 = (a - b) / 2 
        coef1 *= alphas[0]
        coef2 = (a + b + 2) / (r - l)
        coef2 *= alphas[0]
        return coef1 * xs[-1] + coef2 * (adj * xs[-1])  # 

    coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
    coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
    coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
    tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)

    nx = tmp1 * (adj * xs[-1]) - tmp2 * xs[-1] 
    nx -= tmp3 * xs[-2]
    return nx

# Mexican_hat filters
class Mexican_hat(object):
    def __init__(self, lmax, Nf=6, scales=None):

        self.Nf = Nf

        if scales is None:
            lpfactor = 20
            lmin = lmax / lpfactor
            # scales = (4./(3 * lmax)) * np.power(2., np.arange(Nf-2, -1, -1))
            t1 = 1
            t2 = 2

            smin = t1 / lmax
            smax = t2 / lmin
            tt = np.linspace(np.log(smax), np.log(smin), Nf)

            scales = np.exp(tt)  #

        self.gb = [lambda x: (np.exp(-x) * x)]  # high pass filter
        self.gl = [lambda x: (np.exp(-x ** 4))]  # low pass filter

        lminfac = 0.4 * lmin

        self.g = [lambda x: 1.2 * np.exp(-1) * self.gl[0](x / lminfac)]  # 

        for i in range(Nf - 1):
            self.g.append(lambda x, i=i: self.gb[0](scales[i] * x))


    def __call__(self, evals):
        # input:
        #   evals: [K,], pytorch tensor
        # output:
        #   gs: [Nf,K,], pytorch tensor   

        #-----------------------------------#

        gs = np.expand_dims(self.g[0](evals), 0)  # 
        
        for s in range(1, self.Nf):
            gs = np.concatenate((gs, np.expand_dims(self.g[s](evals), 0)), 0)
        # return gs
        return torch.from_numpy(gs.astype(np.float32))


# Mexican_hat filters
class Mexican_hat_MGCN(object): # Tight Frame
    def __init__(self, lmax, Nf=32, scales=None):

        self.Nf = Nf

        if scales is None:
            lpfactor = 20
            lmin = lmax/lpfactor
            # scales = (4./(3 * lmax)) * np.power(2., np.arange(Nf-2, -1, -1))
            t1 =1
            t2 =2
            smin = t1/lmax
            smax = t2/lmin
            tt = np.linspace(np.log(smax*1.15), np.log(smin*0.1), Nf)

            scales = np.exp(tt) # 

        self.gb = [lambda x: (0.443*x**2*np.exp(1-x**2))]  # high pass filter
        self.gl = [lambda x: (1.004*np.exp(-x**3))]  # low pass filter

        lminfac = 0.52 *lmin

        self.g = [lambda x: 1.2*np.exp(-1)*self.gl[0](x/lminfac)]  # 

        for i in range(Nf-1):
            self.g.append(lambda x, i=i: self.gb[0](scales[i] * x))


    def __call__(self, evals):
        # input:
        #   evals: [K,], pytorch tensor
        # output:
        #   gs: [Nf,K,], pytorch tensor   

        #----------------------------------#

        gs = np.expand_dims(self.g[0](evals), 0)  # 扩张维度！
        
        for s in range(1, self.Nf):
            gs = np.concatenate((gs, np.expand_dims(self.g[s](evals), 0)), 0)
        # return gs
        return torch.from_numpy(gs.astype(np.float32))
        # return torch.from_numpy(gs.astype(np.float32), device='cuda')


# Meyer filters
class Meyer(object):
    def __init__(self, lmax, Nf=6, scales=None):

        self.Nf=Nf

        if scales is None:
            scales = (4./(3 * lmax)) * np.power(2., np.arange(Nf-2, -1, -1))

        if len(scales) != Nf - 1:
            raise ValueError('len(scales) should be Nf-1.')

        self.g = [lambda x: kernel(scales[0] * x, 'scaling_function')]

        for i in range(Nf - 1):
            self.g.append(lambda x, i=i: kernel(scales[i] * x, 'wavelet'))

        def kernel(x, kernel_type):
            r"""
            Evaluates Meyer function and scaling function

            * meyer wavelet kernel: supported on [2/3,8/3]
            * meyer scaling function kernel: supported on [0,4/3]
            """

            x = np.asarray(x)

            l1 = 2/3.
            l2 = 4/3.  # 2*l1
            l3 = 8/3.  # 4*l1

            def v(x):
                return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

            r1ind = (x < l1)
            r2ind = (x >= l1) * (x < l2)
            r3ind = (x >= l2) * (x < l3)

            # as we initialize r with zero, computed function will implicitly
            # be zero for all x not in one of the three regions defined above
            r = np.zeros(x.shape)
            if kernel_type == 'scaling_function':
                r[r1ind] = 1
                r[r2ind] = np.cos((np.pi/2) * v(np.abs(x[r2ind])/l1 - 1))
            elif kernel_type == 'wavelet':
                r[r2ind] = np.sin((np.pi/2) * v(np.abs(x[r2ind])/l1 - 1))
                r[r3ind] = np.cos((np.pi/2) * v(np.abs(x[r3ind])/l2 - 1))
            else:
                raise ValueError('Unknown kernel type {}'.format(kernel_type))

            return r


    def __call__(self, evals):
        # input:
        #   evals: [K,], pytorch tensor
        # output: 
        #   gs: [Nf,K,], pytorch tensor
        # evals=evals.numpy()
        gs=np.expand_dims(self.g[0](evals),0)

        for s in range(1, self.Nf):
            gs=np.concatenate((gs,np.expand_dims(self.g[s](evals),0)),0)
        
        return torch.from_numpy(gs.astype(np.float32))
