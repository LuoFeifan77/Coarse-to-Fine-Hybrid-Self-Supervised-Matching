import torch
import torch.nn.functional as F
import numpy as np
import random

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import elas_fmap2pointmap, nn_query, trim_basis, hybrid_fmap2pointmap, fmap2pointmap3D
from utils.geometry_util import get_all_operators
from networks.filter_network import Mexican_hat, Mexican_hat_MGCN, Meyer


def cache_operators(data, cache_dir=None):
    data_x, data_y = data['first'], data['second']
    if 'operators' not in data_x.keys():
        cache_dir = cache_dir or data_x.get('cache_dir', None)
        _, mass, L, evals, evecs, gradX, gradY = get_all_operators(data_x['verts'].cpu(), data_x['faces'].cpu(), k=128,
                                                                    cache_dir=cache_dir)
        
        data_x['operators'] = {'mass': mass, 'L': L, 'evals': evals, 'evecs': evecs, 'gradX': gradX, 'gradY': gradY}
    if 'operators' not in data_y.keys():
        cache_dir = cache_dir or data_y.get('cache_dir', None)
        _, mass, L, evals, evecs, gradX, gradY = get_all_operators(data_y['verts'].cpu(), data_y['faces'].cpu(), k=128,
                                                                    cache_dir=cache_dir)
        data_y['operators'] = {'mass': mass, 'L': L, 'evals': evals, 'evecs': evecs, 'gradX': gradX, 'gradY': gradY}



@MODEL_REGISTRY.register()
class Hybrid_Self_Supervised_Model(BaseModel):
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        self.Nf = opt.get('Nf', 6)
        if self.with_refine > 0:
            opt['is_train'] = True
        super(Hybrid_Self_Supervised_Model, self).__init__(opt)

    def feed_data(self, data):
        cache_dir = self.opt['networks']['feature_extractor'].get('cache_dir', None)
        cache_operators(data, cache_dir=cache_dir)
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # feature extractor for mesh
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'], data=data_x)  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'], data=data_y)  # [B, Ny, C]


        # trim basis
        n_lb = self.opt.get('n_lb', 140)
        n_elas = self.opt.get('n_elas', 60)
        data_x = trim_basis(data_x, n_lb, n_elas)
        data_y = trim_basis(data_y, n_lb, n_elas)

        # ------------------- LB Loss ------------------- #
        if n_lb > 0:
            # get spectral operators
            evals_x = data_x['evals']
            evals_y = data_y['evals']
            evecs_x = data_x['evecs']
            evecs_y = data_y['evecs']
            evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
            evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

        # ------------------- Elas Loss ------------------- #
        if n_elas > 0:
            # get elas spectral operators
            elas_evals_x = torch.abs(data_x['elas_evals'])
            elas_evals_y = torch.abs(data_y['elas_evals'])
            elas_evecs_x = data_x['elas_evecs']
            elas_evecs_y = data_y['elas_evecs']
            elas_evecs_trans_x = data_x['elas_evecs_trans']  # [B, K, Nx]
            elas_evecs_trans_y = data_y['elas_evecs_trans']  # [B, K, Ny]
            elas_mass_x = data_x['elas_mass']
            elas_mass_y = data_y['elas_mass']
            elas_Mk_x = data_x['elas_Mk']
            elas_Mk_y = data_y['elas_Mk']
            elas_invsqrtMk_x = data_x['elas_invsqrtMk']
            elas_invsqrtMk_y = data_y['elas_invsqrtMk']
            elas_sqrtMk_x = data_x['elas_sqrtMk']
            elas_sqrtMk_y = data_y['elas_sqrtMk']
            
        # pointmap computation
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        Pxy = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=False)
        Pyx = self.compute_permutation_matrix(feat_y, feat_x, bidirectional=False)  #在不同的基函数下，也需要这个吗？
        Pyy = self.compute_permutation_matrix(feat_y, feat_y, bidirectional=False)

        Cxy = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x)) #
        Cyx = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y)) #

        elas_Cxy = torch.bmm(elas_evecs_trans_y, torch.bmm(Pyx, elas_evecs_x)) # 得到初始的elas_Cxy
        elas_Cyx = torch.bmm(elas_evecs_trans_x, torch.bmm(Pxy, elas_evecs_y))

         # nn refinement
        Pyx_nn = nn_query(feat_x, feat_y).squeeze() # 得到初始的Pyx
        Cxy_nn = torch.bmm(evecs_trans_y, evecs_x[:, Pyx_nn, :])  
        elas_Cxy_nn = torch.bmm(elas_evecs_trans_y, elas_evecs_x[:, Pyx_nn, :]) 

        Pyx_nn_fine, Cxy_nn_fine= self.nnfap(self.Nf, Cxy_nn, evals_x, evals_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y) 
        elas_Pyx_nn_fine, elas_Cxy_nn_fine= self.elas_nnfap(self.Nf, elas_Cxy_nn, elas_evals_x, elas_evals_y, elas_evecs_x, elas_evecs_y, \
                                                                elas_evecs_trans_x, elas_evecs_trans_y, elas_mass_x, elas_mass_y)

        #------------------------loss terms in LBO basis-------------------#
        # loss on  LB
        self.loss_metrics = self.losses['functional_map_loss'](evecs_x, evecs_y, Pyx, Cxy, Cyx) # 使用refine Cxy_nn
        ## spatial loss
        #cross_contrastive loss + self_contrastive loss
        spatial_contrast_loss = self.losses['spatial_contrast_loss'](evecs_x, evecs_y, Pyx, Pyx_nn_fine, Pyy)  # Pyx_nn_refine 效果更好
        self.loss_metrics.update(spatial_contrast_loss)  # 

        if 'dirichlet_loss' in self.losses:
            Lx, Ly = data_x['operators']['L'], data_y['operators']['L']
            verts_x, verts_y = data_x['verts'], data_y['verts']
            self.loss_metrics['l_d'] = self.losses['dirichlet_loss'](torch.bmm(Pxy, verts_y), Lx) + \
                                       self.losses['dirichlet_loss'](torch.bmm(Pyx, verts_x), Ly)


        elas_spatial_spectral_loss = self.losses['elas_functional_map_loss'](elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y, Pyx, elas_Cxy, elas_Cyx)
        self.loss_metrics.update(elas_spatial_spectral_loss) 

        elas_spatial_contrast_loss = self.losses['elas_spatial_contrast_loss'](elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y, Pyx, elas_Pyx_nn_fine, Pyy)
        self.loss_metrics.update(elas_spatial_contrast_loss) 


    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        # test-time refinement
        if self.with_refine > 0:
            self.refine(data)
        
        # trim basis
        n_lb = self.opt.get('n_lb', 140)
        n_elas = self.opt.get('n_elas', 60)
        data_x = trim_basis(data_x, n_lb, n_elas)
        data_y = trim_basis(data_y, n_lb, n_elas)

        # feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))

        # get spectral operators
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans'].squeeze()
        evecs_trans_y = data_y['evecs_trans'].squeeze()
        elas_evecs_x = data_x['elas_evecs'].squeeze()
        elas_evecs_y = data_y['elas_evecs'].squeeze()
        elas_trans_x = data_x['elas_evecs_trans'].squeeze()
        elas_trans_y = data_y['elas_evecs_trans'].squeeze()
        elas_mass_x = data_x['elas_mass'].squeeze()
        elas_mass_y = data_y['elas_mass'].squeeze()

        evals_x = data_x['evals']  #[1,K]
        evals_y = data_y['evals']

        elas_evals_x = torch.abs(data_x['elas_evals'])
        elas_evals_y = torch.abs(data_y['elas_evals'])

        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)

        # nearest neighbour query
        p2pyx = nn_query(feat_x, feat_y).squeeze()

        if self.non_isometric:
            # compute Pyx from functional map
            Cxy = evecs_trans_y @ evecs_x[p2pyx]
            Pyx = evecs_y @ Cxy @ evecs_trans_x

        else:
            
            iter_num = 5  # for near-isometric matching

            filter_x, filter_y = self.generator_filter(evals_x, evals_y, self.Nf) #[1,Nf,K]
            elas_filter_x, elas_filter_y = self.generator_filter(elas_evals_x, elas_evals_y, self.Nf)

            # gs_x, gs_y = self.networks['conv'](evals_x, evals_y)
            # filter_x, filter_y = self.networks['comb'](gs_x, gs_y) # [1, 6, 200]

            # elas_gs_x, elas_gs_y  = self.networks['elas_conv'](elas_evals_x, elas_evals_y)
            # elas_filter_x, elas_filter_y = self.networks['elas_comb'](elas_gs_x, elas_gs_y) # [1, 6, 200]

            for _ in range(iter_num):

                # compute functional map from point-to-point map
                Cxy = evecs_trans_y @  evecs_x[p2pyx,:]
                elas_Cxy = elas_trans_y @ elas_evecs_x[p2pyx,:]

                Cxy = self.mcfp(filter_y, filter_x, Cxy.unsqueeze(0)).squeeze()  # [K ,K]
                elas_Cxy = self.mcfp(elas_filter_y, elas_filter_x, elas_Cxy.unsqueeze(0)).squeeze()  # [K ,K]

                # convert functional map to point-to-point map
                p2pyx = hybrid_fmap2pointmap(Cxy, evecs_x, evecs_y, elas_Cxy, elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y)

            # compute Pyx from functional map
            Pyx = evecs_y @ Cxy @ evecs_trans_x

        # finish record
        timer.record()

        # resume previous network state dict
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)

        return p2pyx, Pyx, Cxy

    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False, normalize=True):

        # if normalize:
        #     feat_x = F.normalize(feat_x, dim=-1, p=2)
        #     feat_y = F.normalize(feat_y, dim=-1, p=2)

        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # sinkhorn normalization
        Pxy = self.networks['permutation'](similarity)

        if bidirectional:
            Pyx = self.networks['permutation'](similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy

    def refine(self, data):
        self.networks['permutation'].hard = False
        # self.networks['fmap_net'].bidirectional = True

        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.curr_iter += 1
                self.feed_data(data)
                self.optimize_parameters()
        self.curr_iter = 0

        self.networks['permutation'].hard = True
        # self.networks['fmap_net'].bidirectional = False

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # change permutation prediction status
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = True
        # if 'fmap_net' in self.networks:
        #     self.networks['fmap_net'].bidirectional = False
        super(Hybrid_Self_Supervised_Model, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        # if 'fmap_net' in self.networks:
        #     self.networks['fmap_net'].bidirectional = True
    

    def optimize_parameters(self):
        """Override for Hybrid_ULRSSM_Model"""
        # n_lb = self.opt.get('n_lb', 140)
        # n_elas = self.opt.get('n_elas', 60)

        # # Loss normalization and weight scheduling
        # if n_lb > 0 and n_elas > 0:
        #     # Normalize LB and Elas Loss; 
        #     w_lb = 20000 / (n_lb * n_lb)
        #     w_elas = 20000 / (n_elas * n_elas)

        #     # early schduler for Elas Loss
        #     weight_schedule = self.opt['train'].get('weight_schedule', False)
        #     if weight_schedule:
        #         def linear_anneal(current_iter, total_iters):  #采用了退火的方式
        #             return min(max(current_iter / total_iters, 0.0), 1.0)  # 迭代之后开始限制elas 权重了
        #         curr_iter = self.curr_iter
        #         anneal_weight = linear_anneal(curr_iter, weight_schedule)

        #         w_elas = w_elas * anneal_weight

        #     # applying loss weights
        #     self.loss_metrics['l_orth'] *= w_lb
        #     self.loss_metrics['l_bij'] *= w_lb
        #     self.loss_metrics['l_spat_self'] *= w_lb
        #     self.loss_metrics['l_spat_cross'] *= w_lb


        #     self.loss_metrics['l_elas_orth'] *= w_elas
        #     self.loss_metrics['l_elas_bij'] *= w_elas
        #     self.loss_metrics['l_elas_spat_self'] *= w_elas
        #     self.loss_metrics['l_elas_spat_cross'] *= w_elas


        # compute total loss
        loss = 0.0
        for k, v in self.loss_metrics.items():
            if k != 'l_total':
                loss += v

        # update loss metrics
        self.loss_metrics['l_total'] = loss

        # zero grad
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

        # backward pass
        loss.backward()

        # clip gradient for stability
        for key in self.networks:
            torch.nn.utils.clip_grad_norm_(self.networks[key].parameters(), 1.0)

        # update weight
        for name in self.optimizers:
            self.optimizers[name].step()


    # refinemnet moudle on LBOs basis
    def nnfap(self, Nf, Cxy, evals_x, evals_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y, iter_num =1):

        filter_x, filter_y = self.generator_filter(evals_x, evals_y, Nf)  # 

        for _ in range(iter_num): 
            Cxy_fine = self.mcfp(filter_y, filter_x, Cxy)  # [1, K ,K]
            Pyx_fine = fmap2pointmap3D(Cxy_fine, evecs_x, evecs_y).squeeze() #
            # Cxy = torch.bmm(evecs_trans_y, evecs_x[:, Pyx_fine, :])  # 
        
        return Pyx_fine, Cxy_fine  


    # refinemnet moudle on Elastic basis
    def elas_nnfap(self, Nf, elas_Cxy_nn, elas_evals_x, elas_evals_y, elas_evecs_x, elas_evecs_y, elas_evecs_trans_x, \
                   elas_evecs_trans_y, elas_mass_x, elas_mass_y, iter_num = 1):
    
        # 这个结果不放了
        elas_filter_x, elas_filter_y = self.generator_filter(elas_evals_x, elas_evals_y, Nf)

        for _ in range(iter_num): 
            elas_Cxy_nn_fine = self.mcfp(elas_filter_y, elas_filter_x, elas_Cxy_nn)  # [1, K ,K]
            elas_Pyx_nn_fine = elas_fmap2pointmap(elas_Cxy_nn_fine.squeeze(), elas_evecs_x.squeeze(), elas_evecs_y.squeeze(), \
                                                    elas_mass_x.squeeze(), elas_mass_y.squeeze())
            # elas_Cxy_nn = torch.bmm(elas_evecs_trans_y, elas_evecs_x[:, elas_Pyx_nn_fine, :]) # 

        return elas_Pyx_nn_fine, elas_Cxy_nn_fine   # 


    # _functional maps refinemnet moudle
    def mcfp(self, gs_x, gs_y, Cyx):
    # input:
    #   gs_x/y: [1, Nf, Kx/Ky]
    #   Cyx : [1, K, K]

        C_new = torch.zeros_like(Cyx)  # 
        gs_y2 = torch.sum(gs_y**2, dim=1)  # 

        # MWP filters
        Nf = gs_x.size(1)  # 
        for s in range(Nf):
            C_new = C_new + gs_x[:,s,:].t()*Cyx*gs_y[:,s,:]

        C_new=C_new*(1/gs_y2)   # 
    
        return C_new  # [1, K, K]
    
    
    # Generator filter for functional map refinement
    def generator_filter(self, evals_x, evals_y, Ns, filter_type='meyer'):  # filter_type: 'learnable', 'meyer', 'mexican_hat'

        # 默认小波滤波器
        if filter_type =='learnable':
            gs_x, gs_y = self.networks['conv'](evals_x, evals_y)
            filter_x, filter_y = self.networks['comb'](gs_x, gs_y) # [1, 6, 200]

        else:  
            evals_x_cpu = evals_x.cpu().numpy()
            evals_y_cpu = evals_y.cpu().numpy()

            if filter_type =='meyer': 
                wavelet_gs_x = Meyer(max(evals_x_cpu[0]), Nf=Ns)(evals_x_cpu[0])  # evals_x[0] : [,K]; evals_x[1, K]
                wavelet_gs_y = Meyer(max(evals_y_cpu[0]), Nf=Ns)(evals_y_cpu[0])  # 
            
            if filter_type =='mexican_hat':
                wavelet_gs_x = Mexican_hat(max(evals_x_cpu[0]), Nf=Ns)(evals_x_cpu[0])  # evals_x[0] : [,K]; evals_x[1, K]
                wavelet_gs_y = Mexican_hat(max(evals_y_cpu[0]), Nf=Ns)(evals_y_cpu[0])  # 

            gs_x = wavelet_gs_x.to(self.device)  # numpy to torch
            gs_y = wavelet_gs_y.to(self.device)

            filter_x = gs_x.unsqueeze(0)  #[1,Nf,K]
            filter_y = gs_y.unsqueeze(0)


        return filter_x, filter_y


