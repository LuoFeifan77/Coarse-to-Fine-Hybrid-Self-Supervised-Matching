import torch
import torch.nn as nn

from utils.fmap_util import spectral_mass_computation
from utils.registry import LOSS_REGISTRY



# -----------------------------------Part1: LOSS ON LBOs BASIS ----------------------------------- #

@LOSS_REGISTRY.register()
class FunctionalMapLoss(nn.Module):
    def __init__(self, w_orth = 1.0, w_bij = 1.0):  #
        super(FunctionalMapLoss, self).__init__()
        assert w_orth >= 0 and w_bij >= 0 
        self.w_orth = w_orth  # 
        self.w_bij = w_bij

    def forward(self, evecs_x, evecs_y, Pyx, Cxy, Cyx): # 

        losses = dict()

        if self.w_orth > 0:

            orth_loss = torch.linalg.norm(evecs_y - torch.bmm(Pyx, torch.bmm(evecs_x, Cxy.transpose(-2, -1)))) #用refine效果不好
            losses['l_orth'] = self.w_orth * orth_loss 

        if self.w_bij > 0:

            bij_loss = torch.linalg.norm(evecs_y - torch.bmm(Pyx, torch.bmm(evecs_x, Cyx))) 
            losses['l_bij'] = self.w_bij * bij_loss 

        return losses


@LOSS_REGISTRY.register()
class SpatialContrastiveLoss(nn.Module):
    def __init__(self, w_cross=1.0, w_self=1.0,):
        super(SpatialContrastiveLoss, self).__init__()
        assert w_cross >= 0 and w_self >= 0
        self.w_cross = w_cross  # inter contrastive
        self.w_self = w_self
        
        # 
    def forward(self, evecs_x, evecs_y, Pyx, Pyx_nn_fine, Pyy): 

        losses = dict()

        
        if self.w_cross>0:  

            cross_loss = torch.linalg.norm(torch.bmm(Pyx, evecs_x) - evecs_x[:, Pyx_nn_fine, :])             
            losses['l_spat_cross'] = self.w_cross*cross_loss

        if self.w_self>0:    
            self_loss = torch.linalg.norm(torch.bmm(Pyy, evecs_y) - evecs_y)  #self contrastive
            losses['l_spat_self'] = self.w_self*self_loss


        return losses




#-------------------Part2: LOSS ON ELASTIC BASIS ------------------- #
@LOSS_REGISTRY.register()
class ElasSpatialSpectralLoss(nn.Module):
    def __init__(self, w_elas_spat_spec = 1.0):
        super(ElasSpatialSpectralLoss, self).__init__()
        assert w_elas_spat_spec >= 0
        self.w_elas_spat_spec = w_elas_spat_spec

    def forward(self, elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y, Pyx, elas_Cxy):

        losses = dict()

        if self.w_elas_spat_spec>0:

            elas_evecs_x = elas_evecs_x.squeeze()
            elas_evecs_y = elas_evecs_y.squeeze()

            elas_mass_x = elas_mass_x.squeeze()  #[1,N,k] -->[N, k]
            elas_mass_y = elas_mass_y.squeeze()

            elas_Cxy = elas_Cxy.squeeze()

            Mxk, sqrtMxk, invsqrtMxk = spectral_mass_computation(elas_evecs_x, elas_mass_x)
            Myk, sqrtMyk, invsqrtMyk = spectral_mass_computation(elas_evecs_y, elas_mass_y)

           
            # area has been removed.
            dataA = elas_evecs_y @ invsqrtMyk
            dataB = Pyx @ elas_evecs_x @ torch.inverse(Mxk) @ elas_Cxy.t() @ Myk @ invsqrtMyk 

            elas_spat_spec_loss = torch.linalg.norm( dataA.unsqueeze(0) -  dataB.unsqueeze(0)) 

            losses['l_elas_spat_spec'] = self.w_elas_spat_spec * elas_spat_spec_loss

        return losses




@LOSS_REGISTRY.register()
class ElasFunctionalMapLoss(nn.Module):
    def __init__(self, w_elas_orth = 1.0, w_elas_bij = 1.0):  
        super(ElasFunctionalMapLoss, self).__init__()
        assert w_elas_orth >= 0 and w_elas_bij >= 0 
        self.w_elas_orth = w_elas_orth  # 
        self.w_elas_bij = w_elas_bij

    def forward(self, elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y, Pyx, elas_Cxy, elas_Cyx): #

        losses = dict()

        elas_evecs_x = elas_evecs_x.squeeze()
        elas_evecs_y = elas_evecs_y.squeeze()

        elas_mass_x = elas_mass_x.squeeze()  #[1,N,k] -->[N, k]
        elas_mass_y = elas_mass_y.squeeze()

        elas_Cxy = elas_Cxy.squeeze()

        Mxk, sqrtMxk, invsqrtMxk = spectral_mass_computation(elas_evecs_x, elas_mass_x)
        Myk, sqrtMyk, invsqrtMyk = spectral_mass_computation(elas_evecs_y, elas_mass_y)

        
        dataA =  elas_evecs_y @ invsqrtMyk

        if self.w_elas_orth > 0:

            # area has been removed.
            dataB = Pyx @ elas_evecs_x @ torch.inverse(Mxk) @ elas_Cxy.t() @ Myk @ invsqrtMyk 
            losses['l_elas_orth'] = self.w_elas_orth * torch.linalg.norm(dataA-dataB) # 


        if self.w_elas_bij > 0:

            # area has been removed.
            dataB = Pyx @ elas_evecs_x @ elas_Cyx.squeeze() @ invsqrtMyk 
            losses['l_elas_bij'] = self.w_elas_bij * torch.linalg.norm(dataA-dataB) #双向loss

        return losses



@LOSS_REGISTRY.register()
class ElasSpatialContrastiveLoss(nn.Module):
    def __init__(self, w_elas_cross=1.0, w_elas_self=1.0,):
        super(ElasSpatialContrastiveLoss, self).__init__()
        assert w_elas_cross >= 0 and w_elas_self >= 0
        self.w_elas_cross = w_elas_cross
        self.w_elas_self = w_elas_self

    def forward(self, elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y, Pyx, elas_Pyx_nn_fine, Pyy):

        losses = dict()

        elas_evecs_x = elas_evecs_x.squeeze()
        elas_mass_x = elas_mass_x.squeeze()  #[1,N,k] -->[N, k]
        Mxk, sqrtMxk, invsqrtMxk = spectral_mass_computation(elas_evecs_x, elas_mass_x)

        elas_evecs_y = elas_evecs_y.squeeze()
        elas_mass_y = elas_mass_y.squeeze()  #[1,N,k] -->[N, k]
        Myk, sqrtMyk, invsqrtMyk = spectral_mass_computation(elas_evecs_y, elas_mass_y)

        if self.w_elas_cross>0:

            termY = elas_evecs_x @ invsqrtMxk # [N,k]
            termYp = elas_evecs_x[elas_Pyx_nn_fine, :] @ invsqrtMxk    

            losses['l_elas_spat_cross'] = self.w_elas_cross*torch.linalg.norm(torch.bmm(Pyx, termY.unsqueeze(0)) - termYp.unsqueeze(0))

 
        if self.w_elas_self>0:

            termY = elas_evecs_y @ invsqrtMyk # [N,k]
            # elas_self_loss = torch.linalg.norm(torch.bmm(Pyy, termY.unsqueeze(0)) - termY.unsqueeze(0)) 
            losses['l_elas_spat_self'] = self.w_elas_self*torch.linalg.norm(torch.bmm(Pyy, termY.unsqueeze(0)) - termY.unsqueeze(0))


        return losses



# @LOSS_REGISTRY.register()
# class ElasSpectralContrastiveLoss(nn.Module):
#     def __init__(self, w_elas_spec=1.0):
#         super(ElasSpectralContrastiveLoss, self).__init__()
#         self.w_elas_spec = w_elas_spec

#     def forward(self, elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y, Pyx, elas_Pyx_nn_fine, elas_Cxy, elas_Cxy_nn_fine): 
        
#         losses = dict()

#         if self.w_elas_spec>0:    

#             elas_evecs_x = elas_evecs_x.squeeze()
#             # elas_evecs_y = elas_evecs_y.squeeze()

#             elas_mass_x = elas_mass_x.squeeze()  #[1,N,k] -->[N, k]
#             # elas_mass_y = elas_mass_y.squeeze()

#             elas_Cxy = elas_Cxy.squeeze()

#             Mxk, sqrtMxk, invsqrtMxk = spectral_mass_computation(elas_evecs_x, elas_mass_x)
#             # Myk, sqrtMyk, invsqrtMyk = spectral_mass_computation(elas_evecs_y, elas_mass_y)

#             #
#             termY = elas_evecs_x @ invsqrtMxk # [N,k]
#             termYp = elas_evecs_x[elas_Pyx_nn_fine, :] @ invsqrtMxk

#             elas_spec_loss = torch.linalg.norm(torch.bmm(Pyx, termY.unsqueeze(0)) - termYp.unsqueeze(0))

#             losses['l_elas_spec'] = self.w_elas_spec*elas_spec_loss

#         return losses





