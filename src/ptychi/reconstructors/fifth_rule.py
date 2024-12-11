from typing import Optional, TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from torch import Tensor
import copy

from ptychi.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
)
from ptychi.metrics import MSELossOfSqrt
if TYPE_CHECKING:
    import ptychi.api as api
    import ptychi.data_structures.parameter_group as pg


class FifthRuleReconstructor(AnalyticalIterativePtychographyReconstructor):
    """
    The "fifth rule" Conjugate Gradient reconstructor 
    (Carlsson & Nikitin, 2025, in prep)
    """
    
    parameter_group: "pg.PlanarPtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PlanarPtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.fifth_rule.FifthRuleReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            parameter_group=parameter_group,
            dataset=dataset,
            options=options,
            *args,
            **kwargs,
        )
        # temp object to work with patches
        self.object_tmp = copy.deepcopy(self.parameter_group.object)
        # intermeadiate variables for the conjugate gradient directions
        self.forward_model.record_intermediate_variable('eta_o',torch.zeros_like(self.parameter_group.object.get_slice(0)))
        self.forward_model.record_intermediate_variable('eta_p',torch.zeros_like(self.parameter_group.probe.get_opr_mode(0)))
        
    def build_loss_tracker(self):
        if self.displayed_loss_function is None:
            self.displayed_loss_function = MSELossOfSqrt()
        return super().build_loss_tracker()

    def check_inputs(self, *args, **kwargs):
        if self.parameter_group.object.is_multislice:
            raise NotImplementedError("The fifth rule only supports 2D objects.")        
        if self.parameter_group.probe.has_multiple_opr_modes:
            raise NotImplementedError("The fifth rule does not support multiple OPR modes yet.")

    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        (delta_o, delta_p, delta_pos), y_pred = self.compute_updates(
            *input_data, y_true, self.dataset.valid_pixel_mask
        )
        self.apply_updates(delta_o, delta_p, delta_pos)
        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)
        
    def compute_updates(
        self, indices: torch.Tensor, y_true: torch.Tensor, valid_pixel_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Calculates the updates of the whole object, the probe, and other parameters.
        This function is called in self.update_step_module.forward.
        """
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        self.indices = indices.cpu()
        self.positions = probe_positions.tensor[indices]

        y = self.forward_model.forward(indices)

        # intermediate variables for the CG
        eta_o = self.forward_model.intermediate_variables["eta_o"]
        eta_p = self.forward_model.intermediate_variables["eta_p"]
        
        # take current object and probe
        p = probe.get_opr_mode(0)
        o = object_.get_slice(0)
        
        # sqrt of data
        d = torch.sqrt(y_true)[:,torch.newaxis]        

        # Gradient for the Gaussian model
        gradF = self.gradientF(o,p,d)    

        # Gradients for the object and probe
        if object_.optimization_enabled(self.current_epoch):                 
            grad_o = self.gradient_o(p,gradF)                            
        else:
            grad_o = torch.zeros_like(o)
        if probe.optimization_enabled(self.current_epoch):             
            grad_p = self.parameter_group.probe.options.rho*self.gradient_p(o,gradF)
        else:
            grad_p = torch.zeros_like(p)
        
        if self.current_epoch==0:
            eta_o = -grad_o
            eta_p = -grad_p
        else:
            # CG direction
            beta = self.calc_beta(o,p,grad_o,grad_p,eta_o,eta_p,d,gradF)
            eta_o = -grad_o+beta*eta_o
            eta_p = -grad_p+beta*eta_p
        
        # The step size is for both object and probe
        alpha,_,_ = self.calc_alpha(o,p,grad_o,grad_p,eta_o,eta_p,d,gradF)                               
                 
        delta_o = None
        if object_.optimization_enabled(self.current_epoch):             
            delta_o = alpha*eta_o      
            delta_o = delta_o.unsqueeze(0)
            self.forward_model.record_intermediate_variable('eta_o',eta_o)
            
        delta_p_all_modes = None
        if probe.optimization_enabled(self.current_epoch):
            delta_p = self.parameter_group.probe.options.rho*alpha*eta_p
            delta_p_all_modes = delta_p[None, :, :]
            self.forward_model.record_intermediate_variable('eta_p',eta_p)
            
        delta_pos = None
        if probe_positions.optimization_enabled(self.current_epoch) and object_.optimizable:
            print('WARNING: positions update is not implemented yet for the fifth rule method')
       
        return (delta_o, delta_p_all_modes, delta_pos), y

    def extract_patches(self,o):
        '''Extract patches from the object, short version'''

        self.object_tmp.set_data(o)
        patches = self.object_tmp.extract_patches(self.positions, self.parameter_group.probe.get_spatial_shape())   
        return patches

    def place_patches(self,patches):
        '''Place patches to the object, short version'''

        self.object_tmp.set_data(torch.zeros_like(self.object_tmp.get_slice(0)))             
        o = self.object_tmp.place_patches_function(
            torch.zeros_like(self.object_tmp.get_slice(0)),
            self.positions + self.object_tmp.center_pixel,                
            patches[:,0],
            op='add'
            )       
        return o  
    
    def reprod(self,a,b):
        '''Real part of product of vector a and conjugate of vector b '''
        return a.real*b.real+a.imag*b.imag

    def redot(self,a,b,axis=None):    
        '''Real part of dot product of vector a and conjugate of vector b '''
        res = torch.sum(self.reprod(a,b),axis=axis)        
        return res        
    
    def gradientF(self,o,p,d):     
        '''Gradient for the Guassian model'''
        op = p*self.extract_patches(o)
        Lo = self.forward_model.far_field_propagator.propagate_forward(op)
        td = d*(Lo/(torch.abs(Lo)+1e-7))
        res = 2*self.forward_model.far_field_propagator.propagate_backward(Lo-td)        
        return res

    def hessianF(self,o,o1,o2,data):
        '''Hessian for the Guassian model'''
        Lo = self.forward_model.far_field_propagator.propagate_forward(o)
        Lo1 = self.forward_model.far_field_propagator.propagate_forward(o1)                
        Lo2 = self.forward_model.far_field_propagator.propagate_forward(o2)
        l0 = Lo/(torch.abs(Lo)+1e-7)
        d0 = data/(torch.abs(Lo)+1e-7)
        v1 = torch.sum((1-d0)*self.reprod(Lo1,Lo2))                
        v2 = torch.sum(d0*self.reprod(l0,Lo1)*self.reprod(l0,Lo2))    
        return 2*(v1+v2)  
        
    def gradient_o(self,p,gradF):                
        '''Gradient with respect to the object'''
        tmp = torch.conj(p)*gradF
        o = self.place_patches(tmp)        
        return o
    
    def gradient_p(self,o,gradF):
        '''Gradient with respect to the probe'''
        patches = self.extract_patches(o)                             
        return torch.sum(torch.conj(patches)*gradF,axis=0)    
    
    def DM(self,o,p,do,dp):
        '''First differential to compute the Hessian in the bilinear form'''
        patches = self.extract_patches(o)
        patches1 = self.extract_patches(do)
        res = dp*patches+p*patches1
        return res   
    
    def D2M(self,do1,dp1,do2,dp2):    
        '''Second differential to compute the Hessian in the bilinear form'''
        patches1 = self.extract_patches(do1)
        patches2 = self.extract_patches(do2)
        res = dp1*patches2 + dp2*patches1
        return res
    
    def calc_alpha(self,o,p,do1,dp1,do2,dp2,d,gradF):    
        '''Step length for the object and probe update'''        
        top = -self.redot(do1,do2)-self.redot(dp1,dp2)
                        
        dp2 = self.parameter_group.probe.options.rho*dp2
        dm2 = self.DM(o,p,do2,dp2)
        d2m2 = self.D2M(do2,dp2,do2,dp2)
        op = p*self.extract_patches(o)
        bottom = self.redot(gradF,d2m2)+self.hessianF(op, dm2, dm2,d)
        return top/bottom, top, bottom
                           
    def calc_beta(self,o,p,do1,dp1,do2,dp2,d,gradF):
        '''Step length for the CG direction'''  
        dp1 = self.parameter_group.probe.options.rho*dp1
        dp2 = self.parameter_group.probe.options.rho*dp2

        dm1 = self.DM(o,p,do1,dp1)
        dm2 = self.DM(o,p,do2,dp2)
        d2m1 = self.D2M(do1,dp1,do2,dp2)
        d2m2 = self.D2M(do2,dp2,do2,dp2)
        op = p*self.extract_patches(o)

        top = self.redot(gradF,d2m1)        
        top += self.hessianF(op, dm1, dm2,d)    
        bottom = self.redot(gradF,d2m2)    
        bottom += self.hessianF(op, dm2, dm2,d)
        return top/bottom 
                    
    def apply_updates(self, delta_o, delta_p, delta_pos, *args, **kwargs):
        """
        Apply updates to optimizable parameters given the updates calculated by self.compute_updates.

        Parameters
        ----------
        delta_o : Tensor
            A (n_replica, h, w, 2) tensor of object update vector.
        delta_p : Tensor
            A (n_replicate, n_opr_modes, n_modes, h, w, 2) tensor of probe update vector.
        delta_pos : Tensor
            A (n_positions, 2) tensor of probe position vectors.
        """
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        if delta_o is not None:
            object_.set_grad(-delta_o)
            object_.optimizer.step()
            
        if delta_p is not None:
            probe.set_grad(-delta_p)
            probe.optimizer.step()
            
        if delta_pos is not None:
            probe_positions.set_grad(-delta_pos)
            probe_positions.optimizer.step()
           
