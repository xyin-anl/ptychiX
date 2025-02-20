import argparse
import pytest

import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size, get_default_complex_dtype, generate_initial_opr_mode_weights

import test_utils as tutils


class Test2dPtychoDIP(tutils.TungstenDataTester):

    @pytest.mark.local
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_dip')
    def test_2d_ptycho_dip(self):
        self.setup_ptychi(cpu_only=False)
        
        # For now, use high tolerance to avoid failure due to stochasticity.
        self.atol = 0.5
        
        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(additional_opr_modes=0)
        
        options = api.AutodiffPtychographyOptions()
        options.data_options.data = data
        
        options.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.deep_image_prior_options.enabled = True
        # options.object_options.deep_image_prior_options.model = api.enums.DIPModels.UNET
        # options.object_options.deep_image_prior_options.model_params = {
        #     "num_in_channels": 32,
        #     "num_out_channels": 2,
        #     "initialize": False,
        #     "skip_connections": False,
        # }
        # options.object_options.deep_image_prior_options.constrain_object_outside_network = True
        options.object_options.deep_image_prior_options.model = api.enums.DIPModels.AUTOENCODER
        options.object_options.deep_image_prior_options.model_params = {
            "num_in_channels": 32,
            "num_levels": 4,
            "base_channels": 32,
            "use_batchnorm": True,
        }
        options.object_options.deep_image_prior_options.constrain_object_outside_network = False
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.ADAM
        options.object_options.step_size = 1e-5
        
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.step_size = 0.1

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = False
        
        options.reconstructor_options.batch_size = 100
        options.reconstructor_options.num_epochs = 16
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0, 250:500, 250:500]
        return recon
    
    @pytest.mark.local
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_dip_l2')
    def test_2d_ptycho_dip_l2(self):
        self.setup_ptychi(cpu_only=False)
        
        # For now, use high tolerance to avoid failure due to stochasticity.
        self.atol = 0.5
        
        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(additional_opr_modes=0)
        
        options = api.AutodiffPtychographyOptions()
        options.data_options.data = data
        
        options.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.deep_image_prior_options.enabled = True
        options.object_options.deep_image_prior_options.model = api.enums.DIPModels.AUTOENCODER
        options.object_options.deep_image_prior_options.model_params = {
            "num_in_channels": 32,
            "num_levels": 4,
            "base_channels": 32,
            "use_batchnorm": True,
        }
        options.object_options.deep_image_prior_options.constrain_object_outside_network = False
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.ADAM
        options.object_options.step_size = 1e-5
        options.object_options.l2_norm_constraint.enabled = True
        options.object_options.l2_norm_constraint.weight = 1e-2
        
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.step_size = 0.1

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = False
        
        options.reconstructor_options.batch_size = 100
        options.reconstructor_options.num_epochs = 16
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0, 250:500, 250:500]
        return recon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Test2dPtychoDIP()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_dip()
    tester.test_2d_ptycho_dip_l2()
