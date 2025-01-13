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
        
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.num_epochs = 1000
        
        task = PtychographyTask(options)
        
        if self.debug:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2)
        for i in range(1000):
            task.run(1)
            if self.debug:
                ax[0].clear()
                ax[1].clear()
                try:
                    data_mag = task.reconstructor.parameter_group.object.dip_output_magnitude[0].cpu().numpy()
                    data_phase = task.reconstructor.parameter_group.object.dip_output_phase[0].cpu().numpy()
                except:
                    data_mag = task.reconstructor.parameter_group.object.data.detach().abs()[0].cpu().numpy()
                    data_phase = task.reconstructor.parameter_group.object.data.detach().angle()[0].cpu().numpy()
                im0 = ax[0].imshow(data_mag)
                im1 = ax[1].imshow(data_phase)
                if i == 0:
                    # First iteration - create plots and colorbar
                    cb0 = plt.colorbar(im0, ax=ax[0])
                    cb1 = plt.colorbar(im1, ax=ax[1])
                else:
                    cb0.update_normal(im0)
                    cb1.update_normal(im1)
                
                ax[0].set_title(f'Magnitude Epoch {i}')
                ax[1].set_title(f'Phase Epoch {i}')
                plt.draw()
                plt.pause(0.001)  # Small pause to allow GUI to update
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0]
        return task
    
    def plot_object(self, task: PtychographyTask):
        import matplotlib.pyplot as plt
        import tifffile
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(task.reconstructor.parameter_group.object.dip_output_magnitude[0].cpu().numpy())
        ax[1].imshow(task.reconstructor.parameter_group.object.dip_output_phase[0].cpu().numpy())
        plt.show()
        tifffile.imwrite('dip_output_magnitude.tif', task.reconstructor.parameter_group.object.dip_output_magnitude[0].cpu().numpy())
        tifffile.imwrite('dip_output_phase.tif', task.reconstructor.parameter_group.object.dip_output_phase[0].cpu().numpy())
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Test2dPtychoDIP()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_dip()
