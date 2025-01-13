import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, num_in_channels=1, num_levels=3, base_channels=32, use_batchnorm=False):
        """
        Convolutional autoencoder model with adjustable number of levels.

        Parameters
        ----------
        num_in_channels : int
            The number of input channels.
        num_levels : int
            The number of levels in the autoencoder.
        base_channels : int
            The base number of channels. In the encoder part, the number of output channels
            of level `i` is `base_channels * 2 ** i`.
        use_batchnorm : bool
            Whether to use batch normalization.
        """
        super(Autoencoder, self).__init__()
        self.num_levels = num_levels
        self.num_in_channels = num_in_channels
        self.base_channels = base_channels
        self.use_batchnorm = use_batchnorm

        down_blocks = []
        for level in range(self.num_levels):
            down_blocks += self.get_down_block(level)
        up_blocks_1 = []
        up_blocks_2 = []
        for level in range(self.num_levels - 1, -1, -1):
            up_blocks_1 += self.get_up_block(level)
            up_blocks_2 += self.get_up_block(level)
        self.encoder = nn.Sequential(
            # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *down_blocks
        )

        self.decoder1 = nn.Sequential(
            *up_blocks_1,
            nn.Conv2d(self.base_channels * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Sigmoid()  # Amplitude model
        )

        self.decoder2 = nn.Sequential(
            *up_blocks_2,
            nn.Conv2d(self.base_channels * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Tanh()  # Phase model
        )

    def get_down_block(self, level):
        """
        Get a list of layers in a downsampling block.

        Parameters
        ----------
        level : int
            0-based level index.
            
        Returns
        -------
        list[nn.Module]
            List of layers in the downsampling block.
        """
        num_in_channels = int(self.base_channels * 2 ** (level - 1))
        num_out_channels = int(self.base_channels * 2 ** level)
        if level == 0:
            num_in_channels = self.num_in_channels

        blocks = []
        blocks.append(nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,
                      kernel_size=3, stride=1, padding=(1, 1)))
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv2d(num_out_channels, num_out_channels, 3, stride=1, padding=(1, 1)))
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        blocks.append(nn.MaxPool2d((2, 2)))

        return blocks

    def get_up_block(self, level):
        """
        Get a list of layers in a upsampling block.
        Parameters
        ----------
        level : int
            0-based level index.
            
        Returns
        -------
        list[nn.Module]
            List of layers in the upsampling block.
        """
        if level == self.num_levels - 1:
            num_in_channels = self.base_channels * 2 ** level
            num_out_channels = self.base_channels * 2 ** level
        elif level == 0:
            num_in_channels = self.base_channels * 2 ** (level + 1)
            num_out_channels = self.base_channels * 2 ** (level + 1)
        else:
            num_in_channels = self.base_channels * 2 ** (level + 1)
            num_out_channels = self.base_channels * 2 ** level
        num_in_channels = int(num_in_channels)
        num_out_channels = int(num_out_channels)

        blocks = []
        blocks.append(nn.Conv2d(num_in_channels, num_out_channels, 3, stride=1, padding=(1, 1)))
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv2d(num_out_channels, num_out_channels, 3, stride=1, padding=(1, 1)))
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        blocks.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        return blocks

    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        # Restore -pi to pi range
        ph = ph * torch.pi  # Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp, ph