from typing import Sequence
from torch import nn, Tensor

from .convnet import UXNetConv
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock


class UXNET(nn.Module):
    ''' 3D UX-Net
    * Authors: Ho Hin Lee et al (2022)
    * Paper: a Large Kernel Volumetric ConvNet Modernizing Hierarchical Transformer for Medical Image Segmentation
    * Link: https://arxiv.org/pdf/2209.15076.pdf
    '''

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 16,
                 depths: Sequence = [2, 2, 2, 2],
                 feature_size: Sequence = 48,
                 drop_path_rate: float = 0.0,
                 layer_scaling: float = 1e-6,
                 spatial_dim: int = 3):
        """ Args:
        * `in_channels`: dimension of input channels.
        * `out_channels`: dimension of output channels.
        * `depths`: number of ConvNeXt blocks in each stage.
        * `feature_size`: output channels of the steam layer.
        * `drop_path_rate`: stochastic depth rate.
        * `layer_scaling`: initial value of layer scaling.
        * `spatial_dim`: number of spatial dimensions.
        """
        super(UXNET, self).__init__()

        self.backbone = UXNetConv(
            in_channels=in_channels,
            depths=depths,
            feature_size=feature_size,
            spatial_dim=spatial_dim,
            drop_path_rate=drop_path_rate,
            layer_scaling=layer_scaling
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size,
            out_channels=feature_size*2,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size*2,
            out_channels=feature_size*4,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size*4,
            out_channels=feature_size*8,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size*8,
            out_channels=feature_size*16,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size*16,
            out_channels=feature_size*8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size*8,
            out_channels=feature_size*4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size*4,
            out_channels=feature_size*2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size*2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.output = UnetOutBlock(
            in_channels=feature_size,
            out_channels=out_channels,
            spatial_dims=spatial_dim
        )

    def forward(self, x: Tensor):
        hidden_states = self.backbone(x)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(hidden_states[0])
        enc3 = self.encoder3(hidden_states[1])
        enc4 = self.encoder4(hidden_states[2])
        dec5 = self.encoder5(hidden_states[3])
        dec4 = self.decoder5(dec5, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        out = self.decoder1(dec1)

        logits = self.output(out)
        return logits


if __name__ == '__main__':
    import torch
    model = UXNET(1, 2)
    input = torch.randn(size=(1, 1, 64, 64, 64))
    output = model(input)
    n_params = sum([param.nelement() for param in model.parameters()])
    print(model)
    print("Dimension of outputs:", output.shape)
    print("Number of parameters: %.2fM" % (n_params / 1024**2))
