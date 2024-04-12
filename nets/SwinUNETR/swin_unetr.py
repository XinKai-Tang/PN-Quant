from typing import Sequence
from torch import nn, Tensor, concat
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

from .swin_vit import SwinTransformer


class SwinUNETR(nn.Module):
    ''' Shifted windows UNEt TRansformers (Swin UNETR)
    * Authors: Ali Hatamizadeh, et al (2022)
    * Paper: Swin Transformers for Semantic Segmentation of Brain Tumours in MRI Images
    * Link: https://arxiv.org/pdf/2201.01266.pdf
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 patch_size: Sequence[int] = (2, 2, 2),
                 window_size: Sequence[int] = (7, 7, 7),
                 depths: Sequence[int] = (2, 2, 2, 2),
                 num_heads: Sequence[int] = (3, 6, 12, 24),
                 feature_size: int = 48,
                 dropout_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 end_norm: bool = True,
                 use_checkpoint: bool = False):
        ''' Args:
        * `in_channels`: dimension of input channels.
        * `out_channels`: dimension of output channels.
        * `patch_size`: dimension of patch size.
        * `window_size`: number of patchs in each window dimension.
        * `depths`: number of transformers in each stage.
        * `num_heads`: number of attention heads in each stage.
        * `feature_size`: dimension of network feature size.
        * `dropout_rate`: (SwinTransformer) dropout rate.
        * `attn_drop_rate`: (SwinTransformer) attention dropout rate.
        * `drop_path_rate`: (SwinTransformer) stochastic depth rate.
        * `end_norm`: whether normalizing output features in each stage.
        * `use_checkpoint`: whether using checkpointing to save memory.
        '''
        super(SwinUNETR, self).__init__()

        if 3 != len(patch_size):
            raise ValueError(f"Dimension of patch_size should be 3.")
        if 3 != len(window_size):
            raise ValueError(f"Dimension of window_size should be 3.")
        if 0 != feature_size % 12:
            raise ValueError("feature_size should be divisible by 12.")

        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate should be between 0 and 1.")
        if not (0 <= attn_drop_rate <= 1):
            raise ValueError(f"attn_drop_rate should be between 0 and 1.")
        if not (0 <= drop_path_rate <= 1):
            raise ValueError(f"drop_path_rate should be between 0 and 1.")

        self.swinViT = SwinTransformer(in_channels=in_channels,
                                       embed_dim=feature_size,
                                       patch_size=patch_size,
                                       window_size=window_size,
                                       depths=depths,
                                       num_heads=num_heads,
                                       dropout_rate=dropout_rate,
                                       attn_drop_rate=attn_drop_rate,
                                       drop_path_rate=drop_path_rate,
                                       mlp_ratio=4.0,
                                       norm_layer=nn.LayerNorm,
                                       qkv_bias=True,
                                       patch_norm=False,
                                       end_norm=end_norm,
                                       use_checkpoint=use_checkpoint)

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size*2,
            out_channels=feature_size*2,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size*4,
            out_channels=feature_size*4,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size*16,
            out_channels=feature_size*16,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size*16,
            out_channels=feature_size*8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size*8,
            out_channels=feature_size*4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size*4,
            out_channels=feature_size*2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size*2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.output = UnetOutBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def project(self, x: Tensor):
        ''' Project x[B, D, H, W, C] to x[B, C, D, H, W]. '''
        x = x.permute(0, 4, 1, 2, 3).contiguous()   # B, C, D, H, W
        return x

    def forward(self, inputs: Tensor):
        x, hidden_states_out = self.swinViT(inputs)

        enc0 = self.encoder1(inputs)
        enc1 = self.encoder2(self.project(hidden_states_out[0]))
        enc2 = self.encoder3(self.project(hidden_states_out[1]))
        enc3 = self.encoder4(self.project(hidden_states_out[2]))
        enc4 = self.project(hidden_states_out[3])
        dec4 = self.encoder5(self.project(hidden_states_out[4]))

        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        logits = self.output(out)
        return logits


if __name__ == '__main__':
    import torch
    model = SwinUNETR(1, 2)
    input = torch.randn(size=(1, 1, 64, 64, 64))
    output = model(input)
    n_params = sum([param.nelement() for param in model.parameters()])
    print(model)
    print("Dimension of outputs:", output.shape)
    print("Number of parameters: %.2fM" % (n_params / 1024**2))
