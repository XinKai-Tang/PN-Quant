from typing import Sequence, Union
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class DropPath(nn.Module):
    ''' Stochastic drop paths per sample for residual blocks.

        Reference: https://github.com/rwightman/pytorch-image-models
    '''

    def __init__(self,
                 drop_prob: float = 0.0,
                 scale_by_keep: bool = True):
        ''' Args:
            * `drop_prob`: drop paths probability.
            * `scale_by_keep`: whether scaling by non-dropped probaility.
        '''
        super(DropPath, self).__init__()
        if drop_prob < 0 or drop_prob > 1:
            raise ValueError("drop_path_prob should be between 0 and 1.")
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor):
        if self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep and keep_prob > 0.0:
            rand_tensor.div_(keep_prob)
        return x * rand_tensor


class LayerNorm(nn.Module):
    ''' Layer Normalization '''

    def __init__(self,
                 norm_shape: Union[Sequence[int], int],
                 eps: float = 1e-6,
                 channels_last: bool = True):
        ''' Args:
        * `norm_shape`: dimension of the input feature.
        * `eps`: epsilon of layer normalization.
        * `channels_last`: whether the channel is the last dim.
        '''
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(norm_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(norm_shape), requires_grad=True)
        self.channels_last = channels_last
        self.norm_shape = (norm_shape,)
        self.eps = eps

    def forward(self, x: Tensor):
        if self.channels_last:  # [B, ..., C]
            y = F.layer_norm(x, self.norm_shape,
                             self.weight, self.bias, self.eps)
        else:                   # [B, C, ...]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            y = (x - mean) / torch.sqrt(var + self.eps)
            if x.ndim == 4:
                y = self.weight[:, None, None] * y
                y += self.bias[:, None, None]
            else:
                y = self.weight[:, None, None, None] * y
                y += self.bias[:, None, None, None]
        return y


class UX_Block(nn.Module):
    def __init__(self,
                 in_dim: int,
                 drop_path_rate: float = 0.0,
                 layer_scaling: float = 1e-6,
                 spatial_dim: int = 3):
        ''' Args:
        * `in_dim`: dimension of input channels.
        * `drop_path_rate`: stochastic depth rate.
        * `layer_scaling`: initial value of layer scaling.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(UX_Block, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        # Depthwise Networks:
        self.dw_conv = Conv(in_dim, in_dim, kernel_size=7,
                            padding=3, groups=in_dim)
        self.norm = LayerNorm(in_dim, eps=1e-6)
        # Pointwise Networks:
        self.pw_conv = nn.Sequential(Conv(in_dim, 4 * in_dim, 1, groups=in_dim),
                                     nn.GELU(),
                                     Conv(4 * in_dim, in_dim, 1, groups=in_dim))
        # Layer scale
        if layer_scaling > 0:
            self.gamma = nn.Parameter(layer_scaling * torch.ones((in_dim,)),
                                      requires_grad=True)
        else:
            self.gamma = None
        # Drop Path
        if drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x: Tensor):
        shortcut = x        # 暂存处理之前的x
        x = self.dw_conv(x)

        if x.ndim == 5:
            x = x.permute(0, 2, 3, 4, 1)    # [B,C,D,H,W] -> [B,D,H,W,C]
        else:
            x = x.permute(0, 2, 3, 1)       # [B,C,H,W] -> [B,H,W,C]
        x = self.norm(x)

        if x.ndim == 5:
            x = x.permute(0, 4, 1, 2, 3)    # [B,D,H,W,C] -> [B,C,D,H,W]
        else:
            x = x.permute(0, 3, 1, 2)       # [B,H,W,C] -> [B,C,H,W]
        x = self.pw_conv(x)

        if x.ndim == 5:
            x = x.permute(0, 2, 3, 4, 1)    # [B,C,D,H,W] -> [B,D,H,W,C]
        else:
            x = x.permute(0, 2, 3, 1)       # [B,C,H,W] -> [B,H,W,C]
        if self.gamma is not None:
            x = self.gamma * x

        if x.ndim == 5:
            x = x.permute(0, 4, 1, 2, 3)    # [B,D,H,W,C] -> [B,C,D,H,W]
        else:
            x = x.permute(0, 3, 1, 2)       # [B,H,W,C] -> [B,C,H,W]
        y = shortcut + self.drop_path(x)
        return y


class UXNetConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 depths: Sequence[int] = (3, 3, 9, 3),
                 feature_size: int = 96,
                 spatial_dim: int = 3,
                 drop_path_rate: float = 0.0,
                 layer_scaling: float = 1e-6):
        ''' Args:
        * `in_channels`: dimension of input channels.
        * `depths`: number of ConvNeXt block in each stage.
        * `feature_size`: output channels of the steam layer.
        * `spatial_dim`: number of spatial dimensions.
        * `drop_path_rate`: stochastic depth rate.
        * `layer_scaling`: initial value of layer scaling.
        '''
        super(UXNetConv, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        # stem and 3 intermediate downsampling conv layers
        self.down_samplers = nn.ModuleList()
        self.down_samplers.append(nn.Sequential(    # stem
            Conv(in_channels, feature_size, kernel_size=7, stride=2, padding=3),
            LayerNorm(feature_size, eps=1e-6, channels_last=False)
        ))
        for i in range(3):  # 3 intermediate downsampling conv
            dim = (2 ** i) * feature_size
            self.down_samplers.append(nn.Sequential(
                LayerNorm(dim, eps=1e-6, channels_last=False),
                Conv(dim, 2 * dim, kernel_size=2, stride=2)
            ))

        # 4 stages of UX-Block
        self.stages = nn.ModuleList()
        dp_rates = [r.item()
                    for r in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0     # counting the number of blocks
        for i in range(4):  # 4 feature resolution stages
            dim = (2 ** i) * feature_size
            self.stages.append(nn.Sequential(
                *[UX_Block(in_dim=dim,
                           spatial_dim=spatial_dim,
                           drop_path_rate=dp_rates[cur + j],
                           layer_scaling=layer_scaling)
                  for j in range(depths[i])]
            ))
            cur += depths[i]

        # 4 normalization layers
        layer_norm = partial(LayerNorm, eps=1e-6, channels_last=False)
        for i in range(4):
            dim = (2 ** i) * feature_size
            self.add_module(f"norm{i}", layer_norm(dim))

    def forward(self, x: Tensor):
        outputs = list()        # 记录所有stage的输出
        for i in range(4):
            x = self.down_samplers[i](x)
            x = self.stages[i](x)
            # normalize outputs
            y = getattr(self, f"norm{i}")(x)
            outputs.append(y)
        return outputs
