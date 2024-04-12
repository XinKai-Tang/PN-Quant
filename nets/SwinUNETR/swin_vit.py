from typing import Sequence, Union

import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils import checkpoint

from .swin_utils import *


class SwinTransformer(nn.Module):
    ''' 3D Shifted windows Transformer (Swin Transformer)
    * Authors: Ze Liu, et al (2021)
    * Paper: Hierarchical Vision Transformer using Shifted Windows
    * Link: https://arxiv.org/pdf/2103.14030.pdf
    '''

    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 patch_size: Sequence[int],
                 window_size: Sequence[int],
                 depths: Sequence[int],
                 num_heads: Sequence[int],
                 dropout_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 mlp_ratio: float = 4.0,
                 norm_layer: nn.Module = nn.LayerNorm,
                 qkv_bias: bool = True,
                 patch_norm: bool = False,
                 end_norm: bool = True,
                 use_checkpoint: bool = False):
        ''' Args:
            * `in_channels`: dimension of input channels.
            * `embed_dim`: number of linear projection output channels.
            * `patch_size`: dimension of patch size.
            * `window_size`: dimension of local window size.
            * `depths`: number of transformers in each stage.
            * `num_heads`: number of attention heads in each stage.
            * `dropout_rate`: (SwinTransformer) dropout rate.
            * `attn_drop_rate`: (SwinTransformer) attention dropout rate.
            * `drop_path_rate`: (SwinTransformer) stochastic depth rate.
            * `mlp_ratio`: ratio of mlp hidden dim to embedding dim.
            * `norm_layer`: normalization layer.
            * `qkv_bias`: whether adding a learnable bias to query, key and value.
            * `patch_norm`: whether adding normalization after patch embedding.
            * `end_norm`: whether normalizing output features in each stage.
            * `use_checkpoint`: use gradient checkpointing to save memory.
        '''
        super(SwinTransformer, self).__init__()
        self.end_norm = end_norm
        self.num_layers = len(depths)
        self.num_feature = int(2 ** (self.num_layers - 1) * embed_dim)
        # calculate stochastic depth: 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embed = PatchEmbed(in_channels=in_channels,
                                      embed_dim=embed_dim,
                                      patch_size=patch_size,
                                      norm_layer=norm_layer if patch_norm else None,
                                      drop_rate=dropout_rate)
        
        # bulid swin-transformer blocks:
        self.layers = nn.ModuleList([
            BasicBlock(
                in_channels=int(2**i * embed_dim),
                num_blocks=depths[i],
                num_heads=num_heads[i],
                win_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=dropout_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[:i+1])],
                norm_layer=norm_layer,
                downsample=PatchMerging,
                use_checkpoint=use_checkpoint
            ) for i in range(self.num_layers)
        ])

    def norm_output(self, x: Tensor):
        ''' normalize output '''
        if self.end_norm:
            x = F.layer_norm(x, [x.shape[-1],])     # B, D, H, W, C
        return x

    def forward(self, x: Tensor):
        outputs = list()        # 记录所有stage的输出
        x = self.patch_embed(x)
        y = self.norm_output(x)
        outputs.append(y)
        for stage in self.layers:
            x = stage(x.contiguous())
            y = self.norm_output(x)
            outputs.append(y)
        return y, outputs


class BasicBlock(nn.Module):
    ''' 3D SwinTransformer Blocks in Each Stage '''

    def __init__(self,
                 in_channels: int,
                 num_blocks: int,
                 num_heads: int,
                 win_size: Sequence[int],
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 drop_rate: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: Union[float, list] = 0.0,
                 norm_layer: nn.Module = nn.LayerNorm,
                 downsample: nn.Module = None,
                 use_checkpoint: bool = False):
        ''' Args:
            * `in_channels`: dimension of input channels.
            * `num_blocks`: number of transformers in each stage.
            * `num_heads`: number of attention heads in each stage.
            * `win_size`: dimension of local window size.
            * `mlp_ratio`: ratio of mlp hidden dim to embedding dim.
            * `qkv_bias`: whether adding a learnable bias to query, key and value.
            * `drop_rate`: dropout rate.
            * `attn_drop`: attention dropout rate.
            * `drop_path`: stochastic depth rate.
            * `norm_layer`: normalization layer.
            * `downsample`: downsample layer at the end of the block.
            * `use_checkpoint`: use gradient checkpointing to save memory.
        '''
        super(BasicBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.win_size = win_size
        self.shift_size = tuple(i//2 for i in win_size)
        zero_shift = tuple(0 for i in win_size)

        # build Swin Transformer blocks：
        self.blocks = nn.ModuleList([
            SwinTrBlock(
                in_channels=in_channels,
                num_heads=num_heads,
                win_size=win_size,
                shift_size=zero_shift if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(num_blocks)
        ])

        # patch merging:
        if downsample is not None:
            self.downsample = downsample(in_channels=in_channels,
                                         norm_layer=norm_layer)
        else:
            self.downsample = None     

    def forward(self, x: Tensor):
        B, D, H, W, C = x.shape
        win_size, shift_size = get_real_sizes(x_size=(D, H, W),
                                              win_size=self.win_size,
                                              shift_size=self.shift_size)
        pD = int(np.ceil(D / win_size[0])) * win_size[0]
        pH = int(np.ceil(H / win_size[1])) * win_size[1]
        pW = int(np.ceil(W / win_size[2])) * win_size[2]
        attn_mask = compute_mask(shape=[pD, pH, pW], 
                                 win_size=win_size, 
                                 shift_size=shift_size,
                                 device=x.device)
        
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x, attn_mask)
        else:
            for blk in self.blocks:
                x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTrBlock(nn.Module):
    ''' 3D Shifted windows Transformer Block '''

    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 win_size: Sequence[int],
                 shift_size: Sequence[int],
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 drop_rate: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.0,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm):
        ''' Args:
            * `in_channels`: number of input channels.
            * `num_heads`: number of attention heads.
            * `win_size`: local window size.
            * `shift_size`: window shift size.
            * `mlp_ratio`: ratio of mlp hidden dim to embedding dim.
            * `qkv_bias`: whether add a learnable bias to query, key and value.
            * `drop_rate`: dropout rate.
            * `attn_drop`: attention dropout rate.
            * `drop_path`: stochastic depth rate.
            * `act_layer`: activation layer of mlp.
            * `norm_layer`: normalization layer.
        '''
        super(SwinTrBlock, self).__init__()
        self.win_size = win_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(in_channels)
        self.attn = WindowAttention(in_channels=in_channels,
                                    num_heads=num_heads,
                                    win_size=win_size,
                                    qkv_bias=qkv_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=drop_rate)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(in_channels)
        self.mlp = MLP(mlp_dim=in_channels,
                       hidden_dim=int(in_channels * mlp_ratio),
                       act_layer=act_layer,
                       drop_rate=drop_rate)

    def forward(self, x: Tensor, mask: Tensor):
        shortcut = x        # 暂存处理之前的x
        B, D, H, W, C = x.shape
        x = self.norm1(x)

        win_size, shift_size = get_real_sizes(x_size=(D, H, W),
                                              win_size=self.win_size,
                                              shift_size=self.shift_size)
        x = F.pad(input=x, pad=(
            0, 0,
            0, (win_size[2] - W % win_size[2]) % win_size[2],
            0, (win_size[1] - H % win_size[1]) % win_size[1],
            0, (win_size[0] - D % win_size[0]) % win_size[0],
        ))  # 填充x使输入x能够完全被窗口划分
        _, pD, pH, pW, _ = x.shape  # 获取填充后的维度

        # cyclic shift
        if any(i > 0 for i in shift_size):
            # 如果需要移动x，则计算移动后的x和注意力机制中的mask：
            shift_x = torch.roll(shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                                 input=x, dims=(1, 2, 3))
            attn_mask = mask
        else:
            shift_x, attn_mask = x, None
        wins = window_partition(shift_x, win_size)  # B * n_w, wD * wH * wW, C

        # (Shifted) Window-based Multi-heads Self-Attention
        attn_wins = self.attn(wins, mask=attn_mask)  # W-MSA / SW-MSA
        attn_wins = attn_wins.view(-1, win_size[0], win_size[1], 
                                   win_size[2], C)   # merge windows

        # reverse cyclic shift
        shift_x = window_reverse(attn_wins, win_size, shape=(B, pD, pH, pW))
        if any(i > 0 for i in shift_size):
            # 如果之前移动过x，那么接下来需要计算还原后的x：
            x = torch.roll(shift_x, shifts=shift_size, dims=(1, 2, 3))
        else:
            x = shift_x
        x = x[:, :D, :H, :W, :].contiguous()    # 去除填充，还原图像
        
        x = shortcut + self.drop_path(x)                    # 第一次残差相加
        x = x + self.drop_path(self.mlp(self.norm2(x)))     # 第二次残差相加
        return x
