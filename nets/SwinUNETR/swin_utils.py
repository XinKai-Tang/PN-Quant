import math
from typing import Sequence, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F


def trunc_normal_(tensor: Tensor, 
                  mean: float = 0.0, 
                  std: float = 1.0, 
                  a: float = -2.0, 
                  b: float = 2.0):
    """ Tensor initialization with truncated normal distribution.
        * `tensor`: an n-dimensional `Tensor`.
        * `mean`: the mean of the normal distribution.
        * `std`: the standard deviation of the normal distribution.
        * `a`: the minimum cutoff value.
        * `b`: the maximum cutoff value.

        Reference: https://github.com/rwightman/pytorch-image-models
    """
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def window_partition(x: Tensor, win_size: Sequence[int]):
    ''' partition images into many windows
        * `x`: 3D[B,D,H,W,C] input images.
        * `win_size`: dimension of local windows.
    '''
    B, D, H, W, C = x.shape
    x = x.view(B,                 # 0
               D // win_size[0],  # 1
               win_size[0],       # 2
               H // win_size[1],  # 3
               win_size[1],       # 4
               W // win_size[2],  # 5
               win_size[2],       # 6
               C)                 # 7
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # 交换对应维度
    wins = x.view(-1, win_size[0] * win_size[1] * win_size[2], C)
    return wins     # B * n_w, wD * wH * wW, C


def window_reverse(wins: Tensor, 
                   win_size: Sequence[int],
                   shape: Sequence[int]):
    ''' reverse windows into images
        * `wins`: windows tensor [B*n_wins, win_D, win_H, win_W, C].
        * `win_size`: dimension of local windows.
        * `shape`: 3D[B,D,H,W] image dimension.
    '''
    B, D, H, W = shape
    x = wins.view(B,                 # 0
                  D // win_size[0],  # 1
                  H // win_size[1],  # 2
                  W // win_size[2],  # 3
                  win_size[0],       # 4
                  win_size[1],       # 5
                  win_size[2],       # 6
                  -1)                # 7
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()  # 交换对应维度
    imgs = x.view(B, D, H, W, -1)
    return imgs     # B, D, H, W, C


def compute_mask(shape: Sequence[int], 
                 win_size: Sequence[int],
                 shift_size: Sequence[int],
                 device: Union[str, torch.device] = None):
    ''' Computing region masks for windows
        * `shape`: dimension of a 3D[D,H,W] image.
        * `win_size`: size of local windows.
        * `shift_size`: shift size of local windows.
        * `device`: runtime device (cpu/gpu/etc).
    '''
    count = 0
    D, H, W = shape
    mask = torch.zeros((1, D, H, W, 1), device=device)
    d_slices = (slice(0, -win_size[0]),
                slice(-win_size[0], -shift_size[0]),
                slice(-shift_size[0], None))
    h_slices = (slice(0, -win_size[1]),
                slice(-win_size[1], -shift_size[1]),
                slice(-shift_size[1], None))
    w_slices = (slice(0, -win_size[2]),
                slice(-win_size[2], -shift_size[2]),
                slice(-shift_size[2], None))
    for d in d_slices:
        for h in h_slices:
            for w in w_slices:
                mask[:, d, h, w, :] = count
                count += 1
    # 使用与“划分特征图”相同的方法划分初始mask：
    mask_wins = window_partition(mask, win_size).squeeze(-1)
    attn_mask = mask_wins.unsqueeze(1) - mask_wins.unsqueeze(2)
    # 对于无需计算注意力的部分，在mask中填入负数（默认-100）：
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100))
    # 对于需要计算注意力的部分，在mask中填入零：
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    # 产生的mask会与特征图相加，从而降低特征图中无需计算注意力的区域的值：
    return attn_mask


def get_real_sizes(x_size: Sequence[int], 
                   win_size: Sequence[int], 
                   shift_size: Sequence[int]):
    ''' Ensure size of window is not larger than size of x. 
        * `x_size`: dimension of input features.
        * `win_size`: dimension of local windows.
        * `shift_size`: size of window shifting.
    '''
    real_win_size = list(win_size)
    real_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= win_size[i]:
            real_win_size[i] = x_size[i]
            real_shift_size[i] = 0
    return tuple(real_win_size), tuple(real_shift_size)


class MLP(nn.Module):
    ''' Multi-Layer Perceptron (MLP) '''

    def __init__(self,
                 mlp_dim: int,
                 hidden_dim: int,
                 act_layer: nn.Module = nn.GELU,
                 drop_rate: float = 0.0):
        ''' Args:
            * `mlp_dim`: dimension of the input and ouput.
            * `hidden_dim`: dimension of the hidden layer.
            * `act_layer`:  activation function layer.
            * `drop_rate`: dropout rate of the mlp.
        '''
        super(MLP, self).__init__()
        self.act = act_layer()
        self.drop = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(mlp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        y = self.drop(x)
        return y


class PatchEmbed(nn.Module):
    ''' 3D Patch Embedding Layer '''

    def __init__(self,
                 in_channels: int,
                 embed_dim: int = 48,
                 patch_size: Sequence[int] = (2, 2, 2),
                 norm_layer: nn.Module = nn.LayerNorm,
                 drop_rate: float = 0.0):
        ''' Args:
            * `in_channels`: dimension of input channels.
            * `embed_dim`: number of linear projection output channels.
            * `patch_size`: dimension of patch size.
            * `norm_layer`: normalization layer.
            * `drop_rate`: dropout rate.
        '''
        super(PatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels=in_channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        self.pos_drop = nn.Dropout(drop_rate)

    def forward(self, x: Tensor):
        # Reference: https://github.com/Project-MONAI/MONAI/
        B, C, D, H, W = x.shape
        # 对于不能整数倍划分的维度，需要进行填充操作：
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        y = self.proj(x)
        _, _, D, H, W = y.shape             # B, embed_dim, wD, wH, wW
        y = y.flatten(2).transpose(1, 2)    # B, wD*wH*wW, embed_dim
        if self.norm is not None:
            y = self.norm(y)
        y = y.view(-1, D, H, W, self.embed_dim)
        y = self.pos_drop(y)                # B, wD, wH, wW, embed_dim
        return y


class PatchMerging(nn.Module):
    ''' 3D Patch Merging Layer '''

    def __init__(self,
                 in_channels: int,
                 norm_layer: nn.Module = nn.LayerNorm):
        ''' Args:
            * `in_channels`: dimension of input channels.
            * `norm_layer`: normalization layer.
        '''
        super(PatchMerging, self).__init__()
        self.norm = norm_layer(8*in_channels)
        self.reduction = nn.Linear(8 * in_channels, 2 * in_channels, bias=False)

    def forward(self, x: Tensor):
        # Reference: https://github.com/Project-MONAI/MONAI/
        B, D, H, W, C = x.shape
        if (W % 2 == 1) or (H % 2 == 1) or (D % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B, D/2, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.concat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        y = self.reduction(x)
        return y


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


class WindowAttention(nn.Module):
    ''' Window-based Multi-head Self-Attention Module (W-MSA) '''

    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 win_size: Sequence[int] = (7, 7, 7),
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        ''' Args:
            * `in_channels`: number of input channels.
            * `num_heads`: number of attention heads.
            * `win_size`: local window size.
            * `qkv_bias`: whether add a learnable bias to query, key and value.
            * `attn_drop`: attention dropout rate.
            * `proj_drop`: dropout rate of output.
        '''
        super(WindowAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim ** -0.5

        # 定义一个记录相对位置偏置的变量表：
        shape = (2*win_size[0] - 1) * (2*win_size[1] - 1) * (2*win_size[2] - 1)
        self.rel_pos_bias_tab = nn.Parameter(torch.zeros(shape, num_heads))
        trunc_normal_(self.rel_pos_bias_tab, std=0.02)

        # 获取窗口内每个标记的成对相对位置索引：
        coords = torch.stack(torch.meshgrid(
            [torch.arange(s) for s in win_size], indexing = 'ij'
        )).flatten(1)                                           # 3, Wd * Wh * Ww
        rel_coords = coords[:, :, None] - coords[:, None, :]    # 3, Wd*Wh*Wd, Wd*Wh*Ww
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()   # Wd*Wh*Wd, Wd*Wh*Ww, 3
        rel_coords[:, :, 0] += win_size[0] - 1
        rel_coords[:, :, 1] += win_size[1] - 1
        rel_coords[:, :, 2] += win_size[2] - 1
        rel_coords[:, :, 0] *= (2 * win_size[1] - 1) * (2 * win_size[2] - 1)
        rel_coords[:, :, 1] *= (2 * win_size[2] - 1)
        rel_pos_index = rel_coords.sum(-1)                      # Wd*Wh*Wd, Wd*Wh*Ww
        self.register_buffer("rel_pos_index", rel_pos_index)    # forward()会用到该Buffer

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, mask: Tensor = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # 3, B, n_h, N, C/n_h
        q, k, v = qkv[0], qkv[1], qkv[2]                # q/k/v: B, n_h, N, C/n_h
        
        attn = (self.scale * q) @ k.transpose(-2, -1)   # q × k.T / √d_k
        rel_pos_bias = self.rel_pos_bias_tab[
            self.rel_pos_index.clone()[:N, :N].reshape(-1)
        ].reshape(N, N, -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_pos_bias.unsqueeze(0)         # B, n_h, N, N
        if mask is not None:
            nW = mask.shape[0]      # num of Windows
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B // nW, nW, self.num_heads, N, N)
            attn = (attn + mask).view(-1, self.num_heads, N, N)
        
        attn = self.attn_drop(self.softmax(attn))           # B, n_h, N, N
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     # B, N, n_h * C/n_h
        x = self.proj_drop(self.proj(x))
        return x