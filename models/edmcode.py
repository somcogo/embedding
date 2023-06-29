# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch.nn.functional import silu

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x, hyper_a=None, hyper_b=None):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        if hyper_a is not None:
            if len(hyper_a.shape) == 4:
                w = torch.einsum('klmn,ijmn->ijkl', hyper_a, w)
            else:
                w = torch.einsum('jklmn,ijmn->ijkl', hyper_a, w)
        if hyper_b is not None:
            w = w + torch.reshape(hyper_b, w.shape).to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None, use_hypnns=False, version=None,
        ffwrd=None, lightweight=False, affine_mode=False, ffwrd_a=None,
        medium_ffwrd=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.lightweight = lightweight
        self.use_hypnns = use_hypnns
        self.affine_mode = affine_mode
        self.medium_ffwrd = medium_ffwrd

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        if use_hypnns:
            if ffwrd is not None:
                self.ffwrd0 = ffwrd
                self.ffwrd1 = ffwrd
            elif medium_ffwrd:
                self.ffwrd0 = FeedForward(emb_channels, 64, in_channels*3*3, version=version)
                self.ffwrd1 = FeedForward(emb_channels, 64, out_channels*3*3, version=version)
            elif lightweight:
                self.ffwrd0 = FeedForward(emb_channels, 64, 3*3, version=version)
                self.ffwrd1 = FeedForward(emb_channels, 64, 3*3, version=version)
            else:
                self.ffwrd0 = FeedForward(emb_channels, 64, in_channels*out_channels*3*3, version=version)
                self.ffwrd1 = FeedForward(emb_channels, 64, out_channels*out_channels*3*3, version=version)

            if affine_mode:
                if ffwrd_a is not None:
                    self.ffwrd_a0 = ffwrd_a
                    self.ffwrd_a1 = ffwrd_a
                elif medium_ffwrd:
                    self.ffwrd_a0 = FeedForward(emb_channels, 64, in_channels*9*9, version=version)
                    self.ffwrd_a1 = FeedForward(emb_channels, 64, out_channels*9*9, version=version)
                else:
                    self.ffwrd_a0 = FeedForward(emb_channels, 64, 9*9, version=version)
                    self.ffwrd_a1 = FeedForward(emb_channels, 64, 9*9, version=version)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        params = self.affine(emb)
        params = params.repeat(x.shape[0]).view(x.shape[0], -1).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.use_hypnns:
            b0 = self.ffwrd0(emb)
            b1 = self.ffwrd1(emb)
            if self.medium_ffwrd:
                b0 = b0.repeat([self.out_channels])
                b1 = b1.repeat([self.out_channels])
            elif self.lightweight:
                b0 = b0.repeat([self.in_channels*self.out_channels])
                b1 = b1.repeat([self.out_channels*self.out_channels])

            if self.affine_mode:
                a0 = self.ffwrd_a0(emb)
                a1 = self.ffwrd_a1(emb)
                if self.medium_ffwrd:
                    a0 = a0.reshape(self.in_channels, 3, 3, 3, 3)
                    a1 = a1.reshape(self.out_channels, 3, 3, 3, 3)
                else:
                    a0 = a0.reshape(3, 3, 3, 3)
                    a1 = a1.reshape(3, 3, 3, 3)
            else:
                a0, a1 = None, None
        else:
            b0, b1, a0, a1 = None, None, None, None

        orig = x
        x = self.conv0(silu(self.norm0(x)), hyper_a=a0, hyper_b=b0)

        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training), hyper_a=a1, hyper_b=b1)
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

class FeedForward(torch.nn.Module):
    def __init__(self, in_channels, hidden_layer, out_channels, version=None):
        super().__init__()
        self.version = version
        
        if version == 1:
            self.lin1 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        if version == 2:
            self.lin1 = torch.nn.Linear(in_features=in_channels, out_features=hidden_layer)
            self.lin2 = torch.nn.Linear(in_features=hidden_layer, out_features=out_channels)
    
    def forward(self, x):
        if self.version == 1:
            out = self.lin1(x)
        if self.version == 2:
            x = self.lin1(x)
            x = torch.nn.functional.relu(x)
            out = self.lin2(x)
        return out