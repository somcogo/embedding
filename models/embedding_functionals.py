import math

import torch
from torch import nn
import torch.nn.functional as F

MODE_NAMES = {'embedding': 'embedding_weights',
              'residual': 'embedding_residual',
              'vanilla': 'vanilla',
              'fedbn':'fedbn'}

class WeightGenerator(nn.Module):
    def __init__(self, gen_dim, gen_hidden_layer, out_channels, gen_depth=None, target='const', **kwargs):
        super().__init__()
        self.gen_depth = gen_depth

        bound_w_l = - 1 / math.sqrt(gen_dim)
        bound_w_r = 1 / math.sqrt(gen_dim)
        bound_b_l = 1 - 1 / math.sqrt(gen_dim) if target == 'one' else - 1 / math.sqrt(gen_dim)
        bound_b_r = 1 + 1 / math.sqrt(gen_dim) if target == 'one' else 1 / math.sqrt(gen_dim)
        if gen_depth == 1:
            self.lin1 = nn.Linear(in_features=gen_dim, out_features=out_channels)

            nn.init.uniform_(self.lin1.weight, a=bound_w_l, b=bound_w_r)
            nn.init.uniform_(self.lin1.bias, a=bound_b_l, b=bound_b_r)
        if gen_depth == 2:
            self.lin1 = nn.Linear(in_features=gen_dim, out_features=gen_hidden_layer)
            self.lin2 = nn.Linear(in_features=gen_hidden_layer, out_features=out_channels)

            nn.init.uniform_(self.lin2.weight, a=bound_w_l, b=bound_w_r)
            nn.init.uniform_(self.lin2.bias, a=bound_b_l, b=bound_b_r)
    
    def forward(self, x):
        x = x.to(torch.float)
        if self.gen_depth == 1:
            out = self.lin1(x)
        if self.gen_depth == 2:
            x = self.lin1(x)
            x = nn.functional.relu(x)
            out = self.lin2(x)
        return out
    
class CombWeightGenerator(nn.Module):
    def __init__(self, gen_dim, gen_hidden_layer, out_chans, gen_depth=None, targets=['const', 'one'], **kwargs):
        super().__init__()
        self.gen_depth = gen_depth
        bound_w_l, bound_w_r, bound_b_l, bound_b_r = [], [], [], []
        for target in targets:
            bound_w_l.append(- 1 / math.sqrt(gen_dim))
            bound_w_r.append(1 / math.sqrt(gen_dim))
            bound_b_l.append(1 - 1 / math.sqrt(gen_dim) if target == 'one' else - 1 / math.sqrt(gen_dim))
            bound_b_r.append(1 + 1 / math.sqrt(gen_dim) if target == 'one' else 1 / math.sqrt(gen_dim))
        if gen_depth == 1:
            self.lin1 = nn.Linear(in_features=gen_dim, out_features=out_chans[0])

            nn.init.uniform_(self.lin1.weight, a=bound_w_l[0], b=bound_w_r[0])
            nn.init.uniform_(self.lin1.bias, a=bound_b_l[0], b=bound_b_r[0])
        if gen_depth == 2:
            self.lin1 = nn.Linear(in_features=gen_dim, out_features=gen_hidden_layer)
            self.lin2s = nn.ModuleList([])
            for ndx in range(len(out_chans)):
                lin = nn.Linear(in_features=gen_hidden_layer, out_features=out_chans[ndx])
                nn.init.uniform_(lin.weight, a=bound_w_l[ndx], b=bound_w_r[ndx])
                nn.init.uniform_(lin.bias, a=bound_b_l[ndx], b=bound_b_r[ndx])
                self.lin2s.append(lin)
    
    def forward(self, x):
        x = x.to(torch.float)
        if self.gen_depth == 1:
            out = self.lin1(x)
        if self.gen_depth == 2:
            x = self.lin1(x)
            x = nn.functional.relu(x)
            out = []
            for lin in self.lin2s:
                out.append(lin(x))
        return out
    
class Conv2d_emb(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gen_size=1, gen_affine=False, device=None, **kwargs):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.gen_affine = gen_affine
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        self.weight = nn. Parameter(torch.empty(
            (out_channels, in_channels // groups, kernel_size, kernel_size), device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

        if gen_size == 1:
            gen_weight_const_size = kernel_size**2
            gen_weight_affine_size = kernel_size**4 if gen_affine else None
            gen_bias_const_size = 1 if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None
        elif gen_size == 2:
            gen_weight_const_size = in_channels // groups * kernel_size**2
            gen_weight_affine_size = in_channels // groups * kernel_size**4 if gen_affine else None
            gen_bias_const_size = out_channels if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None

        self.weight_const_generator = WeightGenerator(gen_weight_const_size, target='const', **kwargs)
        self.weight_affine_generator = WeightGenerator(gen_weight_affine_size, target='affine', **kwargs) if gen_affine else None
        self.bias_const_generator = WeightGenerator(gen_bias_const_size, target='const', **kwargs) if bias else None
        self.bias_affine_generator = WeightGenerator(gen_bias_affine_size, target='affine', **kwargs) if bias and gen_affine else None

    def forward(self, x, emb):
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        if self.gen_affine:
            weight_affine = self.weight_affine_generator(emb)
            weight_affine = weight_affine.reshape(-1, self.kernel_size, self.kernel_size, self.kernel_size, self.kernel_size).expand(*weight.shape, self.kernel_size, self.kernel_size)
            weight = torch.einsum('ijklmn,ijmn->ijkl', weight_affine, weight)
            if bias is not None:
                bias_affine = self.bias_affine_generator(emb)
                bias_affine = bias_affine.expand(bias.shape)
                bias = bias_affine*bias
        weight_const = self.weight_const_generator(emb).reshape(-1, self.kernel_size, self.kernel_size).expand(weight.shape)
        weight = weight_const + weight
        if bias is not None:
            bias_const = self.bias_const_generator(emb).expand(bias.shape)
            bias = bias_const + bias

        x = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return x
    
class ConvTranspose2d_emb(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, dilation=1, device=None, gen_size=1, gen_affine=False, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.gen_affine = gen_affine
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn. Parameter(torch.empty(
            (in_channels, out_channels // groups, kernel_size, kernel_size), device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

        if gen_size == 1:
            gen_weight_const_size = kernel_size**2
            gen_weight_affine_size = kernel_size**4 if gen_affine else None
            gen_bias_const_size = 1 if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None
        elif gen_size == 2:
            gen_weight_const_size = out_channels // groups * kernel_size**2
            gen_weight_affine_size = out_channels // groups * kernel_size**4 if gen_affine else None
            gen_bias_const_size = out_channels if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None

        self.weight_const_generator = WeightGenerator(gen_weight_const_size, target='const', **kwargs)
        self.weight_affine_generator = WeightGenerator(gen_weight_affine_size, target='affine', **kwargs) if gen_affine else None
        self.bias_const_generator = WeightGenerator(gen_bias_const_size, target='const', **kwargs) if bias else None
        self.bias_affine_generator = WeightGenerator(gen_bias_affine_size, target='affine', **kwargs) if bias and gen_affine else None


    def forward(self, x, emb):
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        if self.gen_affine:
            weight_affine = self.weight_affine_generator(emb)
            weight_affine = weight_affine.reshape(-1, self.kernel_size, self.kernel_size, self.kernel_size, self.kernel_size).expand(*weight.shape, self.kernel_size, self.kernel_size)
            weight = torch.einsum('ijklmn,ijmn->ijkl', weight_affine, weight)
            if bias is not None:
                bias_affine = self.bias_affine_generator(emb)
                bias_affine = bias_affine.expand(bias.shape)
                bias = bias_affine*bias
        weight_const = self.weight_const_generator(emb).reshape(-1, self.kernel_size, self.kernel_size).expand(weight.shape)
        weight = weight_const + weight
        if bias is not None:
            bias_const = self.bias_const_generator(emb).expand(bias.shape)
            bias = bias_const + bias

        x = F.conv_transpose2d(x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
        return x

class Linear_emb(nn.Module):
    def __init__(self, in_channels, out_channels, gen_size, gen_affine=False, bias=True, device=None, **kwargs):
        super().__init__()
        self.gen_affine = gen_affine

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels), device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if gen_size == 1:
            gen_weight_const_size = 1
            gen_weight_affine_size = 1 if gen_affine else None
            gen_bias_const_size = 1 if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None
        elif gen_size == 2:
            gen_weight_const_size = in_channels
            gen_weight_affine_size = in_channels if gen_affine else None
            gen_bias_const_size = 1 if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None

        self.weight_const_generator = WeightGenerator(gen_weight_const_size, target='const', **kwargs)
        self.weight_affine_generator = WeightGenerator(gen_weight_affine_size, target='affine', **kwargs) if gen_affine else None
        self.bias_const_generator = WeightGenerator(gen_bias_const_size, target='const', **kwargs) if bias else None
        self.bias_affine_generator = WeightGenerator(gen_bias_affine_size, target='affine', **kwargs) if bias and gen_affine else None

    def forward(self, x, emb):
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        if self.gen_affine:
            weight_affine = self.weight_affine_generator(emb).expand(weight.shape)
            weight = weight_affine*weight
            if bias is not None:
                bias_affine = self.bias_affine_generator(emb).expand(bias.shape)
                bias = bias_affine*bias
        weight_const = self.weight_const_generator(emb).expand(weight.shape)
        weight = weight+weight_const
        if bias is not None:
            bias_const = self.bias_const_generator(emb).expand(bias.shape)
            bias = bias + bias_const
            
        x = F.linear(x, weight=weight, bias=bias)
        return x
    
class BatchNorm2d_emb(nn.Module):
    def __init__(self, num_features,eps=1e-05, momentum=0.1, gen_affine=False, device=None, dtype=None, gen_size=2, **kwargs):
        super().__init__()
        self.gen_affine = gen_affine
        self.momentum = momentum
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(num_features, device=device))
        self.bias = nn.Parameter(torch.empty(num_features, device=device))
        self.register_buffer('running_mean', torch.zeros(num_features, device=device))
        self.register_buffer('running_var', torch.ones(num_features, device=device))

        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        
        if gen_size == 1:
            gen_weight_const_size = 1
            gen_weight_affine_size = 1 if gen_affine else None
            gen_bias_const_size = 1 
            gen_bias_affine_size = 1 if gen_affine else None
        if gen_size == 2:
            gen_weight_const_size = num_features
            gen_weight_affine_size = num_features if gen_affine else None
            gen_bias_const_size = num_features
            gen_bias_affine_size = num_features if gen_affine else None

        self.weight_const_generator = WeightGenerator(out_channels=gen_weight_const_size, target='const', **kwargs)
        self.weight_affine_generator = WeightGenerator(out_channels=gen_weight_affine_size, target='affine', **kwargs) if gen_affine else None
        self.bias_const_generator = WeightGenerator(out_channels=gen_bias_const_size, target='const', **kwargs)
        self.bias_affine_generator = WeightGenerator(out_channels=gen_bias_affine_size, target='affine', **kwargs) if gen_affine else None

    def forward(self, x, emb):
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype)
        if self.gen_affine:
            weight_affine = self.weight_affine_generator(emb).expand(weight.shape)
            weight = weight_affine*weight
            bias_affine = self.bias_affine_generator(emb).expand(bias.shape)
            bias = bias_affine*bias
        weight_const = self.weight_const_generator(emb).expand(weight.shape)
        weight = weight + weight_const
        bias_const = self.bias_const_generator(emb).expand(bias.shape)
        bias = bias + bias_const

        x = F.batch_norm(x, running_mean=self.running_mean, running_var=self.running_var, weight=weight, bias=bias, training=self.training, momentum=self.momentum, eps=self.eps)
        return x
    
class BatchNorm2d_emb_replace(nn.Module):
    def __init__(self, num_features,eps=1e-05, momentum=0.1, gen_affine=False, device=None, comb_gen=False, **kwargs):
        super().__init__()
        self.gen_affine = gen_affine
        self.momentum = momentum
        self.eps = eps
        self.comb_gen = comb_gen
        self.register_buffer('running_mean', torch.zeros(num_features, device=device))
        self.register_buffer('running_var', torch.ones(num_features, device=device))

        gen_w_size = num_features
        gen_b_size = num_features

        if comb_gen:
            self.w_b_generator = CombWeightGenerator(out_chans=[gen_w_size, gen_b_size], targets=['one', 'zero'], **kwargs)
        else:
            self.weight_const_generator = WeightGenerator(out_channels=gen_w_size, target='one', **kwargs)
            self.bias_const_generator = WeightGenerator(out_channels=gen_b_size, target='zero', **kwargs)

    def forward(self, x, emb):
        if self.comb_gen:
            weight, bias = self.w_b_generator(emb)
            weight = weight.to(x.dtype)
            bias = bias.to(x.dtype)
        else:
            weight = self.weight_const_generator(emb).to(x.dtype)
            bias = self.bias_const_generator(emb).to(x.dtype)

        x = F.batch_norm(x, running_mean=self.running_mean, running_var=self.running_var, weight=weight, bias=bias, training=self.training, momentum=self.momentum, eps=self.eps)
        return x
    
class InstanceNorm2d_emb(nn.Module):
    def __init__(self, num_features, eps, momentum, gen_affine=False, bias=True, device=None, **kwargs):
        super().__init__()
        self.gen_affine = gen_affine
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.empty(num_features, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_features, device=device))
        else:
            self.register_parameter('bias', None)

        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        gen_weight_const_size = 1
        gen_weight_affine_size = 1 if gen_affine else None
        gen_bias_const_size = 1 if bias else None
        gen_bias_affine_size = 1 if bias and gen_affine else None

        self.weight_const_generator = WeightGenerator(gen_weight_const_size, target='const', **kwargs)
        self.weight_affine_generator = WeightGenerator(gen_weight_affine_size, target='affine', **kwargs) if gen_affine else None
        self.bias_const_generator = WeightGenerator(gen_bias_const_size, target='const', **kwargs) if bias else None
        self.bias_affine_generator = WeightGenerator(gen_bias_affine_size, target='affine', **kwargs) if bias and gen_affine else None

    def forward(self, x, emb):
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        if self.gen_affine:
            weight_affine = self.weight_affine_generator(emb).expand(weight.shape)
            weight = weight_affine*weight
            if bias is not None:
                bias_affine = self.bias_affine_generator(emb).expand(bias.shape)
                bias = bias_affine*bias
        weight_const = self.weight_const_generator(emb).expand(weight.shape)
        weight = weight+weight_const
        if bias is not None:
            bias_const = self.bias_const_generator(emb).expand(bias.shape)
            bias = bias + bias_const
        
        x = F.instance_norm(x, weight=weight, bias=bias, momentum=self.momentum, eps=self.eps)
        return x
    
class GeneralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mode='vanilla', stride=1, padding=0, dilation=1, groups=1, bias=True, device=None, **kwargs):
        super().__init__()
        self.mode = mode
        if mode == MODE_NAMES['embedding']:
            self.conv = Conv2d_emb(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, device=device, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, device=device)

    def forward(self, x, emb):
        if self.mode == MODE_NAMES['embedding']:
            out = self.conv(x, emb)
        else:
            out = self.conv(x)
        return out
    
class GeneralConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  mode='vanilla', stride=1, padding=0, groups=1, bias=True, dilation=1, device=None, **kwargs):
        super().__init__()
        self.mode = mode
        if mode == MODE_NAMES['embedding']:
            self.transposed_conv = ConvTranspose2d_emb(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation, device=device, **kwargs)
        else:
            self.transposed_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation, device=device)

    def forward(self, x, emb):
        if self.mode == MODE_NAMES['embedding']:
            out = self.transposed_conv(x, emb)
        else:
            out = self.transposed_conv(x)
        return out
    
class GeneralLinear(nn.Module):
    def __init__(self, in_channels, out_channels,  mode='vanilla', bias=True, device=None, **kwargs):
        super().__init__()
        self.mode = mode
        if mode == MODE_NAMES['embedding']:
            self.linear = Linear_emb(in_features=in_channels, out_features=out_channels, bias=bias, device=device, **kwargs)
        else:
            self.linear = nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias, device=device)

    def forward(self, x, emb):
        if self.mode == MODE_NAMES['embedding']:
            out = self.linear(x, emb)
        else:
            out = self.linear(x)
        return out
    
class GeneralBatchNorm2d(nn.Module):
    def __init__(self, num_features,  mode='vanilla', eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, use_repl_bn=False, **kwargs):
        super().__init__()
        self.mode = mode

        if use_repl_bn:
            self.batch_norm = BatchNorm2d_emb_replace(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, **kwargs)
        elif mode == MODE_NAMES['embedding'] or mode == MODE_NAMES['fedbn']:
            self.batch_norm = BatchNorm2d_emb(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, **kwargs)
        else:
            self.batch_norm = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device)
    
    def forward(self, x, emb):
        if self.mode == MODE_NAMES['embedding'] or self.mode == MODE_NAMES['fedbn']:
            out = self.batch_norm(x, emb)
        else:
            out = self.batch_norm(x)
        return out
    
class BatchNorm2d_noemb(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device)
    
    def forward(self, x, emb):
        out = self.batch_norm(x)
        return out
    
class GeneralInstanceNorm2d(nn.Module):
    def __init__(self, num_features,  mode='vanilla', eps=1e-05, momentum=0.1, device=None, dtype=None, **kwargs):
        super().__init__()
        self.mode = mode
        if mode == MODE_NAMES['embedding']:
            self.instance_norm = InstanceNorm2d_emb(num_features=num_features, eps=eps, momentum=momentum, device=device, **kwargs)
        else:
            self.instance_norm = nn.InstanceNorm2d(num_features=num_features, eps=eps, momentum=momentum, device=device)
    
    def forward(self, x, emb):
        if self.mode == MODE_NAMES['embedding']:
            out = self.instance_norm(x, emb)
        else:
            out = self.instance_norm(x)
        return out
    
class GeneralAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.adaptive_avg_pool= nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x, emb):
        return self.adaptive_avg_pool(x)
    
class GeneralReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x, emb):
        return self.relu(x)