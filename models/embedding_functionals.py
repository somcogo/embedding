import math

import torch
from torch import nn
import torch.nn.functional as F

MODE_NAMES = {'embedding': 'embedding_weights',
              'residual': 'embedding_residual',
              'vanilla': 'vanilla'}

class WeightGenerator(nn.Module):
    def __init__(self, emb_dim, hidden_layer, out_channels, depth=None, target='const'):
        super().__init__()
        self.depth = depth
        if target == 'const':
            init = nn.init.zeros_
        else:
            init = nn.init.ones_
        if depth == 1:
            self.lin1 = nn.Linear(in_features=emb_dim, out_features=out_channels)
            nn.init.zeros_(self.lin1.weight)
            init(self.lin1.bias)
        if depth == 2:
            self.lin1 = nn.Linear(in_features=emb_dim, out_features=hidden_layer)
            self.lin2 = nn.Linear(in_features=hidden_layer, out_features=out_channels)
            nn.init.zeros_(self.lin1.weight)
            nn.init.zeros_(self.lin1.bias)
            nn.init.zeros_(self.lin2.weight)
            init(self.lin2.bias)
    
    def forward(self, x):
        x = x.to(torch.float)
        if self.depth == 1:
            out = self.lin1(x)
        if self.depth == 2:
            x = self.lin1(x)
            x = nn.functional.relu(x)
            out = self.lin2(x)
        return out

class Conv2d_emb(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, emb_dim, size, gen_depth=2, gen_affine=False, gen_hidden_layer=64, stride=1, padding=0, dilation=1, groups=1, bias=True, device=None):
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

        if size == 1:
            gen_weight_const_size = kernel_size**2
            gen_weight_affine_size = kernel_size**4 if gen_affine else None
            gen_bias_const_size = 1 if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None
        elif size == 2:
            gen_weight_const_size = in_channels // groups * kernel_size**2
            gen_weight_affine_size = in_channels // groups * kernel_size**4 if gen_affine else None
            gen_bias_const_size = out_channels if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None

        self.weight_const_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_weight_const_size, gen_depth, target='const')
        self.weight_affine_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_weight_affine_size, gen_depth, target='affine') if gen_affine else None
        self.bias_const_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_bias_const_size, gen_depth, target='const') if bias else None
        self.bias_affine_generator = WeightGenerator(emb_dim,  gen_hidden_layer, gen_bias_affine_size, gen_depth, target='affine') if bias and gen_affine else None

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
    def __init__(self, in_channels, out_channels, kernel_size, emb_dim, size, gen_depth=2, gen_affine=False, gen_hidden_layer=64, stride=1, padding=0, groups=1, bias=True, dilation=1, device=None):
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

        if size == 1:
            gen_weight_const_size = kernel_size**2
            gen_weight_affine_size = kernel_size**4 if gen_affine else None
            gen_bias_const_size = 1 if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None
        elif size == 2:
            gen_weight_const_size = out_channels // groups * kernel_size**2
            gen_weight_affine_size = out_channels // groups * kernel_size**4 if gen_affine else None
            gen_bias_const_size = out_channels if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None

        self.weight_const_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_weight_const_size, gen_depth, target='const')
        self.weight_affine_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_weight_affine_size, gen_depth, target='affine') if gen_affine else None
        self.bias_const_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_bias_const_size, gen_depth, target='const') if bias else None
        self.bias_affine_generator = WeightGenerator(emb_dim,  gen_hidden_layer, gen_bias_affine_size, gen_depth, target='affine') if bias and gen_affine else None


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
    def __init__(self, in_channels, out_channels, emb_dim, size, gen_depth=2, gen_affine=False, gen_hidden_layer=64, bias=True, device=None):
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

        if size == 1:
            gen_weight_const_size = 1
            gen_weight_affine_size = 1 if gen_affine else None
            gen_bias_const_size = 1 if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None
        elif size == 2:
            gen_weight_const_size = in_channels
            gen_weight_affine_size = in_channels if gen_affine else None
            gen_bias_const_size = 1 if bias else None
            gen_bias_affine_size = 1 if bias and gen_affine else None

        self.weight_const_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_weight_const_size, gen_depth, target='const')
        self.weight_affine_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_weight_affine_size, gen_depth, target='affine') if gen_affine else None
        self.bias_const_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_bias_const_size, gen_depth, target='const') if bias else None
        self.bias_affine_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_bias_affine_size, gen_depth, target='affine') if bias and gen_affine else None

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
    def __init__(self, num_features,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,  emb_dim=8, size=1, gen_depth=2, gen_affine=False, gen_hidden_layer=64, device=None, dtype=None):
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
        
        gen_weight_const_size = 1
        gen_weight_affine_size = 1 if gen_affine else None
        gen_bias_const_size = 1 
        gen_bias_affine_size = 1 if gen_affine else None

        self.weight_const_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_weight_const_size, gen_depth, target='const')
        self.weight_affine_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_weight_affine_size, gen_depth, target='affine') if gen_affine else None
        self.bias_const_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_bias_const_size, gen_depth, target='const') 
        self.bias_affine_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_bias_affine_size, gen_depth, target='affine') if gen_affine else None

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
    
class InstanceNorm2d_emb(nn.Module):
    def __init__(self, num_features, eps, momentum, emb_dim, size, gen_depth=2, gen_affine=False, gen_hidden_layer=64, bias=True, device=None):
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

        self.weight_const_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_weight_const_size, gen_depth, target='const')
        self.weight_affine_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_weight_affine_size, gen_depth, target='affine') if gen_affine else None
        self.bias_const_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_bias_const_size, gen_depth, target='const') if bias else None
        self.bias_affine_generator = WeightGenerator(emb_dim, gen_hidden_layer, gen_bias_affine_size, gen_depth, target='affine') if bias and gen_affine else None

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
    def __init__(self, mode, in_channels, out_channels, kernel_size, emb_dim=None, size=None, gen_depth=2, gen_affine=False, gen_hidden_layer=64, stride=1, padding=0, dilation=1, groups=1, bias=True, device=None):
        super().__init__()
        self.mode = mode
        if mode == MODE_NAMES['embedding']:
            self.conv = Conv2d_emb(in_channels, out_channels, kernel_size, emb_dim, size, gen_depth, gen_affine, gen_hidden_layer, stride, padding, dilation, groups, bias, device)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, device=device)

    def forward(self, x, emb):
        if self.mode == MODE_NAMES['embedding']:
            out = self.conv(x, emb)
        else:
            out = self.conv(x)
        return out
    
class GeneralConvTranspose2d(nn.Module):
    def __init__(self, mode, in_channels, out_channels, kernel_size, emb_dim=None, size=None, gen_depth=2, gen_affine=False, gen_hidden_layer=64, stride=1, padding=0, groups=1, bias=True, dilation=1, device=None):
        super().__init__()
        self.mode = mode
        if mode == MODE_NAMES['embedding']:
            self.transposed_conv = ConvTranspose2d_emb(in_channels, out_channels, kernel_size, emb_dim, size, gen_depth, gen_affine, gen_hidden_layer, stride, padding, groups, bias, dilation, device)
        else:
            self.transposed_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation, device=device)

    def forward(self, x, emb):
        if self.mode == MODE_NAMES['embedding']:
            out = self.transposed_conv(x, emb)
        else:
            out = self.transposed_conv(x)
        return out
    
class GeneralLinear(nn.Module):
    def __init__(self, mode, in_channels, out_channels, emb_dim=None, size=None, gen_depth=2, gen_affine=False, gen_hidden_layer=64, bias=True, device=None):
        super().__init__()
        self.mode = mode
        if mode == MODE_NAMES['embedding']:
            self.linear = Linear_emb(in_channels, out_channels, emb_dim, size, gen_depth, gen_affine, gen_hidden_layer, bias, device)
        else:
            self.linear = nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias, device=device)

    def forward(self, x, emb):
        if self.mode == MODE_NAMES['embedding']:
            out = self.linear(x, emb)
        else:
            out = self.linear(x)
        return out
    
class GeneralBatchNorm2d(nn.Module):
    def __init__(self, mode, num_features, eps=1e-05, momentum=0.1, emb_dim=None, size=None, gen_depth=2, gen_affine=False, gen_hidden_layer=64, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.mode = mode
        if mode == MODE_NAMES['embedding']:
            self.batch_norm = BatchNorm2d_emb(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, emb_dim=emb_dim, size=size, gen_depth=gen_depth, gen_affine=gen_affine, gen_hidden_layer=gen_hidden_layer, device=device)
        else:
            self.batch_norm = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device)
    
    def forward(self, x, emb):
        if self.mode == MODE_NAMES['embedding']:
            out = self.batch_norm(x, emb)
        else:
            out = self.batch_norm(x)
        return out
    
class BatchNorm2d_noemb(nn.Module):
    def __init__(self, mode, num_features, eps=1e-05, momentum=0.1, emb_dim=None, size=None, gen_depth=2, gen_affine=False, gen_hidden_layer=64, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device)
    
    def forward(self, x, emb):
        out = self.batch_norm(x)
        return out
    
class GeneralInstanceNorm2d(nn.Module):
    def __init__(self, mode, num_features, eps=1e-05, momentum=0.1, emb_dim=None, size=None, gen_depth=2, gen_affine=False, gen_hidden_layer=64, device=None, dtype=None):
        super().__init__()
        self.mode = mode
        if mode == MODE_NAMES['embedding']:
            self.instance_norm = InstanceNorm2d_emb(num_features, eps, momentum, emb_dim, size, gen_depth, gen_affine, gen_hidden_layer, device)
        else:
            self.instance_norm = nn.InstanceNorm2d(num_features=num_features, eps=eps, momentum=momentum, device=device)
    
    def forward(self, x, emb):
        if self.mode == MODE_NAMES['embedding']:
            out = self.instance_norm(x, emb)
        else:
            out = self.instance_norm(x)
        return out