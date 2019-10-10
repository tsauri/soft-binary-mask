import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import numpy as np
import math
import random
from collections import OrderedDict
from itertools import islice
import copy
import time
OldOrderedDict = OrderedDict
class SlicableOrderedDict(OldOrderedDict):
    def __getitem__(self, k):
        if not isinstance(k, slice):
            return OldOrderedDict.__getitem__(self, k)
        start = 0 if k.start is None else k.start
        stop = -1 if k.stop is None else k.stop
        if stop < 0: stop = len(self.items())+1 + stop
        # import pdb;pdb.set_trace()
        return SlicableOrderedDict(islice(self.items(), start, stop))
OrderedDict = SlicableOrderedDict

# Code from https://github.com/simochen/model-tools.
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import copy
# from models.weight_gate import *
from collections import OrderedDict

THConv2d = nn.Conv2d
THLinear = nn.Linear
def print_model_param_nums(model=None):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total

BATCH_SIZE = 1
def print_model_param_flops(model=None, input_res=224, multiply_adds=True, 
    return_layer_stats=False,
    skip_nonparam=True,
    batch_size=None,
    return_latency=False,
    use_cpu=True):
    if not batch_size: # see main, workaround vgg16 when needed
        batch_size = BATCH_SIZE
    prods = {}
    if return_layer_stats:
        layer_stats = OrderedDict()

    def count_params(m):
        return sum(p.numel() for name, p in m.named_parameters() if p.requires_grad)

    def save_layer_stats(model, module, flops):
        for name, m in model.named_modules():
            if m is module:
                layer_stats[name] = OrderedDict()
                layer_stats[name]['class'] = m._get_name()
                layer_stats[name]['paramc'] = count_params(m)
                layer_stats[name]['flops'] = flops

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width# * batch_size

        list_conv.append(flops)
        if return_layer_stats: save_layer_stats(model, self, flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        # flops = batch_size * (weight_ops + bias_ops)
        flops = (weight_ops + bias_ops)
        list_linear.append(flops)
        if return_layer_stats: save_layer_stats(model, self, flops)

    list_bn=[]
    def bn_hook(self, input, output):
        flops=input[0].nelement()
        list_bn.append(flops)
        if return_layer_stats and not skip_nonparam: save_layer_stats(model, self, flops)

    list_relu=[]
    def relu_hook(self, input, output):
        flops=input[0].nelement()
        list_relu.append(input[0].nelement())
        if return_layer_stats and not skip_nonparam: save_layer_stats(model, self, flops)

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width# * batch_size

        list_pooling.append(flops)
        if return_layer_stats and not skip_nonparam: save_layer_stats(model, self, flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        # flops = output_height * output_width * output_channels * batch_size * 12
        flops = output_height * output_width * output_channels * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, THConv2d) or isinstance(net, GConv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, THLinear) or isinstance(net, GLinear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d) or isinstance(net, GBatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.BatchNorm1d) or isinstance(net, GBatchNorm1d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    else:
        model = copy.deepcopy(model)
    input = Variable(torch.rand(batch_size, 3, input_res,input_res), requires_grad = True)
    if return_latency:
        model_lat = copy.deepcopy(model)
        input_lat = copy.deepcopy(input)
        if torch.cuda.is_available(): 
            model_lat = model_lat.cuda()
            input_lat = input_lat.cuda()
        if use_cpu:
            model_lat = model_lat.cpu()
            input_lat = input_lat.cpu()
        
        t0 = time.time()
        model_lat(input_lat)
        latency = time.time() - t0
        # print("Latency: {:.2f} ms".format(latency/1e-3))
        del model_lat
        del input_lat
    else:
        latency = None

    foo(model)
    # input = Variable(torch.rand(3,input_res,input_res).unsqueeze(0), requires_grad = True)
    if torch.cuda.is_available(): 
        model = model.cuda()
        input = input.cuda()
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))

    # print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    del input
    del model

    
    if return_layer_stats:
        return total_flops, latency, layer_stats
    return total_flops, latency
        # find nonzeros in the weight b
# torch.nonzero(b).numel()//b.dim()

# find percentage of nonzeros
# torch.nonzero(b).numel()//b.dim() / b.numel()
TEMP=1
DROPOUT=0
FILTER_PRUNING = False
HYBRID_PRUNING = False
def sigmoid_temp(x, temp, dropout=None):
    # return x
    if dropout: x = x + torch.randn_like(x)*dropout
    # return F.hardtanh((x/temp)*temp, min_val=0,max_val=1)
    # return F.hardtanh(x, min_val=0,max_val=1)
    return F.hardtanh((x+0.5/temp)*temp, min_val=0,max_val=1)
    # return 1/(1+torch.exp(-x * temp))



def maskw(inp, temp=1.0, dropout=None, expand_dims=None): # above 50, nan
    mask = sigmoid_temp(inp, temp, dropout)
    if expand_dims:
        more_dims = (1 for i in range(expand_dims)) # prune filter level broadcast
        mask = mask.view(*mask.shape, *more_dims)#
    return mask
class GBatchNorm2d(nn.BatchNorm2d):
    dropout = 0
    temp = TEMP
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, mask_before=None):
        super().__init__(num_features, eps, momentum, affine,
                 track_running_stats)
        self.mask_before = []
        self.mask_before.append(mask_before)
        # self._modules.pop('mask_befor')
        # self.mweight = torch.nn.Parameter(torch.zeros_like(self.weight))
        # self.mbias = torch.nn.Parameter(torch.zeros_like(self.bias))#.fill_(-0.5))
        # self.mbias.requires_grad = False
        if self.dropout == 0: self.dropout = None
        self.input_shape = None
        self.output_shape = None
        self.max_flops = None
        self.reset_parameters()

    def get_masked_weights(self):
        _,_,weight_mask, _ = self.mask_before[0].get_masked_weights()
        weight_mask = weight_mask.squeeze()
        # weight_mask =  maskw(mask_before, self.temp, self.dropout).squeeze()
        bias_mask = weight_mask
        # import pdb; pdb.set_trace()
        weight = self.weight * weight_mask
        bias = self.bias * bias_mask
        # bias = torch.zeros_like(self.bias)
        return weight, bias, weight_mask, bias_mask

    def forward(self, input):
        self._check_input_dim(input)
        if self.input_shape is None:
            self.input_shape = input[0:1,:].size()

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        weight, bias, mask, _ = self.get_masked_weights()
        if self.max_flops is None:
            self.max_flops = self.get_max_flops()
        return F.batch_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def get_max_flops(self):
        flops =torch.zeros(self.input_shape).nelement() * 2
        return flops
class GBatchNorm1d(nn.BatchNorm1d):
    dropout = 0
    temp = TEMP

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, mask_before=None):
        super().__init__(num_features, eps, momentum, affine,
                 track_running_stats)
        
        self.mask_before = mask_before
        # self.mweight = torch.nn.Parameter(torch.zeros_like(self.weight))
        # self.mbias = torch.nn.Parameter(torch.zeros_like(self.bias))#.fill_(-0.5))
        # self.mbias.requires_grad = False
        if self.dropout == 0: self.dropout = None
        self.input_shape = None
        self.output_shape = None
        self.max_flops = None
        self.reset_parameters()

    def get_masked_weights(self):
        _,_,weight_mask, _ = self.mask_before.get_masked_weights()
        weight_mask = weight_mask.squeeze()
        # weight_mask =  maskw(mask_before, self.temp, self.dropout).squeeze()
        bias_mask = weight_mask
        # import pdb; pdb.set_trace()
        weight = self.weight * weight_mask
        bias = self.bias * bias_mask
        # bias = torch.zeros_like(self.bias)
        return weight, bias, weight_mask, bias_mask

    def forward(self, input):
        self._check_input_dim(input)
        if self.input_shape is None:
            self.input_shape = input[0:1,:].size()

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        weight, bias, mask, _ = self.get_masked_weights()
        if self.max_flops is None:
            self.max_flops = self.get_max_flops()
        return F.batch_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def get_max_flops(self):
        flops =torch.zeros(self.input_shape).nelement() * 2
        return flops
THBatchNorm2d = nn.BatchNorm2d
# GBatchNorm1d = GBatchNorm2d
THBatchNorm1d = nn.BatchNorm1d


class GLinear(nn.Linear):
    temp = TEMP
    dropout = DROPOUT
    filter_pruning = FILTER_PRUNING
    hybrid_pruning = HYBRID_PRUNING
    mult_add = 2
    def __init__(self, in_features, out_features, bias=True, dropout=None): # 0.5
        super().__init__(in_features, out_features, bias)
        def get_out_channel_mask():
            mask = torch.zeros(self.weight.shape[:1])#.fill_(0.5)
            for i in range(len(self.weight.shape[1:])):
                mask = mask.unsqueeze(-1)
            return mask

        if self.filter_pruning:
            # assert self.hybrid_pruning is False
            mweight = get_out_channel_mask()
        else:
            mweight = torch.zeros(self.weight.shape)#.fill_(0.5)
        self.mweight = torch.nn.Parameter(mweight)

        if self.hybrid_pruning:
            mweight2 = get_out_channel_mask()
            self.mweight2 = torch.nn.Parameter(mweight2)   
                
        # self.mweight = torch.nn.Parameter(torch.zeros_like(self.weight))
        # self.mbias = torch.nn.Parameter(torch.zeros_like(self.bias)) if bias else None
        if self.dropout == 0: self.dropout = None
        self.input_shape = None
        self.output_shape = None
        self.max_flops = None

    def get_masked_weights(self):
        d = self.dropout# if self.training else None
        weight_mask =  maskw(self.mweight, self.temp, d)
        if self.hybrid_pruning:
            channel_mask = maskw(self.mweight2, self.temp, d)
            weight_mask = weight_mask * channel_mask
        weight = self.weight * weight_mask
        if self.bias is not None:       
            # bias_mask = maskw(self.mbias, self.temp, d)
            bias_mask = weight_mask.squeeze()
            bias = self.bias * bias_mask

        else:
            bias = None
            bias_mask = None
        return weight, bias, weight_mask, bias_mask

    def forward(self, input):
        if self.input_shape is None:
            self.input_shape = input.shape
        weight, bias, weight_mask, bias_mask = self.get_masked_weights()
        
        out = F.linear(input, weight, bias)
        if self.output_shape is None:
            self.output_shape = out.shape
        if self.max_flops is None:
            self.max_flops = self.get_max_flops()
        return out

    def get_max_flops(self):
        weight_ops = self.weight.nelement() * self.mult_add
        bias_ops = self.bias.nelement()

        flops = (weight_ops + bias_ops)
        return flops
        # layer = self
        # input_shape = layer.input_shape
        # total_mul = layer.in_features
        # total_add = layer.in_features - 1
        # layer.output_shape
        # y = torch.zeros(layer.output_shape)
            # num_elements = y.numel()
        # total_ops = (total_mul + total_add) * num_elements
        # # Pruning ConvNet ICLR 2017 method
        # total_ops = self.weight.numel()
        # return total_ops

THLinear = nn.Linear
# nn.Linear = GLinear

class GConv2d(nn.Conv2d):
    temp = TEMP
    dropout = DROPOUT
    filter_pruning = FILTER_PRUNING
    hybrid_pruning = HYBRID_PRUNING
    mult_add = 2
    dup_factor = 1
    avg_factor = 1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', dropout=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)

        def get_out_channel_mask():
            mask = torch.zeros(self.weight.shape[:1])#.fill_(0.5)
            for i in range(len(self.weight.shape[1:])):
                mask = mask.unsqueeze(-1)
            return mask

        if self.filter_pruning:
            # assert self.hybrid_pruning is False
            mweight = get_out_channel_mask()
        else:
            mweight = torch.zeros(self.weight.shape)#.fill_(0.5)
        self.mweight = torch.nn.Parameter(mweight)

        if self.hybrid_pruning:
            mweight2 = get_out_channel_mask()
            self.mweight2 = torch.nn.Parameter(mweight2)

        # self.mbias = torch.nn.Parameter(torch.zeros(self.bias.shape)) if bias else None

        if self.dropout == 0: self.dropout = None
        self.input_shape = None
        self.output_shape = None
        self.max_flops = None
        # self.current_mask = None
        # self.current_bias_mask = None
    def get_masked_weights(self):
        d = self.dropout# if self.training else None
        if self.dup_factor > 1:
            r = [1]*self.mweight.dim()
            r[0] = int(r[0]*self.dup_factor)

            mw = self.mweight.repeat(*r)
            weight_mask = maskw(mw, self.temp, d)
            print('weight_mask is', weight_mask.shape)
        else:
            weight_mask = maskw(self.mweight, self.temp, d)
        if self.avg_factor > 1:
            f = self.avg_factor
            s = weight_mask.size(0)
            weight_mask = weight_mask.view(s//f, f, 1, 1)
            # import pdb;pdb.set_trace()
            weight_mask = weight_mask[:,0:1,:,:]#.mean(dim=1)

            # weight_mask = weight_mask.max(dim=1, keepdim=True)[0]#.unsqueeze(1)
            # import pdb;pdb.set_trace()
        if self.hybrid_pruning:
            channel_mask = maskw(self.mweight2, self.temp, d)
            weight_mask = weight_mask * channel_mask
        weight = self.weight * weight_mask
        # try:
        #     weight = self.weight * weight_mask
        # except:
        #     import pdb;pdb.set_trace()
        # self.current_mask = mask

        if self.bias is not None:       
            # bias_mask = maskw(self.mbias, self.temp, d)
            bias_mask = weight_mask.squeeze()
            bias = self.bias * bias_mask
            # self.current_bias_mask = bias_mask

        else:
            bias = None
            bias_mask = None
        return weight, bias, weight_mask, bias_mask

    def forward(self, input):
        # if not self.training:
        #     selected_index = np.squeeze(np.argwhere(self.current_mask.data.cpu().numpy()))
        #     if selected_index.size == 1:
        #         selected_index = np.resize(selected_index, (1,)) 
        #     weight = weight[selected_index,:]

        #     if self.current_bias_mask:

        #         selected_index = np.squeeze(np.argwhere(self.weight_mask.data.cpu().numpy()))
        #         if selected_index.size == 1:
        #             selected_index = np.resize(selected_index, (1,)) 
        #         weight = weight[selected_index,:]

        if self.input_shape is None:
            self.input_shape = input.shape
        
        weight, bias, weight_mask, bias_mask = self.get_masked_weights()

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            out = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            out = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if self.output_shape is None:
            self.output_shape = out.shape
        if self.max_flops is None:
            self.max_flops = self.get_max_flops()


        return out

    def get_max_flops(self):
        # layer = self
        # # input_shape = layer.input_shape
        # cin = layer.in_channels
        # cout = layer.out_channels
        # kh, kw = layer.kernel_size
        # out_h = layer.output_shape[2]
        # out_w = layer.output_shape[3]
        # kernel_ops = 1 * kh * kw
        # bias_ops = 1 if layer.bias is not None else 0
        # ops_per_element = kernel_ops + bias_ops
        # output_elements = 1 * out_w * out_h * cout
        # total_ops = output_elements * ops_per_element * cin // layer.groups
        # return total_ops
        # import pdb;pdb.set_trace()

        batch_size, input_channels, input_height, input_width = self.input_shape
        batch_size, output_channels, output_height, output_width =self.output_shape

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * self.mult_add + bias_ops) * output_channels * output_height * output_width
        return flops 

    def lock_weights(self): # this is one way prune operation
        mask = maskw(self.mweight, self.temp, expand_dims=None)
        self.mweight.data = mask
        self.weight.data = (self.weight.data * mask)
        self.mweight.data[self.mweight.data != 0] = 1
        self.mweight.requires_grad = False

        if self.mbias is not None:
            self.mbias.data =  maskw(self.mbias, self.temp)
            self.bias.data = (self.bias.data * self.mbias.data)
            self.mbias.data[self.mbias.data != 0] = 1
            self.mbias.requires_grad = False

THConv2d = nn.Conv2d
# nn.Conv2d = GConv2d

def initialize_weights2(module,s=1e-2):
    if isinstance(module, GConv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        module.mweight.data.fill_(s)
        if module.bias is not None:
            module.bias.data.zero_()
            module.mbias.data.fill_(s)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()

    elif isinstance(module, GLinear):
        module.mweight.data.fill_(s)
        if module.bias is not None:
            module.bias.data.zero_()
            module.mbias.data.fill_(s)

def initialize_weights(module):
    if isinstance(module, GConv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        module.mweight.data.random_(0, 1).div_(module.temp*2)
        # module.mweight.data.uniform_(0, 1).div_(module.temp)
        if fattr(module,'mweight2',None) is not None:
            module.mweight.data.random_(0, 1).div_(module.temp*2)
            # module.mweight.data.uniform_(0, 1).div_(module.temp)
            # module.mweight2.data.uniform_(.5/module.temp)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, GLinear):
        if module.bias is not None:
            module.bias.data.zero_()
            module.mbias.data.zero_()

def initialize_mweights(module,s=1e-1,until=None):
    if isinstance(module, GConv2d):
        module.mweight.data.fill_(s)
        if getattr(module,'mweight2',None) is not None:
            module.mweight2.data.fill_(s)
        # if module.bias is not None:
        #     module.mbias.data.fill_(s)

    elif isinstance(module, GLinear):
        module.mweight.data.fill_(s)
        if getattr(module,'mweight2',None) is not None:
            module.mweight2.data.fill_(s)
        # if module.bias is not None:
        #     module.mbias.data.fill_(s)

def set_temp(module, temp):
    for layer in module.modules():
        if getattr(layer,'temp',None) is not None:
            layer.temp = temp

def get_relative_layer_stats(pruned_model, train_model, base_layer_stats, epoch=None, 
    iteration=None, batch_size=None, multiply_adds=False, training=None, return_latency=False):
    model_stats = OrderedDict()
    model_stats['base_layer_stats'] = base_layer_stats
    model_stats['epoch'] = epoch
    model_stats['iteration'] = iteration
    model_stats['batch_size'] = batch_size
    model_stats['training'] = training
    relative_stats = OrderedDict()
    flops, latency, layer_stats = print_model_param_flops(pruned_model, 32, 
        multiply_adds=multiply_adds, return_layer_stats=True, return_latency=return_latency)
    assert len(base_layer_stats) == len(layer_stats)
    model_stats['flops'] = flops
    model_stats['latency'] = latency
    model_stats['layer_stats'] = layer_stats

    mask_ratio_dict = get_prune_ratio(train_model)
    # masks = mask_ratio_dict['masks'] # too big
    # chmasks = [mask.cpu().numpy() for mask in mask_ratio_dict['chmasks']]
    # model_stats['chmasks'] = chmasks
    model_stats['mask_ratio_dict'] = mask_ratio_dict

    for name, v in layer_stats.items(): # param sparsity ps; flops sparsity fs
        relative_stats[name] = OrderedDict()
        # import pdb;pdb.set_trace()
        relative_stats[name]['psratio'] = 1 - v['paramc']/base_layer_stats[name]['paramc']
        relative_stats[name]['fsratio'] = 1 - v['flops']/base_layer_stats[name]['flops']
    
    model_stats['relative_stats'] =  relative_stats

        # flops[name] = 
    return model_stats



def get_prune_ratio(module):
    # global flag; flag += 1
    nonzeros = OrderedDict()
    totals = OrderedDict()
    ratios = OrderedDict()

    chnonzeros = OrderedDict()
    chtotals = OrderedDict()
    chratios = OrderedDict()

    activeflops = OrderedDict()
    totalflops = OrderedDict()
    flopsratio = OrderedDict()

    nonzeros2 = OrderedDict() # hybrid
    totals2 = OrderedDict()
    ratios2 = OrderedDict()

    # masks = OrderedDict()
    chmasks = OrderedDict()

    for name, layer in module.named_modules():
        if getattr(layer,'mweight',None) is not None:
            mask = maskw(layer.mweight, layer.temp)
            mask2 = None
            if getattr(layer,'filter_pruning', None) is True:
                if getattr(layer,'hybrid_pruning', None) is True:  
                    mask2 = maskw(layer.mweight2, layer.temp)
                    mask = mask * mask2
                chmask = mask.clone()
                mask = mask.repeat(1,*layer.weight.shape[1:])
            else:
                chmask = mask
            
            # masks[name] = mask
            chmasks[name] = chmask
            nonzero = torch.nonzero(mask).numel()//mask.dim()
            total = mask.numel()
            nonzeros[name] = nonzero
            totals[name] = total
            ratios[name] = 1.0 - nonzero/total

            chnonzero = torch.nonzero(chmask).numel()//chmask.dim()
            chtotal = chmask.numel()
            chnonzeros[name] = chnonzero
            chtotals[name] = chtotal
            chratios[name] = 1.0 - chnonzero/chtotal

            if mask2 is not None:
                nonzero2 = torch.nonzero(mask2).numel()//mask2.dim()
                total2 = mask2.numel()
                nonzeros2[name] = nonzero2
                totals2[name] = total2
                ratios2[name] = 1.0 - nonzero2/total2
            else:
                nonzeros2[name] = nonzero # dummy data
                totals2[name] = total
                ratios2[name] = 1.0 - nonzero/total

            if getattr(layer,'mbias',None) is not None:
                mask = maskw(layer.mbias, layer.temp)
                nonzero = torch.nonzero(mask).numel()//mask.dim()
                total = mask.numel()
                nonzeros[name+'_bias'] = nonzero
                totals[name+'_bias'] = total
                ratios[name+'_bias'] = 1.0 - nonzero/total
                chnonzeros[name+'_bias'] = float(nonzero)
                chtotals[name+'_bias'] = float(total)
                chratios[name+'_bias'] = 1.0 - nonzero/total
                if mask2 is not None:
                    nonzeros2[name+'_bias'] = nonzero
                    totals2[name+'_bias'] = total
                    ratios2[name+'_bias'] = 1.0 - nonzero/total


            total_ops = layer.max_flops # must run fwd pass once
            activeflops[name] = total_ops * (1.0-chratios[name])
            totalflops[name] = total_ops
            flopsratio[name] = 1.0-activeflops[name]/totalflops[name]
            if getattr(layer,'mbias',None) is not None:
                activeflops[name+'_bias'] = 0 # pad the dict
                totalflops[name+'_bias'] = 0
                flopsratio[name+'_bias'] = 0


            
                # nonzeros.append(nonzero)
                # totals.append(total)
    cr_dict = {
    # 'masks':masks,
    'chmasks':chmasks,

    'nonzeros':nonzeros,
    'nonzeros':nonzeros,
    'totals':totals,
    'ratios':ratios,
    
    'chnonzeros':chnonzeros,
    'chtotals': chtotals,
    'chratios': chratios,

    'nonzeros2':nonzeros2,
    'totals2':totals2,
    'ratios2':ratios2,

    'activeflops': activeflops,
    'totalflops': totalflops,
    'flopsratio': flopsratio}
    # if until:
    #     for k in cr_dict.keys():
    #         cr_dict[k] = cr_dict[k][:until]
        
    return cr_dict

def get_masks(module, as_list=False, expand=False):
    masks=[]
    for idx, layer in enumerate(module.modules()):
        if getattr(layer,'mweight',None) is not None:
            mask = sigmoid_temp(layer.mweight, layer.temp)
            if expand:
                # mask = mask.repeat(1,*layer.weight.shape[1:])
                # import pdb;pdb.set_trace()
                mask = mask.expand(mask.size(0),*layer.weight.shape[1:]).contiguous()
                
            # if getattr(layer,'mweight2',None) is not None:
            #     mask2 = sigmoid_temp(layer.mweight2, layer.temp)
            #     mask = mask*mask2
            masks.append(mask.view(-1))

            # if getattr(layer,'mbias',None) is not None:
            #     mask = sigmoid_temp(layer.mbias, layer.temp)
            #     masks.append(mask.view(-1))
    if as_list: 
        return masks
    m = torch.cat(masks)
    return m

def get_masks2(module, as_list=False, expand=False):
    masks=[]
    for idx, layer in enumerate(module.modules()):
        if getattr(layer,'mweight2',None) is not None:
            mask = sigmoid_temp(layer.mweight2, layer.temp)
            if expand:
                # mask = mask.repeat(1,*layer.weight.shape[1:])
                # import pdb;pdb.set_trace()
                mask = mask.expand(mask.size(0),*layer.weight.shape[1:]).contiguous()
                
            # if getattr(layer,'mweight2',None) is not None:
            #     mask2 = sigmoid_temp(layer.mweight2, layer.temp)
            #     mask = mask*mask2
            masks.append(mask.view(-1))

            # if getattr(layer,'mbias',None) is not None:
            #     mask = sigmoid_temp(layer.mbias, layer.temp)
            #     masks.append(mask.view(-1))
    if as_list: 
        return masks
    m = torch.cat(masks)
    return m
# def get_masks2(module, as_list=False):
#     masks=[]
#     for idx, layer in enumerate(module.modules()):
#         if getattr(layer,'mweight2',None) is not None:
#             mask = sigmoid_temp(layer.mweight2, layer.temp)
#             masks.append(mask.view(-1))
#     if as_list: 
#         return masks
#     m = torch.cat(masks)
#     return m


def set_trainable(module,mode="all"):
    flag_mw = mode == "all" or mode=="gate"
    flag_w = mode == "all" or mode=="weight"
    flag_mb = mode == "all" or mode=="gate"
    flag_b = mode == "all" or mode=="weight"
    for layer in module.modules():
        if getattr(layer,'mweight',None) is not None:
            layer.mweight.requires_grad = flag_mw
            layer.weight.requires_grad = flag_w
            if getattr(layer,'mweight2',None) is not None:
                layer.mweight2.requires_grad = flag_mw

            if getattr(layer,'mbias',None) is not None:
                layer.mbias.requires_grad = flag_mb
                layer.bias.requires_grad = flag_b

def update_pruned_model(pruned_model, train_model):
    def prune_conv(pm_conv, tm_conv, prev_tm_conv=None):
        # print('conv shape', tm_conv.weight.shape)
        weight, bias, weight_mask, bias_mask = tm_conv.get_masked_weights()
        # if tm_conv.dup_factor > 1:
            # print('dup factor is',tm_conv.dup_factor)
        prev_weight_mask, prev_bias_mask = None, None
        if prev_tm_conv is not None:
            prev_weight, _, prev_weight_mask, _ = prev_tm_conv.get_masked_weights()
        not_empty = torch.nonzero(weight_mask.squeeze()).numel() > 0
        # not_empty = False
        if tm_conv.filter_pruning:
            # print('final out', torch.nonzero(weight_mask.squeeze()).numel())
            if not_empty:
                pruned_weight = weight[(weight_mask>0).squeeze(),::].clone()
            else:
                pruned_weight = weight.clone()
                # import pdb;pdb.set_trace()
            if prev_weight_mask is not None:
                prev_not_empty = torch.nonzero(prev_weight_mask.squeeze()).numel() > 0
                # prev_not_empty = False
                if prev_not_empty and prev_weight_mask.size(0) == weight.size(1):
                    # try:
                    pruned_weight =pruned_weight[:,(prev_weight_mask>0).squeeze()]
                    # except: 
                    #     import pdb; pdb.set_trace()

                # print('prev out', torch.nonzero((prev_weight_mask.squeeze())).numel())
                    # pruned_weight =pruned_weight[:,(prev_weight_mask>0).squeeze(),::].data
                    # if not pruned_weight.size(1) == torch.nonzero(prev_weight_mask.squeeze()).numel():
                    #     import pdb; pdb.set_trace()
            # if type(pm_conv) is nn.Dropout: import pdb;pdb.set_trace()
            pm_conv.weight.data = pruned_weight
            # print('conv after', pm_conv.weight.shape)
        else:
            pm_conv.weight.data = weight.data.clone()

        if bias is not None:
            if tm_conv.filter_pruning and not_empty:
                pm_conv.bias.data = bias[(weight_mask>0)].data.clone()
            else:
                pm_conv.bias.data = bias.data.clone()

    def prune_linear(pm_conv, tm_conv, prev_tm_conv=None):
        # print('linear shape', tm_conv.weight.shape)
        weight, bias, weight_mask, bias_mask = tm_conv.get_masked_weights()
        prev_weight_mask, prev_bias_mask = None, None
        if prev_tm_conv is not None:
            _, _, prev_weight_mask, prev_bias_mask = prev_tm_conv.get_masked_weights()
        elif getattr(tm_conv, "prev_mweight", None) is not None:
            prev_weight_mask = tm_conv.prev_mweight
            prev_bias_mask = prev_weight_mask.squeeze()

        not_empty = torch.nonzero(weight_mask.squeeze()).numel() > 0
        if tm_conv.filter_pruning:
            # print('final out', torch.nonzero(weight_mask.squeeze()).numel())
            if not_empty:
                pruned_weight = weight[(weight_mask>0).squeeze(),:]
            else:
                pruned_weight = weight[::]
            if prev_weight_mask is not None:
                prev_not_empty = torch.nonzero(prev_weight_mask.squeeze()).numel() > 0
                if prev_not_empty:
                # print('prev out', torch.nonzero((prev_weight_mask.squeeze())).numel())
                    # try:
                    pruned_weight =pruned_weight[:,(prev_weight_mask>0).squeeze()].data
                    # except: 
                    #     import pdb; pdb.set_trace()

            pm_conv.weight.data = pruned_weight
            # print('linear after', pm_conv.weight.shape)
        else:
            pm_conv.weight.data = weight.data.clone()

        if bias is not None:
            if tm_conv.filter_pruning and not_empty:
                pm_conv.bias.data = bias[(weight_mask>0).squeeze()].data.clone()
            else:
                pm_conv.bias.data = bias.data.clone()

            # print(pm_conv.weight.shape)
    def prune_thconv(pm_conv, tm_conv, prev_tm_conv=None):
        pm_conv.weight.data = tm_conv.weight.data
        if pm_conv.bias is not None:
            # import pdb;pdb.set_trace()
            pm_conv.bias.data = tm_conv.bias.data.clone()
        if prev_tm_conv is not None:
            _, _, prev_weight_mask, prev_bias_mask = prev_tm_conv.get_masked_weights()
            prev_not_empty = torch.nonzero(prev_weight_mask.squeeze()).numel() > 0
            # prev_not_empty = False

            if prev_not_empty:
                # if pm_conv.groups != 1:
                #     pm_conv.groups = (prev_weight_mask>0).squeeze().numel()
                #     print("Found,",repr((prev_weight_mask>0).squeeze().numel()))
                pm_conv.weight.data = pm_conv.weight.data[:,(prev_weight_mask>0).squeeze(),::].data
        # print('thconv2d after prune shape', pm_conv.weight.shape)
    def prune_thlinear(pm_conv, tm_conv, prev_tm_conv=None):
        pm_conv.weight.data = tm_conv.weight.data
        if pm_conv.bias is not None:
            pm_conv.bias.data = tm_conv.bias.data.clone()
        if prev_tm_conv is not None:
            _, _, prev_weight_mask, prev_bias_mask = prev_tm_conv.get_masked_weights()
            prev_not_empty = torch.nonzero(prev_weight_mask.squeeze()).numel() > 0
            # prev_not_empty = False
            if prev_not_empty:
                pm_conv.weight.data = pm_conv.weight.data[:,(prev_weight_mask>0).squeeze()].data
    def prune_bn(pm_conv, tm_conv, prev_tm_conv=None):
        # prev_or_gbn = tm_conv if type(tm_conv) in [GBatchNorm2d, GBatchNorm1d] else prev_tm_conv
        # prev_or_gbn = prev_tm_conv
        if type(tm_conv) in [GBatchNorm2d, GBatchNorm1d]:
            # assert type(prev_tm_conv) in [GConv2d, GLinear]
            weight, bias, weight_mask, bias_mask = tm_conv.get_masked_weights()
            not_empty = torch.nonzero(weight_mask).numel() > 0
            # not_empty = False
            if not_empty:
                idx = (weight_mask>0)
                pm_conv.weight.data = weight.data[idx].clone()
                pm_conv.bias.data = bias.data[idx].clone()
                pm_conv.running_mean = tm_conv.running_mean[idx].clone()
                pm_conv.running_var = tm_conv.running_var[idx].clone()
            else:
                pm_conv.weight.data = weight.data.clone()
                pm_conv.bias.data = bias.data.clone()
                pm_conv.running_mean = tm_conv.running_mean.clone()
                pm_conv.running_var = tm_conv.running_var.clone()
                # idx = torch.ones_like(weight_mask).long()
                # pm_conv.weight.data = weight.data[idx].clone()
                # pm_conv.bias.data = bias.data[idx].clone()
                # pm_conv.running_mean = tm_conv.running_mean[idx].clone()
                # pm_conv.running_var = tm_conv.running_var[idx].clone()
        # elif type(prev_tm_conv) in [GConv2d, GLinear]:
        #     weight = tm_conv.weight
        #     bias = tm_conv.bias
        #     _,_, weight_mask,_  = prev_tm_conv.get_masked_weights()
        #     weight_mask.squeeze()
        #     not_empty = torch.nonzero(weight_mask).numel() > 0
        #     not_empty = False
        #     if not_empty:
        #         idx = (weight_mask>0)
        #     else:
        #         idx = torch.ones_like(weight_mask).long()
        #     pm_conv.weight.data = weight.data[idx].clone()
        #     pm_conv.bias.data = bias.data[idx].clone()
        #     pm_conv.running_mean = tm_conv.running_mean[idx].clone()
        #     pm_conv.running_var = tm_conv.running_var[idx].clone()
        else:
            # if type(prev_tm_conv) in [GConv2d, GLinear]: import pdb;pdb.set_trace()
            assert type(prev_tm_conv) not in [GConv2d, GLinear]
            pm_conv.load_state_dict(tm_conv.state_dict())
            


    # tm_modules = [i for i in list(train_model.modules()) if type(i) is not nn.Dropout]
    # pm_modules = [i for i in list(pruned_model.modules()) if type(i) is not nn.Dropout]

    tm_modules = list(train_model.modules())
    pm_modules = list(pruned_model.modules())
    # import pdb;pdb.set_trace()

    # update weights
    prev_tm = None
    downsample = None
    conv1 = None
    ds_i = None
    for i in range(len(tm_modules)):
        # if type(tm_modules[i]) is nn.BatchNorm2d:
        #     prune_bn(pm_modules[i], bn_before_after[tm_modules[i]])
        if type(getattr(tm_modules[i], 'downsample', None)) is GConv2d:
            downsample = tm_modules[i].downsample # wideresnet
            conv1 = tm_modules[i].conv1 # wideresnet
            ds_i = i+6
            assert tm_modules[ds_i] is downsample

        if type(tm_modules[i]) is GConv2d:
            if tm_modules[i] is downsample:
                continue
            prune_conv(pm_modules[i], tm_modules[i], prev_tm)
            if tm_modules[i] is conv1:
                prune_conv(pm_modules[ds_i], tm_modules[ds_i], prev_tm)
            prev_tm = tm_modules[i]

        elif type(tm_modules[i]) is GLinear:# and i < len(tm_modules)-1:
            prune_linear(pm_modules[i], tm_modules[i], prev_tm)
            # if tm_modules[i] is not check_downsample:
            prev_tm = tm_modules[i]
            # prev_tm = tm_modules[i]

        elif type(tm_modules[i]) is THConv2d:# and prev_tm is not None:
            prune_thconv(pm_modules[i], tm_modules[i], prev_tm)
            prev_tm = None
        elif type(tm_modules[i]) is THLinear:# and prev_tm is not None:
            prune_thlinear(pm_modules[i], tm_modules[i], prev_tm)
            prev_tm = None
        elif type(tm_modules[i]) in [GBatchNorm2d, GBatchNorm1d, 
                                        THBatchNorm2d, THBatchNorm1d]: # we need to cut down BN channels
            prune_bn(pm_modules[i], tm_modules[i], prev_tm)

    # from .resnet import BasicBlock # fix downsample avg pool
    # for i in range(len(tm_modules)):
    #     if type(tm_modules[i]) is BasicBlock:
    #         if tm_modules[i].downsample:

    #             print('prune downsample',tm_modules[i].downsample)
    #             print('current planes',tm_modules[i].downsample.planes)
    #             print('m.conv2.weight.size(0)',tm_modules[i].conv2.weight.size(0))
    #             print()

    #             tm_modules[i].downsample.planes = tm_modules[i].conv2.weight.size(0)
            # if m.downsample: continue
            # prev_tm = None
                    
                    # continue
# def fix_bn(model):
#     # return
#     glayer = None
#     for i, m in enumerate(model.modules()): 
#         if type(m) in [GConv2d, GLinear]:
#             glayer = m
#         elif type(m) in [GBatchNorm2d, GBatchNorm1d]: # we need to cut down BN channels
#             # import pdb;pdb.set_trace()
#             if glayer is not None:
#                 mask_before = glayer.mweight
#                 m.mask_before = mask_before
#                 # import pdb;pdb.set_trace()
#                 # d = m.state_dict()
#                 # bn = m
#                 # gbn = GBatchNorm2d if type(bn) is THBatchNorm2d else GBatchNorm1d
#                 # _, _, mask_before, _ = glayer.get_masked_weights()
#                 # m = gbn(bn.num_features, bn.eps, bn.momentum, bn.affine,
#                 #             bn.track_running_stats, mask_before) 
#                 # import pdb;pdb.set_trace()
# def fix_resnet(model):
#     from .resnet import BasicBlock
#     share_m = None
#     net = None
#     share_m = None
#     net = model
#     for i, m in enumerate(reversed(list(model.modules()))): 
#         if type(m) is BasicBlock:
#             if share_m is None:
#                 share_m = m.block[3]
#                 continue
#             # if m.downsample: continue
#             print("set ", m.block[3], "with mask from", share_m)
#             if m.block[3].mweight.size() != share_m.mweight.size():
#                 f = share_m.mweight.size(0)//m.block[3].mweight.size(0)
#                 m.block[3].avg_factor = f
#                 print("avg factor is", f)
#             else:
#                 print("no avg")

#             # elif m.conv2.mweight.size() == share_m.mweight.size():
#             print()
#             m.block[3].mweight = share_m.mweight
#         elif m is model:
#             print("found model")
#             # import pdb;pdb.set_trace()

#             print("set ", m.layer0[0], "with mask from", share_m)
#             if m.layer0[0].mweight.size() != share_m.mweight.size():
#                 f = share_m.mweight.size(0)//m.layer0[0].mweight.size(0)
#                 m.layer0[0].avg_factor = f
#                 print("avg factor is", f)
#             else:
#                 print("no avg")

#             # elif m.conv2.mweight.size() == share_m.mweight.size():
#             print()
#             m.layer0[0].mweight = share_m.mweight

def fix_resnet(model):
    from .resnet import BasicBlock
    for i, m in enumerate(model.modules()): 
        if type(m) is BasicBlock:
            c = m.block[3]
            d = c.state_dict()
            m.block[3] = THConv2d(c.in_channels,c.out_channels,c.kernel_size, c.stride, c.padding)#m.conv2.input_channel,m.conv2.output_channels)
            m.block[3].load_state_dict(d, strict=False)
            c = m.block[4]
            d = c.state_dict()
            m.block[4] = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
                            c.track_running_stats)
            m.block[4].load_state_dict(d, strict=False)
        elif i == 0:
            c = m.layer0[0]
            d = c.state_dict()
            m.layer0[0] = THConv2d(c.in_channels,c.out_channels,
                c.kernel_size, c.stride, c.padding, 
                c.dilation,
                 c.groups, bias=False, padding_mode=c.padding_mode)
            m.layer0[0].load_state_dict(d, strict=False)
            # import pdb;pdb.set_trace()

            c = m.layer0[1]
            d = c.state_dict()
            m.layer0[1] = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
                            c.track_running_stats)
            m.layer0[1].load_state_dict(d, strict=False)

# def ofix_resnet(model):
#     from .resnet import BasicBlock
#     share_m = None
#     conv1_mweight = None
#     net = None
#     prev_layer = None
#     curr_layer = None
#     dup_factor = 1
#     for i, m in enumerate(model.modules()): 
            
#         if i == 0:
#             net = m
#             share_m = m.conv1

#         elif type(m) is BasicBlock:
#             # if m.downsample: continue
#             print("set ", m.conv2, "with mask from", share_m)
#             if m.conv2.mweight.size() != share_m.mweight.size():
#                 f = m.conv2.mweight.size(0)/share_m.mweight.size(0)
#                 m.conv2.dup_factor = f
#                 print("dup factor is", f)

#             # elif m.conv2.mweight.size() == share_m.mweight.size():
#             print()
#             m.conv2.mweight = share_m.mweight

    # import pdb; pdb.set_trace()
def fix_wideresnet(model):
    from .wideresnet import wide_basic
    share_m = None
    last_bn1 = None
    for i, m in enumerate(reversed(list(model.modules()))): 
        if type(m) is wide_basic:
            if share_m is None:
                share_m = m.conv2
                m.bn1.mask_before[0] = share_m
                continue

            # if last_bn == None:
            # if m.downsample: continue
            print("set ", m.conv2, "with mask from", share_m)
            f = share_m.mweight.size(0)//m.conv2.mweight.size(0)
            if m.conv2.mweight.size() != share_m.mweight.size():
                m.conv2.avg_factor = f
                print("avg factor is", f)
            else:
                print("no avg")

            # elif m.conv2.mweight.size() == share_m.mweight.size():
            print()
            m.conv2.mweight = share_m.mweight
            if m.downsample:
                m.downsample.mweight = m.conv2.mweight
                m.downsample.avg_factor = f
                m.downsample.prev_mweight = m.conv1.mweight # to refer mask to prune input dim
            if last_bn1:
                last_bn1.mask_before[0] = m.conv2
                last_bn1 = m.bn1
            else:
                last_bn1 = m.bn1
        elif m is model:
            print("found model")
            # import pdb;pdb.set_trace()

            print("set ", m.conv1, "with mask from", share_m)
            if m.conv1.mweight.size() != share_m.mweight.size():
                f = share_m.mweight.size(0)//m.conv1.mweight.size(0)
                m.conv1.avg_factor = f
                print("avg factor is", f)
            else:
                print("no avg")

            # elif m.conv2.mweight.size() == share_m.mweight.size():
            print()
            m.conv1.mweight = share_m.mweight
            m.bn1.mask_before[0] = share_m
            # import pdb;pdb.set_trace()
            

def ofix_resnet(model):
    from .resnet import BasicBlock
    for i, m in enumerate(model.modules()): 
        if type(m) is BasicBlock:
            c = m.conv2
            d = c.state_dict()
            m.conv2 = THConv2d(c.in_channels,c.out_channels,c.kernel_size, c.stride, c.padding)#m.conv2.input_channel,m.conv2.output_channels)
            m.conv2.load_state_dict(d, strict=False)
            c = m.bn2
            d = c.state_dict()
            m.bn2 = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
                            c.track_running_stats)
            m.bn2.load_state_dict(d, strict=False)
        elif i == 0:
            c = m.conv1
            d = c.state_dict()
            m.conv1 = THConv2d(c.in_channels,c.out_channels,
                c.kernel_size, c.stride, c.padding, 
                c.dilation,
                 c.groups, bias=False, padding_mode=c.padding_mode)
            m.conv1.load_state_dict(d, strict=False)
            # import pdb;pdb.set_trace()

            c = m.bn1
            d = c.state_dict()
            m.bn1 = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
                            c.track_running_stats)
            m.bn1.load_state_dict(d, strict=False)

            # c = m.fc
            # d = c.state_dict()
            # m.fc = THLinear(c.in_features,c.out_features)#m.conv2.input_channel,m.conv2.output_channels)
            # m.fc.load_state_dict(d, strict=False)

def fix_vgg16(model):
    for i, m in enumerate(model.modules()): 
        if i == 0:
            a = m.classifier[0]
            b = m.classifier[3]
            # b = m.classifier[2]
            # del m.classifier
            m.classifier = nn.Sequential(
              THLinear(a.in_features, a.out_features),
              nn.BatchNorm1d(a.out_features),
              nn.ReLU(inplace=False),
              THLinear(512, b.out_features)
            )
            m._initialize_weights()

def fix_vgg(model):
    for i, m in enumerate(model.modules()): 
        if i == 0:
            c = m.classifier
            m.classifier = THLinear(c.in_features, c.out_features)
            m._initialize_weights()

def fix_preresnet(model):
    from .preresnet import Bottleneck
    for i, m in enumerate(model.modules()): 
        if type(m) is Bottleneck:
            c = m.conv3
            d = c.state_dict()
            m.conv3 = THConv2d(c.in_channels,c.out_channels,c.kernel_size, c.stride, c.padding)#m.conv2.input_channel,m.conv2.output_channels)
            m.conv3.load_state_dict(d, strict=False)
            if m.downsample is None:
                continue
            c = m.downsample[0] # refernce, not mutate
            d = c.state_dict()
            m.downsample[0] = THConv2d(c.in_channels,c.out_channels,c.kernel_size, c.stride, c.padding)#m.conv2.input_channel,m.conv2.output_channels)
            m.downsample[0].load_state_dict(d, strict=False)
        elif i == 0:
            c = m.conv1
            d = c.state_dict()
            m.conv1 = THConv2d(c.in_channels,c.out_channels,c.kernel_size, c.stride, c.padding)#m.conv2.input_channel,m.conv2.output_channels)
            m.conv1.load_state_dict(d, strict=False)

            c = m.layer1[0].bn1
            d = c.state_dict()
            m.layer1[0].bn1 = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
                            c.track_running_stats)
            m.layer1[0].bn1.load_state_dict(d, strict=False)


            
        
# def ofix_wideresnet(model):
#     from .wideresnet import wide_basic

    # for i, m in enumerate(model.modules()): 
    #     if type(m) is wide_basic:
    #         c = m.conv2
    #         d = c.state_dict()
    #         m.conv2 = THConv2d(c.in_channels,c.out_channels,c.kernel_size, c.stride, c.padding)#m.conv2.input_channel,m.conv2.output_channels)
    #         m.conv2.load_state_dict(d, strict=False)
    #         c = m.bn2
    #         d = c.state_dict()
    #         m.bn2 = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
    #                         c.track_running_stats)
    #         m.bn2.load_state_dict(d, strict=False)
    #     elif i == 0:
    #         c = m.conv1
    #         d = c.state_dict()
    #         m.conv1 = THConv2d(c.in_channels,c.out_channels,
    #             c.kernel_size, c.stride, c.padding, 
    #             c.dilation,
    #              c.groups, bias=False, padding_mode=c.padding_mode)
    #         m.conv1.load_state_dict(d, strict=False)
    #         # import pdb;pdb.set_trace()

    #         c = m.bn1
    #         d = c.state_dict()
    #         m.bn1 = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
    #                         c.track_running_stats)
    #         m.bn1.load_state_dict(d, strict=False)
def fix_mobilenetv2(model):
    from .mobilenetv2 import Block
    for i, m in enumerate(model.modules()): 
        if i == 0:
            c = m.conv1
            d = c.state_dict()
            m.conv1 = THConv2d(c.in_channels,c.out_channels,
                c.kernel_size, c.stride, c.padding, 
                c.dilation,
                 c.groups, bias=False, padding_mode=c.padding_mode)
            m.conv1.load_state_dict(d, strict=False)
            # import pdb;pdb.set_trace()

            c = m.bn1
            d = c.state_dict()
            m.bn1 = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
                            c.track_running_stats)
            m.bn1.load_state_dict(d, strict=False)
        elif type(m) is Block:
            
            c = m.conv2
            d = c.state_dict()
            m.conv2 = THConv2d(c.in_channels,c.out_channels,c.kernel_size, c.stride, c.padding, c.dilation,c.groups)#m.conv2.input_channel,m.conv2.output_channels)
            m.conv2.load_state_dict(d, strict=False)
            c = m.bn2
            d = c.state_dict()
            m.bn2 = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
                            c.track_running_stats)
            m.bn2.load_state_dict(d, strict=False)

            c = m.conv3
            d = c.state_dict()
            m.conv3 = THConv2d(c.in_channels,c.out_channels,c.kernel_size, c.stride, c.padding)#m.conv2.input_channel,m.conv2.output_channels)
            m.conv3.load_state_dict(d, strict=False)
            c = m.bn3
            d = c.state_dict()
            m.bn3 = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
                            c.track_running_stats)
            m.bn3.load_state_dict(d, strict=False)
            # if len(m.shortcut) == 0:
            if len(m.shortcut) == 0:
                continue
            c = m.shortcut[0] # refernce, not mutate
            d = c.state_dict()
            m.shortcut[0] = THConv2d(c.in_channels,c.out_channels,c.kernel_size, c.stride, c.padding)#m.conv2.input_channel,m.conv2.output_channels)
            m.shortcut[0].load_state_dict(d, strict=False)
            c = m.shortcut[1]
            d = c.state_dict()
            m.shortcut[1] = THBatchNorm2d(c.num_features, c.eps, c.momentum, c.affine,
                            c.track_running_stats)
            m.shortcut[1].load_state_dict(d, strict=False)
        
def _reparam_layers(module, masked_module=None):
    for name, m in module._modules.items():
        # print(name)
        if type(m) is THConv2d:
            bias = True if m.bias is not None else False
            module._modules[name] = GConv2d(m.in_channels,m.out_channels,
                m.kernel_size, m.stride, m.padding, 
                m.dilation,
                 m.groups, bias=bias, padding_mode=m.padding_mode)

            masked_module = module._modules[name]
        elif type(m) is THLinear:
            bias = True if m.bias is not None else False
            module._modules[name] = GLinear(m.in_features,m.out_features, bias=bias)
            masked_module = module._modules[name]

        elif type(m) is THBatchNorm2d and masked_module is not None:
            # mask_before = masked_module.mweight
            # if mask_before.size(0) != m.weight.size(0):
            if not name == "downsample":
                mask_before = masked_module
            # if mask_before.weight.size(0) != m.weight.size(0): # TODO wideresnet
                # import pdb;pdb.set_trace()
                # continue
            module._modules[name] = GBatchNorm2d(m.num_features, m.eps, m.momentum, m.affine,
                            m.track_running_stats, mask_before=mask_before) 

        elif type(m) is THBatchNorm1d and masked_module is not None:
            # mask_before = masked_module.mweight
            # if mask_before.size(0) != m.weight.size(0):
            mask_before = masked_module
            if mask_before.weight.size(0) != m.weight.size(0):
                continue
            module._modules[name] = GBatchNorm1d(m.num_features, m.eps, m.momentum, m.affine,
                            m.track_running_stats, mask_before=mask_before) 


        # if m._modules and name != 'downsample':
        if m._modules:
            _reparam_layers(m, masked_module)


def reparam_layers(module):
    masked_module =  None
    _reparam_layers(module, masked_module)
    # classifier must not be prune in channel Pruning
    for name, m in reversed(module._modules.items()):
        if type(m) is GLinear:
            if m.filter_pruning is False:
                break
            bias = True if m.bias is not None else False
            module._modules[name] = THLinear(m.in_features,m.out_features, bias=bias)
            masked_module = module._modules[name]
            break