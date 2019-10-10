from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
import models
from tqdm import tqdm
import collections
import time
import pickle
import numpy as np
from functools import partial

import utils
import copy

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def mixup_data(x, y, alpha=0.4, cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if cuda:
        index.cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
torch_dtypes = {
    'float': torch.float,
    'float32': torch.float32,
    'float64': torch.float64,
    'double': torch.double,
    'float16': torch.float16,
    'half': torch.half,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'short': torch.short,
    'int32': torch.int32,
    'int': torch.int,
    'int64': torch.int64,
    'long': torch.long
}


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output

def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean',
                  smooth_eps=None, smooth_dist=None, from_logits=True):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1. - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss



parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--mwd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')

parser.add_argument('--maxth', default=0.05, type=float,
                    help='depth of the neural network')
parser.add_argument('--wg', default=0, type=float,
                    help='depth of the neural network')
parser.add_argument('--wgd', default=1.00, type=float,
                    help='depth of the neural network')
parser.add_argument('--lrd', default=1.0, type=float,
                    help='decay lr per peoch')
parser.add_argument('--clrwme', default=1, type=float,
                    help='constant lr warmup epoch')
parser.add_argument('--wme', default=0, type=int,
                    help='warmup')
parser.add_argument('--gze', default=0.07, type=float,
                    help='gate zero epsilon')
parser.add_argument('--tw', default=1 , type=float,
                    help='gate zero epsilon')
parser.add_argument('--twd', default=1.0, type=float,
                    help='depth of the neural network')
parser.add_argument('--pt', default=0.90, type=float,
                    help='depth of the neural network')
parser.add_argument('--temp', default=1, type=float,
                    help='depth of the neural network')
parser.add_argument('--op', default='mul', type=str, 
                    help='path to save prune model (default: current directory)')
parser.add_argument('--op2', default='abs', type=str, 
                    help='path to save prune model (default: current directory)')
parser.add_argument('--op2arg', default=2, type=float, 
                    help='path to save prune model (default: current directory)')
parser.add_argument('--optim', default='sgd', type=str, 
                    help='path to save prune model (default: current directory)')
parser.add_argument('--rmdropout', default=0, type=float,
                    help='residual mask dropout')
parser.add_argument('--target', default=None, type=float,
                    help='depth of the neural network')
parser.add_argument('--htarget', default=None, type=float,
                    help='channel sparsity (hybrid')
parser.add_argument('--newlr', default=0.0, type=float,
                    help='depth of the neural network')
parser.add_argument('--reinit-sig', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--sbias', default=0.5, type=float,
                    help='adaptive scale bias loss')
parser.add_argument('--nfp', action='store_true', default=False,
                    help='no filter pruning')
parser.add_argument('--hp', action='store_true', default=False,
                    help='hybrid pruning')
parser.add_argument('--teps', default=1e-2, type=float,
                    help='hard threshold epsilon')
parser.add_argument('--lawm', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--trained', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lseps', default=0, type=float,
                    help='label smoothing epsilon')
parser.add_argument('--confp', default=0, type=float,
                    help='confidence penalty')

parser.add_argument('--wg2', default=None, type=float,
                    help='depth of the neural network')
parser.add_argument('--tw2', default=None, type=float,
                    help='depth of the neural network')

parser.add_argument('--initfromtea', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--skipteaload', action='store_true', default=False,
                    help='skip load from teacher')
parser.add_argument('--minit', default=None, type=float,
                    help='label smoothing epsilon')
parser.add_argument('--crint', default=50, type=int,
                    help='warmup')
parser.add_argument('--crema', default=0.9, type=float,
                    help='warmup')
parser.add_argument('--ltea', default=0, type=float,
                    help='warmup')
parser.add_argument('--ftarget', default=None, type=float,
                    help='flops target')
parser.add_argument('--expand-mask', action='store_true',
                    help='expand')
parser.add_argument('--no-ma', action='store_true', default=False,
                    help='flops multiply adds')

parser.add_argument('--alt-sparse', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pflops', action='store_true',
                    help='prune flops instead of params')
parser.add_argument('--platency', action='store_true',
                    help='prune latency instead of params')

parser.add_argument('--stats-int', type=int, default=0, metavar='N',
                    help='stats interval')
parser.add_argument('--stats-start', type=int, default=0, metavar='N',
                    help='stats start save epoch')
parser.add_argument('--stats-end', type=int, default=0, metavar='N',
                    help='stats end save epoch')
parser.add_argument('--stats-latency', action='store_true', default=False,
                    help='save latency')

parser.add_argument('--depth', default=28, type=int,
                    help='depth of the neural network')
parser.add_argument('--wf', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--d', type=float, default=0.3, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--swa', action='store_true', default=False,
                    help='swa')
parser.add_argument('--mixup', action='store_true',
                    help='mixup')
parser.add_argument('--holes', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--length', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--alpha', type=float, default=0.4, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--swa-start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa-lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa-c-epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
args = parser.parse_args()
args.flops_ma = not args.no_ma
if args.wg2 is None: args.wg2 = args.wg
if args.tw2 is None: args.tw2 = args.wg
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.pflops:
    assert not args.hp

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.initfromtea:
    if not args.skipteaload:
        print("=> loading teacher checkpoint '{}'".format(args.initfromtea))
        tcheckpoint = torch.load(args.initfromtea)
        tmodel = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, widen_factor=args.wf, dropout_rate=args.d)
        tmodel.load_state_dict(tcheckpoint['state_dict'], strict=False)
    else:
        tmodel = torch.load(args.initfromtea)


global pruned_model
pruned_model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, widen_factor=args.wf, dropout_rate=args.d)


def schedule(epoch):
    t = epoch / args.swa_start 
    lr_ratio = args.swa_lr / args.lr
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr * factor

from models import weight_gate
weight_gate.GLinear.temp = args.temp
weight_gate.GConv2d.temp = args.temp

#override dropout
weight_gate.GLinear.dropout = args.rmdropout
weight_gate.GConv2d.dropout = args.rmdropout

#override filter pruning
weight_gate.GConv2d.filter_pruning = not args.nfp
weight_gate.GConv2d.hybrid_pruning = args.hp
weight_gate.GLinear.filter_pruning = not args.nfp
weight_gate.GLinear.hybrid_pruning = args.hp

if ( args.arch == 'vgg' and args.depth == 16): # vs Pruning Filter paper
    weight_gate.GLinear.mult_add = 1
    weight_gate.GConv2d.mult_add = 1



if not os.path.exists(args.save):
    os.makedirs(args.save)

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, widen_factor=args.wf, dropout_rate=args.d)

weight_gate.reparam_layers(model)
if args.arch == 'resnet':
    weight_gate.fix_resnet(model)
elif args.arch == 'vgg':
    if args.depth == 16:
        weight_gate.fix_vgg16(model)
    # else:
    #     weight_gate.fix_vgg(model)
elif args.arch == 'preresnet':
    weight_gate.fix_preresnet(model)
    # model = model.cuda()
elif args.arch == 'wideresnet':
    weight_gate.fix_wideresnet(model)
if args.arch == 'mobilenetv2':
    weight_gate.fix_mobilenetv2(model)
# weight_gate.fix_bn(model)

if args.initfromtea and not args.skipteaload:
    print('loading teacher weight to student')
    model.load_state_dict(tcheckpoint['state_dict'], strict=False)
# else:
#     model.apply(weight_gate.initialize_weights)

if args.minit:
    from functools import partial
    iw = partial(weight_gate.initialize_mweights, s=(args.minit)/args.temp)
    model.apply(iw)
    # if (args.arch == 'vgg' and args.depth == 16):
    #     iw = partial(weight_gate.initialize_mweights, s=(0.5)/args.temp)
    #     model.classifier.apply(iw)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           # transforms.Resize(edepth2res[args.depth]) if args.arch=='efficientnet' else transforms.Pad(0),
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           # transforms.Resize(edepth2res[args.depth]) if args.arch=='efficientnet' else transforms.Pad(0),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           # transforms.Resize(edepth2res[args.depth]) if args.arch=='efficientnet' else transforms.Pad(0),
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           Cutout(n_holes=args.holes, length=args.length)
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           # transforms.Resize(edepth2res[args.depth]) if args.arch=='efficientnet' else transforms.Pad(0),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

swa_model = copy.deepcopy(model)
swa_n = 0

if args.cuda:
    model.cuda()
    if args.initfromtea:
        tmodel.cuda()
    swa_model.cuda()
    pruned_model.cuda()


print(model)
print(args)


params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)
mparams=  sum(p.numel() for name, p in model.named_parameters() if '.mbias' in name or '.mweight' in name)
# base_params = params - mparams
base_params = sum(p.numel() for name, p in pruned_model.named_parameters() if p.requires_grad)

# hack to support vgg batchnorm1d
BATCH_SIZE = 1
if args.arch == 'vgg' and args.depth == 16:
    BATCH_SIZE = 2
    weight_gate.BATCH_SIZE = BATCH_SIZE
model(torch.randn(weight_gate.BATCH_SIZE,3,32,32).cuda())
base_flops, _ = weight_gate.print_model_param_flops(pruned_model, 32, multiply_adds=args.flops_ma)


if args.stats_latency:
    lats = []
    for i in range(100):
        _, lat = weight_gate.print_model_param_flops(pruned_model, 32, return_latency=True)#, batch_size=weight_gate.BATCH_SIZE)
        if i > 20: # first early passes has memory latency
            lats.append(lat)
    base_latency = sum(lats)/len(lats)
    print("Base Latency: {:.2f} ms".format(base_latency/1e-3))

_, _, base_layer_stats = weight_gate.print_model_param_flops(pruned_model, 32, 
    multiply_adds=args.flops_ma, return_layer_stats=True)
# a=weight_gate.get_paramc_flops(pruned_model, base_stats=None, epoch=None, iteration=None, batch_size=None, multiply_adds=args.flops_ma)
relative_layer_stats = weight_gate.get_relative_layer_stats(
    pruned_model, model, base_layer_stats, multiply_adds=args.flops_ma)
# import pdb;pdb.set_trace()
# print("param count", params)


print("param count", params, '@ {:.2f} M'.format(params/1e6))
print("FLOPs count", base_flops, '@ {:.2f} GFLOPs'.format(base_flops/1e9))

print("mparam count", mparams)
print("param-mparam", params-mparams)
# print("flops2", weight_gate.print_model_param_flops(model, 32, multiply_adds=args.flops_ma))
if args.optim == "sgd":
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if 'mweight' not in name], 'weight_decay:':args.weight_decay}, 
        {'params': [param for name, param in model.named_parameters() if 'mweight' in name], 'weight_decay': args.mwd}], 
        lr=args.lr, momentum=args.momentum)
elif args.optim == "adam":
    optimizer = optim.AdamW([
        {'params': [param for name, param in model.named_parameters() if 'mweight' not in name], 'weight_decay:':args.weight_decay}, 
        {'params': [param for name, param in model.named_parameters() if 'mweight' in name], 'weight_decay': args.mwd}], 
        lr=args.lr)

# if args.swa:
#     optimizer = torchcontrib.optim.SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)


print(args.optim)
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

history_score = np.zeros((args.epochs - args.start_epoch + 1, 3))

def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def train(epoch, cr=0, fcr=None,hcr=None, wg=30, tw=30, sparser=False, stats_list=None): # 0.5
    t0 = time.time()

    model.train()
    
    if args.initfromtea:
        tmodel.train()

    global history_score
    avg_loss = 0.
    train_acc = 0.

    
    delta_target = args.target - cr
    # hdelta_target = args.htarget - hcr if args.hp else None
    fdelta_target = args.ftarget - fcr if args.ftarget else None
    print('dt',repr(delta_target))
    for batch_idx, (data, target) in enumerate(train_loader):
        # until = 13 if ( args.arch == 'vgg' and args.depth == 16) else None
        if batch_idx % args.crint == 0: 
        # if False:
            weight_gate.update_pruned_model(pruned_model, model)
            if args.pflops:
                # if batch_idx == 20: import pdb;pdb.set_trace()
                cr_flops, _ = weight_gate.print_model_param_flops(pruned_model, 32, multiply_adds=args.flops_ma)
                new_cr = 1.0-(cr_flops/base_flops)
                cr = cr + (new_cr - cr)*args.crema # exponential moving average
                delta_target = args.target - cr
                # print(cr_flops)
            elif args.platency:
                _, cr_latency = weight_gate.print_model_param_flops(pruned_model, 32, multiply_adds=args.flops_ma, return_latency=True)
                new_cr = 1.0-(cr_latency/base_latency)
                cr = cr + (new_cr - cr)*args.crema # exponential moving average
                delta_target = args.target - cr
            else:
                cr_params = sum(p.numel() for name, p in pruned_model.named_parameters() if p.requires_grad)
                new_cr = 1.0-(cr_params/base_params)
                cr = cr + (new_cr - cr)*args.crema # exponential moving average
                delta_target = args.target - cr
       
            if args.ftarget:
                cr_flops, _ = weight_gate.print_model_param_flops(pruned_model, 32, multiply_adds=args.flops_ma)
                new_fcr = 1.0-(cr_flops/base_flops)
                fcr = fcr + (new_fcr - fcr)*args.crema # exponential moving average
                fdelta_target = args.ftarget - fcr
            # if args.htarget:
            #     hcr_dict = weight_gate.get_prune_ratio(model, until=until)
            #     new_hcr = 1.0-(sum(cr_dict['activeflops'].values())/sum(cr_dict['totalflops'].values()))
            #     hcr = hcr + (new_hcr - hcr)*args.crema # exponential moving average
            #     hdelta_target = args.htarget - hcr

            # if args.trained:
            #     print(model.feature[0].ptemp)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # output2 = pruned_model(data)
        if args.mixup:
            data, targets_a, targets_b, lam = mixup_data(
                data, target, args.alpha, args.cuda)


            loss1 = mixup_criterion(
                partial(cross_entropy,smooth_eps=args.lseps), output, targets_a, targets_b, lam)
        else:
            loss1 = cross_entropy(output, target, smooth_eps=args.lseps)

        


        loss1 = cross_entropy(output, target, smooth_eps=args.lseps)
        # loss1 = F.cross_entropy(output, target)
        loss =  torch.zeros_like(loss1)
        # loss += loss1
        loss1b = (F.softmax(output, dim=1) * F.log_softmax(output, dim=1)).sum()/output.size(0)
        loss += args.confp*loss1b
        if args.initfromtea and args.ltea>0:
            loss1c = -(F.softmax( tmodel(data).detach(), dim=1)* F.log_softmax(output, dim=1)).sum()/output.size(0)
            loss +=  (args.ltea)*loss1c + (1-args.ltea)*(loss1)
        else:
            loss += loss1 

        
        # better accuracy but failed to sparse early layers.
        # maybe because channel expansion bias:
        # later layers have more filters, expaned 100x bigger than early layers
        # masks = weight_gate.get_masks(model, as_list=True, expand=True) 

        # # no expansion, masks use channel size, more uniform sparsity but worse accuracy.
        masks = weight_gate.get_masks(model, as_list=True, expand=args.expand_mask)

        # masks = masks[:until]
        masks = torch.cat(masks)
        loss2 = masks.mean()
        loss3 = masks.var() / loss2
        loss3b = masks.std() / loss2

        if args.ftarget:
            if args.hp:
                masksf = weight_gate.get_masks2(model, as_list=True, expand=args.expand_mask)
            else:
                masksf = masks = weight_gate.get_masks(model, as_list=True, expand=False)
            # masksf = masksf[:until]
            masksf = torch.cat(masksf)
            # maskf = mask
            loss2f = masksf.mean()
            loss3f = masksf.var() / loss2


        # if args.htarget:
        #     masks2 = weight_gate.get_masks2(model, as_list=True, expand=True)
        #     masks2 = masks2[:until]
        #     masks2 = torch.cat(masks2)
        #     loss2hp = masks2.mean()
        #     loss3hp = masks2.var() / loss2hp

        if sparser:
            mult = delta_target
            loss += mult  *wg * loss2# * math.sin(2*math.pi*(batch_idx/len(train_loader)))
            loss -= mult * tw * loss3# * math.sin(2*math.pi*(batch_idx/len(train_loader)))
            if args.ftarget:
                fmult = fdelta_target
                
                loss += fmult *args.wg2 * loss2f
                loss -= fmult *args.tw2 * loss3f

            # if args.hp:
            #     hmult = hdelta_target
            #     loss += hmult *args.wg2 * loss2hp
            #     loss -= hmult *args.tw2 * loss3hp
        # else:



        avg_loss +=  loss1.item()
        pred = output.data.max(1, keepdim=True)[1]

        if args.mixup:
            train_acc += (lam * pred.eq(targets_a).sum().item() \
                        + (1 - lam) * pred.eq(targets_b).sum().item())
        else:
            train_acc += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        # train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        # if not sparser:
        #     # print('reset mgrad')
        #     weight_gate.zero_mgrad(model)
        loss.backward() # grad last for two iter

        # if not sparser:
        #     # print('reset mgrad')
        #     weight_gate.zero_mgrad(model)
            # import pdb;pdb.set_trace()

        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\txent: {:.4f}\touth: {:.4f}\tmean: {:.4f}\tvar/mean: {:.5f}\tstd/mean: {:.5f}\tdelta_target: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss1.item(), loss1b.item(), loss2.item(), loss3.item(), loss3b.item(), delta_target*100))
            if args.ftarget:
                print('f_mean: {:.4f}\tf_var/f_mean: {:.5f}\tflops_delta_target: {:.2f}%'.format(
                    loss2f.item(), loss3f.item(), fdelta_target*100))
        if args.stats_start <= epoch < args.stats_end and\
                    batch_idx % args.stats_int == 0:
            rls = weight_gate.get_relative_layer_stats(
                    pruned_model=pruned_model, train_model=model, base_layer_stats=base_layer_stats, 
                    epoch=epoch,  iteration=batch_idx+epoch*len(train_loader), batch_size=args.batch_size,
                    multiply_adds=args.flops_ma, return_latency=args.stats_latency, training = True)
            rls['loss'] = loss1.item()
            rls['sparsity'] = new_cr
            rls['delta_target'] = delta_target

            if args.ftarget:
                rls['flops_sparsity'] = new_fcr
                rls['fdelta_target'] = fdelta_target
            if args.stats_latency:
                latency = rls['latency']
                print("Latency: {:.2f} ms / {:.2f} ms".format(latency/1e-3, base_latency/1e-3))
            stats_list.append(rls)

    print('{} seconds'.format(time.time() - t0))
    train_acc_pct = train_acc/len(train_loader.dataset)*100
    print(epoch, '{:.2f}% {}/{} correct'.format(train_acc_pct, train_acc, len(train_loader.dataset)))
    # import pdb;pdb.set_trace()
    return cr, fcr, hcr, stats_list

def test(prefix="",model=model, base_flops=base_flops, base_params=base_params, test_stats_list=None, train_model=None,epoch=None):
    model.eval()
    # if model == pruned_model: model.train()
    # model.train()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        # if model == pruned_model:import pdb;pdb.set_trace()
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    if model == pruned_model:
        # print('yes')
        flops, _ = weight_gate.print_model_param_flops(model, 32, multiply_adds=args.flops_ma)
        params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)

        if args.stats_start <= epoch < args.stats_end and stats_list is not None:
            rls = weight_gate.get_relative_layer_stats(
                    pruned_model=pruned_model, train_model=train_model, base_layer_stats=base_layer_stats, 
                    epoch=epoch,  iteration=(epoch+1)*len(train_loader), batch_size=args.batch_size,
                    multiply_adds=args.flops_ma, return_latency=args.stats_latency, training=False)
            rls['loss'] = test_loss
            rls['accuracy'] = float(correct) / float(len(test_loader.dataset))
            rls['sparsity'] = 1-(params/base_params)
            rls['flops_sparsity'] = 1-(flops/base_flops)
            rls['delta_target'] = args.target - rls['sparsity']
            if args.ftarget:
                rls['fdelta_target'] = args.ftarget - rls['flops_sparsity']
            if args.stats_latency:
                latency = rls['latency']
                print("Test Latency: {:.2f} ms / {:.2f} ms".format(latency/1e-3, base_latency/1e-3))
            test_stats_list.append(rls)

    else:
        flops, params = base_flops, base_params

    print('{}\tTest loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), {:.2e}/{:.2e} FLOPs ({:.2f}% pruned); {:.2f}/{:.2f} M params ({:.2f}% pruned)'.format(
        prefix,
        test_loss, correct, len(test_loader.dataset),
        100.0 * float(correct) / float(len(test_loader.dataset)),
        flops, base_flops, (1-flops/base_flops) * 100,
        params*1e-6, base_params*1e-6, (1-params/base_params) * 100

        )
    )
# print(base_flops)
    return float(correct) / float(len(test_loader.dataset)), test_stats_list

def save_checkpoint(state, is_best, filepath, postfix=''):
    torch.save(state, os.path.join(filepath, 'checkpoint{}.pth.tar'.format(postfix)))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
        torch.save(pruned_model, os.path.join(filepath, 'pruned_model.pth.tar'))

def format_matrix(header, matrix,
                  top_format, left_format, cell_format, row_delim, col_delim):
    table = [[''] + header] + [[name] + row for name, row in zip(header, matrix)]
    table_format = [['{:^{}}'] + len(header) * [top_format]] \
                 + len(matrix) * [[left_format] + len(header) * [cell_format]]
    col_widths = [max(
                      len(format.format(cell, 0))
                      for format, cell in zip(col_format, col))
                  for col_format, col in zip(zip(*table_format), zip(*table))]
    return row_delim.join(
               col_delim.join(
                   format.format(cell, width)
                   for format, cell, width in zip(row_format, row, col_widths))
               for row_format, row in zip(table_format, table))


def print_current_sparsity(cr_dict, prefix=''):
    totalsum = sum(cr_dict['totals'].values())
    prunedsum = totalsum - sum(cr_dict['nonzeros'].values())
    chtotalsum = sum(cr_dict['chtotals'].values())
    chprunedsum = chtotalsum - sum(cr_dict['chnonzeros'].values())
    totalflopssum = sum(cr_dict['totalflops'].values())
    prunedflopssum = totalflopssum - sum(cr_dict['activeflops'].values())
    pcr = prunedsum/totalsum
    chcr = chprunedsum/chtotalsum
    flopsr = prunedflopssum/totalflopssum
    print(prefix+' Parameter sparsity\t{:.2f}%\t(active {:.2f}M / {:.2f}M)'.format(
        pcr*100,
        (totalsum - prunedsum)*1e-6,
        (totalsum)*1e-6,
        ))
    print(prefix+' Channel sparsity\t{:.2f}%\t(active {} /{})'.format(
        chcr*100,
        (chtotalsum - chprunedsum),
        (chtotalsum),
        ))
    print(prefix+' FLOPs sparsity\t{:.2f}%\t(active {:.2e} / {:.2e})'.format(
        flopsr*100,
        (totalflopssum - prunedflopssum),
        (totalflopssum),
        ))
    # if args.htarget:
    #     pretotalsum = sum(cr_dict['totals'].values())
    #     preprunedsum = totalsum - sum(cr_dict['nonzeros'].values())
    #     prepcr= preprunedsum/pretotalsum
    #     print(prefix+' PreCh Hybrid sparsity\t{:.2f}%\t(active params {:.2f}M / {:.2f}M)'.format(
    #     prepcr*100,
    #     (retotalsum - preprunedsum)*1e-6,
    #     (pretotalsum)*1e-6,
    #     ))
    print()

best_prec1 = 0.
fixed = False

wg = args.wg
tw = args.tw
warmup_epochs = args.wme
sparser = True

# mode = "all"
cr = 0.
fcr = 0. if args.ftarget else None
hcr = 0. if args.htarget else None
print(wg, tw)

stats_list = []
test_stats_list = []
swa_stats_list = []

lr_pow = 1
for epoch in range(args.start_epoch, args.epochs):
    # epts = [80, 120, 200, 300] if not args.initfromtea else [10, 20, 30]
    # # epts = [60, 120, 180,300] if not args.initfromtea else [40, 80, 120]
    # if epoch in epts:
    #     for param_group in optimizer.param_groups:
    #         # param_group['lr'] *= 0.1
    #         param_group['lr'] = args.lr*math.pow(0.1, lr_pow)
    #         # param_group['lr'] = args.lr*math.pow(0.2, lr_pow)
    #         lr_now = param_group['lr']
    #     lr_pow += 1
    #     print("lr = {:.5f}".format(lr_now), 'at epoch', repr(epoch))


    # if epoch > args.clrwme: # begin decay
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= args.lrd
    #         wg *= args.wgd
    #         tw *= args.twd
    #         lr_now = param_group['lr']
    #     print("lr = {:.5f}".format(lr_now), 'at epoch', repr(epoch))
    #     print("(wg, tw, cr) = ", repr((lr_now, wg, tw, cr)))

    lr = schedule(epoch)
    # utils.adjust_learning_rate(optimizer, lr)
    # print("lr = {:.5f}".format(lr), 'at epoch', repr(epoch))

    sparser = True
    if (epoch < warmup_epochs):
        sparser= False
    elif args.alt_sparse:
        # if epoch % 2 == 0:
        #     args.wg = -args.wg
        #     args.wg2 = -args.wg2
        #     args.tw = -args.tw
        #     args.tw2 = -args.tw2

        sparser = False if epoch % 2 == 0 else True

    # mode = "weight" if (epoch < warmup_epochs or not sparser) else "all"
    # weight_gate.set_trainable(model, mode)
    print(sparser)
    # print(mode)
    cr, fcr, hcr, stats_list = train(epoch, sparser=sparser, stats_list=stats_list, wg=wg, tw=tw, cr=cr, fcr=fcr,hcr=hcr)



    print()
    epoch_str = "Epoch "+repr(epoch)
    test("Training "+epoch_str+" Test")
    print()
    prec1, test_stats_list =test("Pruned "+epoch_str+" Test", model=pruned_model, train_model=model, test_stats_list=test_stats_list, epoch=epoch)
    
    if epoch >= args.swa_start:
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
        swa_n += 1
        weight_gate.update_pruned_model(pruned_model, swa_model)

        print('start update bn')
        t0 = time.time()
    # utils.bn_update(train_loader, pruned_model,pct=0.1)
        utils.bn_update(train_loader, pruned_model)
        print('bn update took {} seconds'.format(time.time() - t0))
        prec1, swa_stats_list = test("SWA "+epoch_str+" Test", model=pruned_model, train_model=model, test_stats_list=swa_stats_list, epoch=epoch)
    
    # try:
    #     prec1 = test("P "+epoch_str+" Test", model=pruned_model)
    # except Exception as e:
    #     print("Model failed to test, probably 0-dim weights exist:", e)
    print()
    cr_dict = weight_gate.get_prune_ratio(model)
    print_current_sparsity(cr_dict, prefix=epoch_str)


    # until = 13 if ( args.arch == 'vgg' and args.depth == 16) else None
    # # until = None
    # if until:
    #     cr_dict_until = weight_gate.get_prune_ratio(model, until=until)
    #     print_current_sparsity(cr_dict_until, prefix=epoch_str+' Until layer '+repr(until))
     
    for i in range(len(cr_dict['totals'])):
        layer_str ="({})\t{}:\t{:.2f}%\t| {:}/{:}\t| {:}/{:}\t| {:.2e}/{:.2e}".format(
            i+1, 
            list(cr_dict['totals'].keys())[i], 
            list(cr_dict['ratios'].values())[i]*100, 
            list(cr_dict['nonzeros'].values())[i], 
            list(cr_dict['totals'].values())[i], 
            list(cr_dict['chnonzeros'].values())[i], 
            list(cr_dict['chtotals'].values())[i],
            list(cr_dict['activeflops'].values())[i], 
            list(cr_dict['totalflops'].values())[i])
        # if args.htarget:
        #     layer_str += "\t| {}/{}".format(
        #     list(cr_dict['nonzeros2'].values())[i],
        #     list(cr_dict['totals2'].values())[i]
        #      )
        print(layer_str)
    print()

    # history_score[epoch][2] = prec1
    # np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'pruned_state_dict': pruned_model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        'stats_list': stats_list,
        'test_stats_list': test_stats_list,
        'swa_state_dict':swa_model.state_dict(),
        'swa_n':swa_n
    }, is_best, filepath=args.save)

print("Best accuracy: "+str(best_prec1))
history_score[-1][0] = best_prec1
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
    
