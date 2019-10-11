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
from compute_flops import print_model_param_flops
import models
import time
import numpy as np
REF_PARAM=36.5e6
REF_FLOPS=10.49e9
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--pruned', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--freebie', action='store_true', default=False,
                    help='16-bit')
parser.add_argument('--half', action='store_true', default=False,
                    help='half-precision')
parser.add_argument('--nbit', type=int, default=32, metavar='N',
                    help='weight bit')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
else:

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]), download=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)

assert args.pruned
# model = torch.load(args.pruned)
model = torch.load(args.pruned, map_location='cpu')

if args.cuda:
    model.cuda()

print(model)
print("-"*80)
print("Freebie 16-bit is", args.freebie)
print("Pytorch half-precision is", args.half)
print("Weight-bit is", args.nbit)
base_params = sum(p.numel() for name, p in model.named_parameters())

mult = 1
if args.freebie:
  assert args.nbit == 32
  args.nbit=16
mult = 32//args.nbit
print("params count / ", mult)
params = base_params // mult
print("original param count", base_params, '@ {:.2f} M'.format(base_params/1e6))
print("quantized param count", params, '@ {:.2f} M'.format(params/1e6))
base_flops = print_model_param_flops(model, 32, multiply_adds=True, freebie=False) # only either 32 or 16 bit
flops = print_model_param_flops(model, 32, multiply_adds=True, freebie=args.freebie or args.half) # only either 32 or 16 bit
print("original FLOPs count", base_flops, '@ {:.2f} GFLOPs'.format(base_flops/1e9))
print("FLOPs count", flops, '@ {:.2f} GFLOPs'.format(flops/1e9))
print("-"*80)
print("Reference param count", REF_PARAM, '@ {:.2f} M'.format(REF_PARAM/1e6))
print("Reference FLOPs count", REF_FLOPS, '@ {:.2f} GFLOPs'.format(REF_FLOPS/1e9))
param_score = params/REF_PARAM
flops_score = flops/REF_FLOPS
score = param_score + flops_score
print("-"*80)
print("Score = Param/refParam + FLOPs/refFLOPs = {:.5f} + {:.5f} = {:.5f}".format(param_score, flops_score, score))
print("-"*80)
if args.half:
    model = model.half()
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.half:
            data = data.half()
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return float(correct) / float(len(test_loader.dataset))



test()