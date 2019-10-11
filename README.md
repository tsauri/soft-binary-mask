# End-to-End Pruning with Soft-binary Weight Mask
- Apply trainable channel mask on weights of WideResNet-34-9, constrained mask to [0,1]
- Sparsity regularize mask by minimizing mean `--wg` and maximizing variance `--tw`
- To prune params, multiply mask by weights
- To prune FLOPs, prune dimensions by deleting output channel dims with mask == 0
- Target sparsity percentage `--target` with value [0,1], and mask (reverse) weight decay `--mwd` to control mask unsparsification if miss `--target`
- Warmup for several epochs before start regularization with `--wme`
- Used Stochastic Weight Averaging to boost accuracy, start averaging at epoch `--swa-start`
- Tried using Confidence Penalty `--confp`, Dropout `--d`, and Label Smoothing `--lseps`, but negligible effect, so set them to 0

Pre-trained model are in `chkpt/`

## To reproduce models
Run this command to train from scratch. Training takes almost 12 hours. Single GPU only.
```shell
python -u main.py --dataset cifar100 --arch wideresnet --epochs 300 --seed 1 --batch-size 64      --lr 1e-2 --wd 1e-4 --momentum 0.9   --test-batch-size 64 --depth 34 --wf 9 --d 0 --lseps 0.0 --confp 0.0   --target 0.4  --wg 30 --tw 10 --wme 50  --mwd 5e-4  --save log-k7d1re1 --swa-start 161

```
Pruned model `pruned_model.pth.tar` is stored in `--save` folder `log-w1`

## To evaluate models
- Pretrained models are in `chkpt/` folder.

- eval.py is old eval code, for 16-bit  `--freebie`.
- eval_quantize.py is final eval code, uses Pytorch 1.3 quantization (CPU-only, very slow eval)
- `compute_flops.py` is the code to calculate FLOPs
- `base_params = sum(p.numel() for name, p in model.named_parameters())`

)

This is the final submission. If 8-bit quantization is valid
```
python eval_quantize.py --pruned chkpt/pruned_model2.pth.tar

--------------------------------------------------------------------------------
params count /  4
original param count 20696798 @ 20.70 M
8-bit quantized param count 5174199 @ 5.17 M
original FLOPs count 10193691876.0 @ 10.19 GFLOPs
8-bit FLOPs count 2548422969.0 @ 2.55 GFLOPs
--------------------------------------------------------------------------------
Reference param count 36500000.0 @ 36.50 M
Reference FLOPs count 10490000000.0 @ 10.49 GFLOPs
--------------------------------------------------------------------------------
Score = Param/refParam + FLOPs/refFLOPs = 0.14176 + 0.24294 = 0.38470
--------------------------------------------------------------------------------
Test set: Average loss: 1.2146, Accuracy: 8019/10000 (80.19%)

```

Otherwise use fallback freebie model (bigger)
```
`python eval.py --pruned chkpt/pruned_model.pth.tar  --freebie --no-cuda`

--------------------------------------------------------------------------------
Freebie 16-bit is True
Pytorch half-precision is False
Weight-bit is 32
params count /  2
original param count 23181353 @ 23.18 M
quantized param count 11590676 @ 11.59 M
original FLOPs count 10625795100.0 @ 10.63 GFLOPs
FLOPs count 5318413340.0 @ 5.32 GFLOPs
--------------------------------------------------------------------------------
Reference param count 36500000.0 @ 36.50 M
Reference FLOPs count 10490000000.0 @ 10.49 GFLOPs
--------------------------------------------------------------------------------
Score = Param/refParam + FLOPs/refFLOPs = 0.31755 + 0.50700 = 0.82455
--------------------------------------------------------------------------------

```
