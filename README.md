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

Run this to quantize.
```shell
python quantize_model.py --pruned log-w1/pruned_model.pth.tar --wbit 8 --save log-w1

```
After quantization, `quantized_model.pth.tar` is stored in `--save` folder `log-w1`.

## To evaluate models
- Pretrained models are in `chkpt/` folder.
- Mult ops and params are halved to be 16-bit with `--freebie`.
- `--half` to use Pytorch half-precision.
- `--nbit N` to specify bit length, params will be divided by `32/nbit`
- `compute_flops.py` is the code to calculate FLOPs
- `base_params = sum(p.numel() for name, p in model.named_parameters())`

8-bit (best score)
```
python eval.py --pruned chkpt/quantized_model.pth.tar  --half --nbit 8

Output:
--------------------------------------------------------------------------------
Freebie 16-bit is False
Pytorch half-precision is True
Weight-bit is 8
params count /  4
original param count 23181353 @ 23.18 M
quantized param count 5795338 @ 5.80 M
original FLOPs count 10625795100.0 @ 10.63 GFLOPs
FLOPs count 5318413340.0 @ 5.32 GFLOPs
--------------------------------------------------------------------------------
Reference param count 36500000.0 @ 36.50 M
Reference FLOPs count 10490000000.0 @ 10.49 GFLOPs
--------------------------------------------------------------------------------
Score = Param/refParam + FLOPs/refFLOPs = 0.15878 + 0.50700 = 0.66577
--------------------------------------------------------------------------------

Test set: Average loss: 1.2828, Accuracy: 8028/10000 (80.28%)

```

16-bit freebie
```
python eval.py --pruned chkpt/pruned_model.pth.tar  --freebie --half

--------------------------------------------------------------------------------
Freebie 16-bit is True
Pytorch half-precision is True
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

Test set: Average loss: 1.2811, Accuracy: 8046/10000 (80.46%)
```