# Training runs: commands and logs

This file lists, for every published checkpoint and for the comparison runs cited in the README, the `train.py` command that sets the same training recipe and the log of the original run.

The commands repeat the arguments that `train.py` stored with each run, checked against the run headers in [`logs/`](logs). Only the data and output folders differ, because the original runs used paths on the rented machines. Each run used seed 42 (the default of `--seed`). cuDNN benchmark mode is on (PyTorch then picks the fastest convolution algorithm at run time), so a repeated run will not give bit-for-bit the same weights.

`python tests/test_smoke.py` (or `python -m pytest tests -q`) checks that `train.py` accepts every command below.

## Published checkpoints and comparison runs

File names are those on [huggingface.co/Artem3000-dev/cifar-10-sknet](https://huggingface.co/Artem3000-dev/cifar-10-sknet). Their SHA-256 checksums are in [`SHA256SUMS`](SHA256SUMS). "Best during training" is the highest test accuracy over the epochs, raw or EMA weights, as printed at the end of the log. The notebook evaluates the same weights in bfloat16 on a different GPU and gets slightly different numbers (README, "Limitations").

| Run | Files on Hugging Face | Role | Log | Best during training | Time, GPU |
|---|---|---|---|---|---|
| A. I2 base | `I2_historical_ckpt.pt`, `I2_base_stats.pt` | starting point of `ft_I2` | `base_I2.log` | 98.10% (raw, epoch 491) | 277.5 min, RTX 5090 |
| B. I1 base | `I1_historical_ckpt.pt` | ensemble member `I1_historical` | `base_I1.log` | 97.99% (raw, epoch 499) | 300.3 min, RTX 5090 |
| C. 1.1 base, 5 warmup epochs | `ckpt_run_1_1.pt`, `stats_run_1_1.pt` | starting point of `ft_1_1`, and the 1.1 run of the layout comparison in the README | `base_1_1_warmup5.log` | 97.92% (EMA, epoch 489) | 323.6 min, RTX 5090 |
| D. 1.1 base, 25 warmup epochs and MixUp | none | starting point of `ft_g0` | `base_1_1_warmup25_mixup.log` | 97.84% (raw, epoch 473) | 329.7 min, RTX 5090 |
| E. `ft_I2` | `ft_I2_ckpt.pt`, `ft_I2_stats.pt` | ensemble member, best single model | `finetune_I2.log` | 98.29% (EMA, epoch 131), SWA average 97.24% | 245.3 min, RTX 5090 |
| F. `ft_g0` | `ft_g0_ckpt.pt` | ensemble member | `finetune_g0.log` | 98.25% (raw, epoch 105), SWA average 97.39% | 250.5 min, RTX 5090 |
| G. `ft_1_1` | `ft_1_1_ckpt.pt` | ensemble member | `finetune_1_1.log` | 98.00% (EMA, epoch 148), SWA average 97.02% | 251.3 min, RTX 5090 |
| H. I1 fine-tune, SWA only | none | not used (97.97% against 97.99% for B) | `finetune_I1_swa_only.log` | 97.97% (EMA, epoch 122), SWA average 97.21% | 89.7 min, RTX 5090 |
| I. 2.1, 500 epochs | none | comparison | `compare_2_1_500ep.log` | 97.78% (EMA, epoch 486) | 223.7 min, RTX 5090 |
| J. S2 (I2 architecture without MixUp), 500 epochs | none | comparison | `compare_wider_no_mixup_500ep.log` | 97.80% (raw, epoch 487) | 275.7 min, RTX 5090 |

From epoch 21 to 85 the log of J prints a block-bypass warning: in at least one block the largest branch BatchNorm weight fell to 0.011 (epoch 66) and rose above the warning threshold of 0.05 again after epoch 85. The log of A prints the warning once (epoch 23, 0.049).

Each published file has the same SHA-256 as the file that the run in its row wrote (checked against the author's copies of the run folders).

The fine-tuned checkpoints (E, F, G) also hold the SWA average under the key `swa_state`, which is why they are about twice the size of the base checkpoints. The weights the notebook uses are under `state_dict`.

## Commands

Run from the repository root. `train.py` writes `ckpt_run_<key>.pt`, `stats_run_<key>.pt`, `summary_run_<key>.json`, diagnostic plots and, every 50 epochs, `live_snapshot_<key>.pt` into `--out-dir`, where `<key>` is the `--run` value with the dot replaced by an underscore.

### Base runs (500 epochs)

A. I2 base:

```
python train.py --run I2 --epochs 500 --warmup-epochs 5 --mixup-prob 0.5 --out-dir runs/I2_base
```

B. I1 base (DropPath is part of the I1 configuration):

```
python train.py --run I1 --epochs 500 --warmup-epochs 5 --out-dir runs/I1_base
```

C. 1.1 base with 5 warmup epochs:

```
python train.py --run 1.1 --epochs 500 --warmup-epochs 5 --out-dir runs/1_1_base_warmup5
```

D. 1.1 base with 25 warmup epochs and MixUp:

```
python train.py --run 1.1 --epochs 500 --warmup-epochs 25 --mixup-prob 0.5 --num-workers 6 --out-dir runs/1_1_base_warmup25_mixup
```

### Fine-tunes (150 epochs, starting from a base checkpoint)

`--init-from` can also point to the published base checkpoint instead of a retrained one: `I2_historical_ckpt.pt` for E, `ckpt_run_1_1.pt` for G and `I1_historical_ckpt.pt` for H. The base of F is not published and has to be trained first with command D. SAM switches mixed precision off, so E, F and G run in 32-bit precision. H has no SAM and runs in bfloat16 like the base runs.

E. `ft_I2`:

```
python train.py --run I2 --epochs 150 --lr 0.005 --warmup-epochs 5 --cutmix-prob 0.25 --mixup-prob 0.25 --use-sam --sam-rho 0.05 --use-swa --swa-start 75 --swa-lr 0.001 --num-workers 6 --init-from runs/I2_base/ckpt_run_I2.pt --out-dir runs/ft_I2
```

F. `ft_g0`:

```
python train.py --run 1.1 --epochs 150 --lr 0.005 --warmup-epochs 5 --cutmix-prob 0.25 --mixup-prob 0.25 --use-sam --sam-rho 0.05 --use-swa --swa-start 75 --swa-lr 0.001 --num-workers 6 --init-from runs/1_1_base_warmup25_mixup/ckpt_run_1_1.pt --out-dir runs/ft_g0
```

G. `ft_1_1`:

```
python train.py --run 1.1 --epochs 150 --lr 0.005 --warmup-epochs 5 --cutmix-prob 0.25 --mixup-prob 0.25 --use-sam --sam-rho 0.05 --use-swa --swa-start 75 --swa-lr 0.001 --num-workers 6 --init-from runs/1_1_base_warmup5/ckpt_run_1_1.pt --out-dir runs/ft_1_1
```

H. I1 fine-tune with SWA only (no SAM, DropPath stays on):

```
python train.py --run I1 --epochs 150 --lr 0.005 --warmup-epochs 5 --cutmix-prob 0.25 --mixup-prob 0.25 --use-swa --swa-start 75 --swa-lr 0.001 --num-workers 6 --init-from runs/I1_base/ckpt_run_I1.pt --out-dir runs/ft_I1
```

### Comparison runs

I. Layout 2.1, 500 epochs:

```
python train.py --run 2.1 --epochs 500 --warmup-epochs 5 --out-dir runs/2_1_500ep
```

J. S2, the I2 architecture without MixUp, 500 epochs. S2 is not among the configurations of `sknet/lib.py`. It had the same architecture as I2 (only the name differed), so the command uses `--run I2`. The stored arguments of the original run say `run: S2` and match the command below in every other field. The original run saved `ckpt_run_S2.pt`, the command below saves `ckpt_run_I2.pt`:

```
python train.py --run I2 --epochs 500 --warmup-epochs 5 --out-dir runs/S2_500ep
```

### Evaluation

`evaluate.py` downloads the four ensemble members, checks their SHA-256 and prints the accuracy of each member and of the ensemble, as Section 13 of the notebook does (README, "How to reproduce"):

```
python evaluate.py
```

`tta_eval.py` evaluates one checkpoint with several kinds of test-time augmentation (flips, fixed crops, random crops):

```
python tta_eval.py --ckpt weights/ft_I2_ckpt.pt
```
