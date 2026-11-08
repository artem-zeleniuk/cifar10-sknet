"""Command-line entry point: train one SKNet configuration on one GPU.

The model and the training loop come from the library in sknet/, imported under
its original module name sknet_lib (see sknet_lib.py). experiments.md lists the
commands of the published runs.

Usage
-----
    # Train one entry of RUN_CONFIGS:
    python train.py --run I2 --epochs 500 --warmup-epochs 5 --mixup-prob 0.5 --out-dir runs/I2_base
    # Several runs in parallel on a machine with several GPUs:
    CUDA_VISIBLE_DEVICES=0 python train.py --run 1.1 --epochs 500 --warmup-epochs 5 --out-dir runs/ &
    CUDA_VISIBLE_DEVICES=1 python train.py --run I1 --epochs 500 --warmup-epochs 5 --out-dir runs/ &
    CUDA_VISIBLE_DEVICES=2 python train.py --run I2 --epochs 500 --warmup-epochs 5 --out-dir runs/ &
    CUDA_VISIBLE_DEVICES=3 python train.py --run 2.1 --epochs 500 --warmup-epochs 5 --out-dir runs/ &
    wait

Outputs in <out-dir>: ckpt_run_<key>.pt (best state_dict and config, plus the SWA
weights if SWA was on), stats_run_<key>.pt (batch losses, train/test/EMA accuracy,
timings, the arguments), summary_run_<key>.json and the diagnostic plots plot_*.png.
Every 50 epochs it also overwrites live_snapshot_<key>.pt (best weights so far and
the history lists).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sknet_lib as L  # model and training loop (sknet/lib.py, see sknet_lib.py)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train one SKNet config.')
    p.add_argument('--run', required=True, choices=sorted(L.RUN_CONFIGS.keys()),
                   help='Entry of RUN_CONFIGS to train.')
    p.add_argument('--epochs', type=int, default=300,
                   help='Number of epochs (default: 300).')
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--cutmix-alpha', type=float, default=1.0)
    p.add_argument('--cutmix-prob', type=float, default=0.5)
    p.add_argument('--mixup-alpha', type=float, default=0.2,
                   help='MixUp Beta(alpha,alpha) parameter (0 disables MixUp).')
    p.add_argument('--mixup-prob', type=float, default=0.0,
                   help='Share of batches mixed with MixUp. CutMix and MixUp exclude each other, keep --cutmix-prob + --mixup-prob at most 1.')
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--data-root', type=str, default='./data')
    p.add_argument('--out-dir', type=str, default='./runs')
    p.add_argument('--no-amp', action='store_true',
                   help='Disable mixed precision (only needed for debugging).')
    # Recipe options (the defaults give the baseline recipe).
    p.add_argument('--warmup-epochs', type=int, default=0,
                   help='Linear LR warmup before cosine. 0 = disabled (default).')
    p.add_argument('--decoupled-wd', action='store_true',
                   help='Exclude BatchNorm weights and biases from weight decay (He et al. 2019, "Bag of Tricks").')
    p.add_argument('--aug-recipe', type=str, default='trivial',
                   choices=['trivial', 'autoaugment_cifar10', 'randaugment'],
                   help='Auto-augmentation operator (always combined with crop+flip+Cutout).')
    p.add_argument('--use-sam', action='store_true',
                   help='Use SAM (sharpness-aware minimisation). Two passes per step, mixed precision is switched off.')
    p.add_argument('--sam-rho', type=float, default=0.05,
                   help='SAM neighbourhood radius (default 0.05).')
    # SWA (Stochastic Weight Averaging) flags
    p.add_argument('--use-swa', action='store_true',
                   help='Average the weights from --swa-start on and recalibrate BatchNorm at the end.')
    p.add_argument('--swa-start', type=int, default=75,
                   help='Epoch at which SWA averaging begins (default 75).')
    p.add_argument('--swa-lr', type=float, default=0.001,
                   help='Constant LR during SWA phase (default 0.001).')
    # Start from an existing checkpoint (fine-tune)
    p.add_argument('--init-from', type=str, default=None,
                   help='Path to existing ckpt_run_*.pt to load weights from before training.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Line-buffer stdout so that a log piped through tee is written line by
    # line, not in 4 to 8 KB chunks.
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The library already seeds at import time. Seeding again here applies
    # the value of --seed.
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[{args.run}] device={device}, '
          f'gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"}')

    # Build the loaders for this process. The module-level loaders of the
    # library are not reused (they use num_workers=2), so that --num-workers
    # takes effect on machines with several GPUs.
    import torchvision
    train_tf = L.train_transform
    test_tf = L.test_transform
    train_ds = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=False,
    )

    cfg = L.RUN_CONFIGS[args.run]
    print(f'[{args.run}] config: {cfg}')

    # train_one_config takes no device or loader arguments. It reads the
    # module-level device, train_loader and test_loader of the library, so
    # they are set here.
    L.device = device
    L.train_loader = train_loader
    L.test_loader = test_loader

    key_safe = args.run.replace('.', '_')
    ckpt_path = out_dir / f'ckpt_run_{key_safe}.pt'
    stats_path = out_dir / f'stats_run_{key_safe}.pt'

    # Called by train_one_config after every epoch.
    # At epochs 1, 3, 5 and 10 and every 25 epochs after that it prints a
    # detailed report. Every PLOT_EVERY epochs it rewrites the diagnostic PNGs
    # in out_dir, so progress can be followed with `tail -f log` and
    # `ls runs/`.
    EARLY_CHECKPOINTS = {1, 3, 5, 10}
    CHECKPOINT_EVERY  = 25
    PLOT_EVERY        = 10

    def _is_checkpoint(ep: int) -> bool:
        return ep in EARLY_CHECKPOINTS or ep % CHECKPOINT_EVERY == 0

    def _live_callback(epoch: int, stats: dict) -> None:
        import numpy as np

        # Detailed report at the checkpoint epochs
        if _is_checkpoint(epoch):
            gn = stats['grad_norms']
            bl = stats['batch_losses']
            batches_per_epoch = max(1, len(gn) // epoch)
            recent_gn = gn[-batches_per_epoch:] if gn else []
            recent_bl = bl[-batches_per_epoch:] if bl else []
            print(f'\n=== [{args.run}] checkpoint at epoch {epoch} ===')
            print(f'    train/test_raw/test_ema: '
                  f'{stats["train_acc"]*100:5.2f}% / '
                  f'{stats["test_acc_raw"]*100:5.2f}% / '
                  f'{stats["test_acc_ema"]*100:5.2f}%')
            print(f'    best so far : {stats["best_acc"]*100:5.2f}% '
                  f'({"EMA" if stats["best_is_ema"] else "raw"})')
            print(f'    lr / elapsed: {stats["lr"]:.4f} / {stats["elapsed"]/60:.2f} min')
            if recent_gn:
                print(f'    |grad| mean/max : {float(np.mean(recent_gn)):.3f} / '
                      f'{float(np.max(recent_gn)):.3f}')
            if recent_bl:
                print(f'    loss mean/max   : {float(np.mean(recent_bl)):.3f} / '
                      f'{float(np.max(recent_bl)):.3f}')
            if stats['test_acc_raw'] < 0.15 and epoch >= 3:
                print('    WARNING: test accuracy near chance, possible bug')
            if recent_gn and float(np.max(recent_gn)) > 50:
                print('    WARNING: gradient norm above 50, clipping may not be enough')
            print(f'=== next report in at most {CHECKPOINT_EVERY} epochs ===\n')

        # Live plotting: rewrite PNGs every PLOT_EVERY epochs.
        # Same per-epoch averaging as the final plots (readable trends).
        if epoch % PLOT_EVERY == 0:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import numpy as np

                bl   = stats['batch_losses']
                gn   = stats['grad_norms']
                bpe  = stats.get('num_train_batches') or max(1, len(bl) // epoch)

                # Loss per epoch (mean of batches in epoch)
                if bl:
                    n_done = len(bl) // bpe
                    if n_done > 0:
                        a = np.array(bl[: n_done * bpe], dtype=np.float64).reshape(n_done, bpe)
                        loss_pe = np.where(np.isfinite(a), a, np.nan).mean(axis=1)
                        ee = np.arange(1, n_done + 1)
                        plt.figure(figsize=(10, 3))
                        plt.plot(ee, loss_pe, linewidth=1.5)
                        plt.xlabel('Epoch'); plt.ylabel('Mean CE loss')
                        plt.title(f'[{args.run}] LIVE loss, epoch {epoch}/{args.epochs}')
                        plt.grid(alpha=0.3); plt.tight_layout()
                        plt.savefig(out_dir / f'plot_loss_{key_safe}.png', dpi=100)
                        plt.close()

                # Gradient norms per epoch (mean)
                if gn:
                    n_done = len(gn) // bpe
                    if n_done > 0:
                        a = np.array(gn[: n_done * bpe], dtype=np.float64).reshape(n_done, bpe)
                        a = np.where(np.isfinite(a), a, np.nan)
                        gmean = np.nanmean(a, axis=1)
                        gmax  = np.nanmax(a, axis=1)
                        ee = np.arange(1, n_done + 1)
                        plt.figure(figsize=(10, 3))
                        plt.plot(ee, gmean, linewidth=1.5, label='mean')
                        plt.plot(ee, gmax,  linewidth=1.0, alpha=0.5, label='max')
                        plt.axhline(5.0, color='red', linestyle=':', alpha=0.5, label='clip')
                        plt.xlabel('Epoch'); plt.ylabel('|grad|')
                        plt.title(f'[{args.run}] LIVE |grad|, epoch {epoch}/{args.epochs}')
                        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
                        plt.savefig(out_dir / f'plot_grad_{key_safe}.png', dpi=100)
                        plt.close()
            except Exception as e:
                print(f'  (live plot failed: {e})')

        # Every 50 epochs, save the best weights so far and the history lists
        # to live_snapshot_<key>.pt (overwritten each time), so that an
        # interrupted run can still be plotted and its best weights loaded.
        if epoch % 50 == 0 and epoch > 0:
            try:
                torch.save({
                    'epoch':           epoch,
                    'best_state':      stats.get('best_state'),     # state_dict (best so far)
                    'best_acc':        stats.get('best_acc'),
                    'best_is_ema':     stats.get('best_is_ema'),
                    'train_accs':      list(stats.get('train_accs', [])),
                    'test_accs':       list(stats.get('test_accs', [])),
                    'test_accs_ema':   list(stats.get('test_accs_ema', [])),
                    'gaps':            list(stats.get('gaps', [])),
                    'batch_losses':    list(stats.get('batch_losses', [])),
                    'grad_norms':      list(stats.get('grad_norms', [])),
                    'nan_steps':       stats.get('nan_steps', 0),
                }, out_dir / f'live_snapshot_{key_safe}.pt')   # single-file (overwrites prev)
            except Exception as e:
                print(f'  (periodic snapshot save failed: {e})')

    result = L.train_one_config(
        cfg,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        label_smoothing=args.label_smoothing,
        ema_decay=args.ema_decay,
        use_amp=(not args.no_amp),
        warmup_epochs=args.warmup_epochs,
        decoupled_wd=args.decoupled_wd,
        aug_recipe=args.aug_recipe,
        use_sam=args.use_sam,
        sam_rho=args.sam_rho,
        use_swa=args.use_swa,
        swa_start_epoch=args.swa_start,
        swa_lr=args.swa_lr,
        init_from=args.init_from,
        # workers and data root also build the clean-train loader, batch size
        # only matters for a non-default augmentation loader
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        verbose=True,
        on_epoch_end=_live_callback,
    )

    # Do not save a checkpoint when best_state is None (the run stopped before
    # any improvement, for example with a NaN in epoch 1). A checkpoint with a
    # None state_dict could not be loaded later.
    if result.get('best_state') is None:
        print(f'[{args.run}] WARNING: best_state is None (run aborted before any '
              f'improvement), skipping checkpoint save. Final state was at epoch '
              f'{len(result.get("train_accs", []))} with best_acc='
              f'{result.get("best_acc", 0)*100:.2f}%.')
    else:
        torch.save({
            'cfg': result['cfg'],
            'state_dict': result['best_state'],
            'best_acc': result['best_acc'],
            'best_is_ema': result['best_is_ema'],
            'best_epoch': result.get('best_epoch'),
            # The final SWA weights are saved separately. If they scored higher
            # than every epoch, best_state above already holds them.
            'swa_acc': result.get('swa_acc'),
            'swa_state': result.get('swa_state'),
        }, ckpt_path)

    # Statistics without weights, small enough to copy from the training machine and plot
    torch.save({
        'name': result['name'],
        'n_params': result['n_params'],
        'batch_losses': result['batch_losses'],
        'grad_norms': result.get('grad_norms', []),
        'train_accs': result['train_accs'],
        'test_accs': result['test_accs'],
        'test_accs_ema': result['test_accs_ema'],
        'gaps': result.get('gaps', []),
        'train_accs_clean': result.get('train_accs_clean', []),
        'clean_eval_epochs': result.get('clean_eval_epochs', []),
        'best_acc': result['best_acc'],
        'best_is_ema': result['best_is_ema'],
        'best_epoch': result.get('best_epoch'),
        'nan_steps': result.get('nan_steps', 0),
        'aborted': result.get('aborted', False),
        'total_time': result['total_time'],
        'args': vars(args),
        'block_stats': result.get('block_stats', []),  # per-epoch block weight statistics
    }, stats_path)

    # Also write a small JSON summary for quick inspection
    summary = {
        'run': args.run,
        'name': result['name'],
        'n_params': result['n_params'],
        'best_acc': result['best_acc'],
        'best_is_ema': result['best_is_ema'],
        'best_epoch': result.get('best_epoch'),
        'swa_acc': result.get('swa_acc'),          # None if SWA disabled
        'total_time_min': result['total_time'] / 60.0,
        'epochs': args.epochs,
        'nan_steps': result.get('nan_steps', 0),
        'aborted': result.get('aborted', False),
    }
    with open(out_dir / f'summary_run_{key_safe}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Diagnostic plots (headless, written as PNG files so the health of a
    # run can be checked without the notebook).
    # Loss and gradient plots show averages per epoch, not per batch: 117k
    # batch points are unreadable, while 300 epoch averages show the trend.
    # The raw per-batch arrays are saved in stats_run_<key>.pt for a closer
    # look. ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        n_ep = len(result['train_accs'])
        if n_ep == 0:
            # No epoch was completed, so there is nothing to plot.
            print(f'[{args.run}] no epoch data, skipping plots')
            raise StopIteration
        eps  = np.arange(1, n_ep + 1)

        def per_epoch_mean(arr):
            if n_ep == 0:
                return np.array([])
            a = np.array(arr, dtype=np.float64)
            a = np.where(np.isfinite(a), a, np.nan)
            bpe = max(1, len(a) // n_ep)
            a = a[: bpe * n_ep].reshape(n_ep, bpe)
            return np.nanmean(a, axis=1)

        def per_epoch_max(arr):
            if n_ep == 0:
                return np.array([])
            a = np.array(arr, dtype=np.float64)
            a = np.where(np.isfinite(a), a, np.nan)
            bpe = max(1, len(a) // n_ep)
            a = a[: bpe * n_ep].reshape(n_ep, bpe)
            return np.nanmax(a, axis=1)

        # Loss per epoch (mean of all batches in the epoch)
        if result.get('batch_losses'):
            plt.figure(figsize=(10, 3))
            plt.plot(eps, per_epoch_mean(result['batch_losses']), linewidth=1.5)
            plt.xlabel('Epoch'); plt.ylabel('Mean cross-entropy loss')
            plt.title(f'[{args.run}] {result["name"]}: train loss per epoch')
            plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(out_dir / f'plot_loss_{key_safe}.png', dpi=100); plt.close()

        # Accuracy per epoch: train (augmented), train (clean), test (raw), test (EMA)
        plt.figure(figsize=(10, 4.5))
        plt.plot(eps, [a*100 for a in result['train_accs']],
                 label='train (with augmentation)', linewidth=1.5)
        plt.plot(eps, [a*100 for a in result['test_accs']],
                 label='test (raw)', linewidth=1.5)
        plt.plot(eps, [a*100 for a in result['test_accs_ema']],
                 label='test (EMA)', linestyle='--', linewidth=1.5)
        # Clean train accuracy markers (sampled every eval_clean_every epochs)
        ce_eps = result.get('clean_eval_epochs', [])
        ce_acc = result.get('train_accs_clean', [])
        if ce_eps and ce_acc:
            plt.plot(ce_eps, [a*100 for a in ce_acc], 'o-',
                     color='red', markersize=6, linewidth=1.0, alpha=0.85,
                     label='train (clean, no augmentation)')
        plt.axhline(result['best_acc']*100, color='red', linestyle=':', alpha=0.4,
                    label=f'best test = {result["best_acc"]*100:.2f}%')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
        plt.title(f'[{args.run}] accuracy per epoch  (best {result["best_acc"]*100:.2f}% '
                  f'via {"EMA" if result["best_is_ema"] else "raw"} weights)')
        plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(out_dir / f'plot_acc_{key_safe}.png', dpi=100); plt.close()

        # Gap between train and test accuracy per epoch.
        # gap_aug   = train_aug - test_raw. Negative under strong augmentation
        #             (CutMix lowers the train accuracy), so it does not measure
        #             overfitting. It documents the effect of the augmentation.
        # gap_clean = train_clean - test_raw, the overfitting gap, measured at
        #             the epochs in clean_eval_epochs.
        if result.get('gaps'):
            plt.figure(figsize=(10, 4))
            gap_pct = [g*100 for g in result['gaps']]
            plt.plot(eps, gap_pct, linewidth=1.5,
                     label='gap_aug = train(aug) − test (negative under CutMix)')
            plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
            plt.axhline(5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
            # Real gap from clean-train evaluations
            if ce_eps and ce_acc:
                test_at_ce = [result['test_accs'][e-1] for e in ce_eps]
                real_gap = [(c - t)*100 for c, t in zip(ce_acc, test_at_ce)]
                plt.plot(ce_eps, real_gap, 'o-', color='red', markersize=6, linewidth=1.0,
                         label='gap_clean = train(clean) − test')
            for cp in [50, 100, 200, 300]:
                if cp <= len(gap_pct):
                    plt.axvline(cp, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
            plt.xlabel('Epoch'); plt.ylabel('train − test accuracy gap (%)')
            plt.title(f'[{args.run}] train-test accuracy gap per epoch')
            plt.legend(loc='best', fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(out_dir / f'plot_gap_{key_safe}.png', dpi=100); plt.close()

        # Gradient norm per epoch: mean and max
        if result.get('grad_norms'):
            gmean = per_epoch_mean(result['grad_norms'])
            gmax  = per_epoch_max(result['grad_norms'])
            plt.figure(figsize=(10, 3.5))
            plt.plot(eps, gmean, label='mean |grad|', linewidth=1.5)
            plt.plot(eps, gmax,  label='max |grad|',  linewidth=1.0, alpha=0.6)
            plt.axhline(5.0, color='red', linestyle=':', alpha=0.5, label='clip = 5.0')
            plt.xlabel('Epoch'); plt.ylabel('|gradient| L2 norm')
            plt.title(f'[{args.run}] gradient norm per epoch')
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(out_dir / f'plot_grad_{key_safe}.png', dpi=100); plt.close()

        print(f'[{args.run}] wrote diagnostic plots: plot_*_{key_safe}.png')
    except Exception as e:
        print(f'[{args.run}] plotting failed (non-fatal): {e}')

    print(f'[{args.run}] saved {ckpt_path}, {stats_path}')
    print(f'[{args.run}] best_acc = {result["best_acc"]*100:.2f}% '
          f'({"EMA" if result["best_is_ema"] else "raw"}) '
          f'@ epoch {result.get("best_epoch", "?")}')


if __name__ == '__main__':
    main()
