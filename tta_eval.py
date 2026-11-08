"""Test-time augmentation (TTA) evaluation of one trained checkpoint on the CIFAR-10 test set.

Views: plain (1), horizontal flip (2, as evaluate_tta in the library), 5 crops of a
2-pixel padded image without flip, 10 crops (corners and centre, each also flipped)
and 5, 10 or 20 random crops with random flips. Multi-view results are given for
averaged logits and for averaged softmax outputs. The script also counts the test
images that flip averaging fixes and the ones it breaks.

Crops are taken from a reflection-padded image (32x32 becomes 40x40 with --pad 4),
not by resizing, to avoid interpolation artefacts.

Usage, with a checkpoint downloaded from Hugging Face or written by train.py:
    python tta_eval.py --ckpt ft_I2_ckpt.pt
    python tta_eval.py --ckpt runs/I2_base/ckpt_run_I2.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

import sknet_lib as L


@torch.no_grad()
def evaluate_plain(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def evaluate_flip_tta(model, loader, device):
    """2 views: original + horizontal flip. Returns (logit_acc, softmax_acc)."""
    model.eval()
    correct_l, correct_s, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            l1 = model(x).float()
            l2 = model(torch.flip(x, dims=[3])).float()
        logits_avg = l1 + l2
        probs_avg = F.softmax(l1, dim=1) + F.softmax(l2, dim=1)
        correct_l += (logits_avg.argmax(dim=1) == y).sum().item()
        correct_s += (probs_avg.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct_l / total, correct_s / total


@torch.no_grad()
def evaluate_10crop_tta(model, loader, device, pad: int = 4):
    """10 views: 5 crops × 2 (flip). Returns (logit_acc, softmax_acc)."""
    model.eval()
    crop_size = 32
    correct_l, correct_s, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        H_pad = x_pad.shape[2]
        offs = [
            (0, 0),
            (0, H_pad - crop_size),
            (H_pad - crop_size, 0),
            (H_pad - crop_size, H_pad - crop_size),
            ((H_pad - crop_size) // 2, (H_pad - crop_size) // 2),
        ]
        all_logits, all_probs = [], []
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            for h, w in offs:
                crop = x_pad[:, :, h:h + crop_size, w:w + crop_size]
                for view in (crop, torch.flip(crop, dims=[3])):
                    l = model(view).float()
                    all_logits.append(l)
                    all_probs.append(F.softmax(l, dim=1))
        logits_avg = torch.stack(all_logits, dim=0).mean(dim=0)
        probs_avg = torch.stack(all_probs, dim=0).mean(dim=0)
        correct_l += (logits_avg.argmax(dim=1) == y).sum().item()
        correct_s += (probs_avg.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct_l / total, correct_s / total


@torch.no_grad()
def evaluate_aug_tta(model, loader, device, n_views: int = 10, pad: int = 4):
    """N random crops + horizontal flips. Returns (logit_acc, softmax_acc)."""
    model.eval()
    crop_size = 32
    correct_l, correct_s, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        B = x.shape[0]
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        sum_logits = torch.zeros(B, 10, device=device)
        sum_probs = torch.zeros(B, 10, device=device)
        for _ in range(n_views):
            h_off = torch.randint(0, 2 * pad + 1, (1,)).item()
            w_off = torch.randint(0, 2 * pad + 1, (1,)).item()
            x_aug = x_pad[:, :, h_off:h_off + crop_size, w_off:w_off + crop_size]
            if torch.rand(1).item() < 0.5:
                x_aug = torch.flip(x_aug, dims=[3])
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                l = model(x_aug).float()
            sum_logits += l
            sum_probs += F.softmax(l, dim=1)
        correct_l += ((sum_logits / n_views).argmax(dim=1) == y).sum().item()
        correct_s += ((sum_probs / n_views).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct_l / total, correct_s / total


@torch.no_grad()
def fixed_broken_analysis(model, loader, device):
    """Compare plain vs flip TTA per-image. Counts:
       fixed = wrong without flip averaging, correct with it
       broken = correct without, wrong with
       net = fixed - broken
    """
    model.eval()
    plain_p, flip_logit_p, flip_softmax_p, labels = [], [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            l1 = model(x).float()
            l2 = model(torch.flip(x, dims=[3])).float()
        plain_p.append(l1.argmax(dim=1).cpu())
        flip_logit_p.append((l1 + l2).argmax(dim=1).cpu())
        flip_softmax_p.append((F.softmax(l1, dim=1) + F.softmax(l2, dim=1)).argmax(dim=1).cpu())
        labels.append(y.cpu())
    plain_p = torch.cat(plain_p)
    flip_logit_p = torch.cat(flip_logit_p)
    flip_softmax_p = torch.cat(flip_softmax_p)
    labels = torch.cat(labels)
    n = len(labels)
    for name, flip_p in [('flip TTA [logit]', flip_logit_p), ('flip TTA [softmax]', flip_softmax_p)]:
        fixed = ((plain_p != labels) & (flip_p == labels)).sum().item()
        broken = ((plain_p == labels) & (flip_p != labels)).sum().item()
        both_wrong = ((plain_p != labels) & (flip_p != labels)).sum().item()
        print(f'  {name}:')
        print(f'    fixed:  {fixed:4d}  ({fixed/n*100:+.2f}%)')
        print(f'    broken: {broken:4d}  ({broken/n*100:-.2f}%)')
        print(f'    net:    {fixed-broken:+4d}  ({(fixed-broken)/n*100:+.2f}%)')
        print(f'    both_wrong: {both_wrong}')


@torch.no_grad()
def evaluate_5crop_tta(model, loader, device, pad: int = 2):
    """5 views: 4 corners and the centre, without flip. Returns (logit_acc, softmax_acc)."""
    model.eval()
    crop_size = 32
    correct_l, correct_s, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        H_pad = x_pad.shape[2]
        offs = [
            (0, 0),
            (0, H_pad - crop_size),
            (H_pad - crop_size, 0),
            (H_pad - crop_size, H_pad - crop_size),
            ((H_pad - crop_size) // 2, (H_pad - crop_size) // 2),
        ]
        all_logits, all_probs = [], []
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            for h, w in offs:
                crop = x_pad[:, :, h:h + crop_size, w:w + crop_size]
                l = model(crop).float()
                all_logits.append(l)
                all_probs.append(F.softmax(l, dim=1))
        logits_avg = torch.stack(all_logits, dim=0).mean(dim=0)
        probs_avg = torch.stack(all_probs, dim=0).mean(dim=0)
        correct_l += (logits_avg.argmax(dim=1) == y).sum().item()
        correct_s += (probs_avg.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct_l / total, correct_s / total


def main():
    p = argparse.ArgumentParser(description='Evaluate one checkpoint with several kinds of test-time augmentation.')
    p.add_argument('--ckpt', required=True, type=str, help='Checkpoint file (ckpt_run_*.pt from train.py or *_ckpt.pt from Hugging Face)')
    p.add_argument('--data-root', type=str, default='./data')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--pad', type=int, default=4,
                   help='Reflection padding in pixels for the 10-crop and random-crop views (4 gives 40x40).')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ck = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    cfg = ck['cfg']
    state_dict = ck['state_dict']
    best_acc_saved = ck.get('best_acc', None)
    best_is_ema = ck.get('best_is_ema', None)

    print(f'\n=== Loaded {ckpt_path} ===')
    print(f'  config name : {cfg.name}')
    print(f'  channels    : {cfg.channels}')
    print(f'  blocks/stage: {cfg.blocks_per_stage}')
    print(f'  branches    : {cfg.branches}')
    print(f'  use_intra   : {cfg.use_intra_trans}')
    # Fields that older configurations may lack:
    print(f'  stoch_depth : {getattr(cfg, "stoch_depth_mode", "n/a")} '
          f'@ max_rate={getattr(cfg, "stoch_depth_max_rate", 0.0)}')
    print(f'  shake_shake : {getattr(cfg, "shake_shake", False)}')
    print(f'  best_acc (from training): {best_acc_saved*100:.2f}% '
          f'({"EMA" if best_is_ema else "raw"})' if best_acc_saved else '  best_acc: n/a')

    # Build the model and load the weights
    model = L.SKNet(cfg).to(device)
    model.load_state_dict(state_dict)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  params      : {n_params/1e6:.2f}M')

    # Build test loader (no aug, normalise only)
    test_tf = L.test_transform
    test_ds = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_tf)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0))
    print(f'  test set    : {len(test_ds)} examples, {len(test_loader)} batches\n')

    # Reproducibility for random aug TTA
    torch.manual_seed(42)

    # Run all eval modes (both logit-mean and softmax-mean)
    print('=== Evaluating  (logit-avg / softmax-avg) ===')
    results = {}

    t0 = time.time()
    acc = evaluate_plain(model, test_loader, device)
    results['plain (1 view)'] = ((acc, acc), time.time() - t0)
    print(f'  plain          (1  view) : {acc*100:6.2f}%  ({time.time()-t0:.1f}s)')

    plain_acc = acc

    def _print_pair(label, accs, dt):
        l, s = accs
        print(f'  {label:24s}: logit={l*100:6.2f}%  softmax={s*100:6.2f}%  '
              f'(Δlogit={(l-plain_acc)*100:+.2f}%  Δsoftmax={(s-plain_acc)*100:+.2f}%)  ({dt:.1f}s)')

    t0 = time.time(); accs = evaluate_flip_tta(model, test_loader, device)
    results['flip TTA (2 views)'] = (accs, time.time() - t0); _print_pair('flip TTA (2 views)', accs, time.time()-t0)

    t0 = time.time(); accs = evaluate_5crop_tta(model, test_loader, device, pad=2)
    results['5-crop pad=2 (5 views)'] = (accs, time.time() - t0); _print_pair('5-crop pad=2 (5 views)', accs, time.time()-t0)

    t0 = time.time(); accs = evaluate_10crop_tta(model, test_loader, device, pad=args.pad)
    results[f'10-crop pad={args.pad} (10 views)'] = (accs, time.time() - t0); _print_pair(f'10-crop pad={args.pad} (10 views)', accs, time.time()-t0)

    for n in [5, 10, 20]:
        t0 = time.time(); accs = evaluate_aug_tta(model, test_loader, device, n_views=n, pad=args.pad)
        results[f'random-aug TTA ({n} views)'] = (accs, time.time() - t0); _print_pair(f'random-aug TTA ({n} views)', accs, time.time()-t0)

    print()
    print('=== Fixed/Broken analysis (plain vs flip TTA) ===')
    fixed_broken_analysis(model, test_loader, device)

    print()
    print('=== Summary ===')
    # Flatten: each entry has (logit_acc, softmax_acc), pick best across both
    flat = []
    for name, ((l, s), _) in results.items():
        flat.append((f'{name} [logit]', l))
        flat.append((f'{name} [softmax]', s))
    best_name, best_acc = max(flat, key=lambda kv: kv[1])
    print(f'  Best: {best_acc*100:.2f}%  ({best_name})')
    print(f'  vs plain: {(best_acc - plain_acc)*100:+.2f}%')


if __name__ == '__main__':
    main()
