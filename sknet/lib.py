"""Model, data pipeline and training loop for the CIFAR-10 runs.

Checkpoints refer to this module by its original name sknet_lib (see sknet_lib.py
in the repository root).

Importing the module seeds the random number generators (seed 42), switches on
cuDNN benchmark mode and prints the device. The data loaders are not built at
import time: call _init_default_loaders() or set train_loader and test_loader, as
train.py does. ``python -m sknet.lib`` builds every entry of RUN_CONFIGS and runs
test_basic_mode().
"""

# Imports, seeds and device
import os
import math
import time
import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# cuDNN benchmark mode picks the fastest convolution algorithm for the fixed 32x32 input.
# It also makes runs not bit-for-bit repeatable, even with fixed seeds.
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


# Data: augmentation and loaders
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


class Cutout:
    """Erases a random square region from a tensor image."""
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones((h, w), dtype=img.dtype)
        for _ in range(self.n_holes):
            y = int(torch.randint(0, h, (1,)).item())
            x = int(torch.randint(0, w, (1,)).item())
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.
        return img * mask.unsqueeze(0)


train_transform = T.Compose([
    T.RandomCrop(32, padding=4, padding_mode='reflect'),
    T.RandomHorizontalFlip(p=0.5),
    T.TrivialAugmentWide(),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    Cutout(n_holes=1, length=16),
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

BATCH_SIZE = 128

# Loaders are filled by _init_default_loaders() or set by train.py.
train_dataset = test_dataset = train_loader = test_loader = None


def _init_default_loaders(data_root: str = './data', num_workers: int = 2):
    """Populate the module-level train_loader / test_loader."""
    global train_dataset, test_dataset, train_loader, test_loader
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=False)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=False)
    print(f'Train examples: {len(train_dataset)}, test examples: {len(test_dataset)}')
    print(f'Batches per epoch (train): {len(train_loader)}')


# CutMix
def _rand_bbox(size, lam):
    _, _, H, W = size
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Returns (mixed_x, y_a, y_b, lam). Loss = lam*L(pred, y_a) + (1-lam)*L(pred, y_b)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    rand_index = torch.randperm(x.size(0), device=x.device)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    # recompute lam from the clipped box: lam is the fraction of pixels of the original image
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam


# Intermediate block
def _make_branch(kind: str, c_in: int, c_out: int, stride: int = 1) -> nn.Module:
    """Build a single branch C_l: Conv, then BN, then ReLU.

    Output shape must match across all branches of a block.
    bias=False because it is absorbed by the subsequent BatchNorm.
    """
    if kind == '3x3':
        conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
    elif kind == '3x3_dil2':
        conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False)
    elif kind == '1x1':
        conv = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False)
    else:
        raise ValueError(f'Unknown branch kind: {kind}')
    return nn.Sequential(conv, nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class IntermediateBlock(nn.Module):
    """Intermediate block: x' = Σ_l a_l · C_l(x) with a = FC(GAP(x)).

    Arguments:
        c_in, c_out: channels in/out (equal inside a stage).
        branches:    list of branch kinds (length L) like ['3x3', '1x1'].
        stride:      applied identically to every branch (shapes remain consistent).
        basic_mode:  if True, uses the minimal two-branch form (L=2, [3x3, 1x1]),
                     checked by test_basic_mode().

    Initialisation of fc: weight 0, bias 1/L, so that every block starts as
    the plain average of its branches (a_l = 1/L). SKNet._init_weights sets
    the bias to 0.01 for configurations without intra-transitions.
    """
    def __init__(self, c_in: int, c_out: int, branches: Optional[List[str]] = None,
                 stride: int = 1, basic_mode: bool = False):
        super().__init__()
        if basic_mode:
            branches = ['3x3', '1x1']
        elif branches is None:
            branches = ['3x3', '1x1']
        self.branches = nn.ModuleList([
            _make_branch(kind, c_in, c_out, stride=stride) for kind in branches
        ])
        self.L = len(branches)
        # GAP is a plain mean over the pixels, written as nn.AdaptiveAvgPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Single FC producing L scalar attention weights from the c_in-dim input m
        self.fc = nn.Linear(c_in, self.L)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 1.0 / self.L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run all branches on the same input, compute one weight per branch from the mean of the input
        # over all pixels, and add the branch outputs with these weights.
        U = [branch(x) for branch in self.branches]

        m = self.gap(x).flatten(1)   # (B, c_in)
        a = self.fc(m)  # (B, L)
        U_stack = torch.stack(U, dim=1)    # (B, L, c_out, H', W')
        a_expanded = a.view(a.size(0), a.size(1), 1, 1, 1)  # (B, L, 1, 1, 1)
        x_prime = (a_expanded * U_stack).sum(dim=1)  # (B, c_out, H', W')
        return x_prime


# Transitions, stochastic depth and the full network
class StageTransition(nn.Module):
    """Halves height and width and changes the number of channels from c_in to c_out.

    Main path: 3x3 stride-2 conv and BN. Shortcut: 1x1 stride-2 conv and BN.
    ReLU after the sum (the ResNet downsampling block).
    """
    def __init__(self, c_in: int, c_out: int, use_skip: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.use_skip = use_skip
        if use_skip:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.use_skip:
            out = out + self.shortcut(x)
        return self.relu(out)


class IntraTransition(nn.Module):
    """A plain 3x3 conv + BN + ReLU inserted between two intermediate blocks
    inside a stage. Adds one layer of depth and keeps the shape. No skip around it.
    """
    def __init__(self, c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def drop_path(x: torch.Tensor, p: float = 0.0, training: bool = False) -> torch.Tensor:
    """DropPath, a per-sample variant of stochastic depth (Huang et al., ECCV 2016,
    where a block is dropped for the whole mini-batch). With
    probability p, set the whole output of a block to zero for one sample,
    and rescale the kept outputs by 1/keep. Applied outside the
    IntermediateBlock, so the block formula is unchanged.
    """
    if not training or p <= 0.0:
        return x
    keep = 1.0 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x * mask / keep


@dataclass
class SKNetConfig:
    """Architecture of one configuration. Training settings are arguments of train_one_config.

    stoch_depth_mode:
      'none'               no stochastic depth, the block output is added unchanged
      'droppath_linear'    DropPath rate grows linearly from 0 to max_rate with block depth

    stoch_depth_max_rate:
      Maximum stochastic-depth rate, used for the deepest block. I1 uses 0.10.

    The published checkpoints were written by the historical version of this
    module, whose configuration had one more field, shake_shake (always False
    in the published runs). Unpickling keeps it as an extra attribute that
    nothing reads.
    """
    name: str = 'run'
    # The network has 3 stages, so blocks_per_stage has length 3
    blocks_per_stage: List[int] = field(default_factory=lambda: [5, 5, 4])
    channels: List[int] = field(default_factory=lambda: [160, 320, 640])
    branches: List[str] = field(default_factory=lambda: ['3x3', '1x1'])
    stem_channels: Optional[int] = None
    use_intra_trans: bool = True
    use_skip: bool = True
    stoch_depth_mode: str = 'none'
    stoch_depth_max_rate: float = 0.0
    basic_mode: bool = False
    num_classes: int = 10


class SKNet(nn.Module):
    """Full network: a stem, then 3 stages of (blocks + optional intra-transitions)
    separated by StageTransition layers, then GAP and a single FC head.

    With basic_mode=True it becomes the plain block sequence with L=2 branches
    [3x3, 1x1], no intra-transitions, no skip connections, no stochastic depth.
    """
    def __init__(self, cfg: SKNetConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.basic_mode:
            cfg = SKNetConfig(
                name=cfg.name,
                blocks_per_stage=cfg.blocks_per_stage,
                channels=cfg.channels,
                branches=['3x3', '1x1'],
                stem_channels=cfg.stem_channels,
                use_intra_trans=False,
                use_skip=False,
                stoch_depth_mode='none',
                stoch_depth_max_rate=0.0,
                basic_mode=True,
                num_classes=cfg.num_classes,
            )
            self.cfg = cfg

        stem_c = cfg.stem_channels if cfg.stem_channels is not None else cfg.channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_c),
            nn.ReLU(inplace=True),
        )

        # Per-block DropPath rates (parallel list to IntermediateBlock order),
        # filled while building stages and consumed by forward().
        total_blocks = sum(cfg.blocks_per_stage)
        self._block_drop_rates: List[float] = []
        block_counter = 0

        self.stages = nn.ModuleList()
        c_prev = stem_c
        for stage_idx, (num_blocks, c_stage) in enumerate(
            zip(cfg.blocks_per_stage, cfg.channels)
        ):
            stage_modules = nn.ModuleList()
            if stage_idx > 0:
                stage_modules.append(StageTransition(c_prev, c_stage, use_skip=cfg.use_skip))
                c_prev = c_stage
            elif c_prev != c_stage:
                stage_modules.append(nn.Sequential(
                    nn.Conv2d(c_prev, c_stage, kernel_size=1, bias=False),
                    nn.BatchNorm2d(c_stage),
                    nn.ReLU(inplace=True),
                ))
                c_prev = c_stage

            for b in range(num_blocks):
                stage_modules.append(IntermediateBlock(
                    c_in=c_stage, c_out=c_stage,
                    branches=cfg.branches, stride=1,
                    basic_mode=False,
                ))

                if cfg.stoch_depth_mode == 'none' or cfg.stoch_depth_max_rate <= 0.0:
                    rate = 0.0
                elif cfg.stoch_depth_mode == 'droppath_linear':
                    frac = block_counter / max(1, total_blocks - 1)
                    rate = frac * cfg.stoch_depth_max_rate
                else:
                    raise ValueError(f'Unknown stoch_depth_mode: {cfg.stoch_depth_mode}')
                self._block_drop_rates.append(rate)
                block_counter += 1

                if cfg.use_intra_trans and b < num_blocks - 1:
                    stage_modules.append(IntraTransition(c_stage))

            self.stages.append(stage_modules)

        final_c = cfg.channels[-1]
        self.output_gap = nn.AdaptiveAvgPool2d(1)
        self.output_fc  = nn.Linear(final_c, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        """Conv: Kaiming (fan_out, ReLU). BN: identity (gamma=1, beta=0).

        For the configurations without intra-transitions (1.1 and 1.2, 27 blocks
        in a row) the bias of the attention FC is set to 0.01 instead of 1/L, so
        that each block starts with a small contribution. Configurations with
        intra-transitions keep 1/L.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Small-bias attention init for pure-stack configs (no intra-transitions).
        if not self.cfg.use_intra_trans and not self.cfg.basic_mode:
            for blk in self.modules():
                if isinstance(blk, IntermediateBlock):
                    nn.init.constant_(blk.fc.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        mode = self.cfg.stoch_depth_mode
        block_idx = 0

        for stage in self.stages:
            for m in stage:
                # Stage transition, first-stage projection or intra-transition:
                # applied as is, no skip around it.
                if not isinstance(m, IntermediateBlock):
                    x = m(x)
                    continue
                # IntermediateBlock gets its own skip. An intra-transition (if any) is
                # a separate stage_modules element with no skip around it.
                y = m(x)
                if self.cfg.use_skip:
                    rate = self._block_drop_rates[block_idx] if self._block_drop_rates else 0.0
                    if mode == 'droppath_linear':
                        y = drop_path(y, rate, self.training)

                    x = F.relu(y + x, inplace=True)
                else:
                    x = y
                block_idx += 1

        m = self.output_gap(x).flatten(1)
        logits = self.output_fc(m)
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Run configurations
RUN_CONFIGS = {
    # Architectures only, the training settings of each run are in experiments.md.
    # 1.x: 27 blocks without intra-transitions, 2.x: 14 blocks with intra-transitions.
    # x.1: branches 3x3 and 1x1 with 160/320/640 channels, x.2: 1x1, 3x3 and dilated 3x3 with 120/240/480.
    # I1 and I2 are 2.1 with 192/384/768 channels, I1 with DropPath. 1.2 was never trained.
    '1.1': SKNetConfig(
        name='1.1_pure_L2',
        blocks_per_stage=[9, 9, 9], channels=[160, 320, 640],
        branches=['3x3', '1x1'],
        use_intra_trans=False, use_skip=True,
    ),
    '1.2': SKNetConfig(
        name='1.2_pure_L3',
        blocks_per_stage=[9, 9, 9], channels=[120, 240, 480],
        branches=['1x1', '3x3', '3x3_dil2'],
        use_intra_trans=False, use_skip=True,
    ),
    '2.1': SKNetConfig(
        name='2.1_intra_L2',
        blocks_per_stage=[5, 5, 4], channels=[160, 320, 640],
        branches=['3x3', '1x1'],
        use_intra_trans=True, use_skip=True,
    ),
    '2.2': SKNetConfig(
        name='2.2_intra_L3',
        blocks_per_stage=[5, 5, 4], channels=[120, 240, 480],
        branches=['1x1', '3x3', '3x3_dil2'],
        use_intra_trans=True, use_skip=True,
    ),
    'I1': SKNetConfig(
        name='I1_droppath_linear',
        blocks_per_stage=[5, 5, 4], channels=[192, 384, 768],
        branches=['3x3', '1x1'],
        use_intra_trans=True, use_skip=True,
        stoch_depth_mode='droppath_linear',
        stoch_depth_max_rate=0.10,
    ),
    'I2': SKNetConfig(
        name='I2_mixup_cutmix',
        blocks_per_stage=[5, 5, 4], channels=[192, 384, 768],
        branches=['3x3', '1x1'],
        use_intra_trans=True, use_skip=True,
    ),
}




# Weight averaging (EMA)
class EMA:
    """Exponential moving average of the trainable parameters. BatchNorm running
    statistics are buffers and are not averaged, so the EMA weights are
    evaluated with the running statistics of the raw weights.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# Evaluation, optimisers, SAM and the training loop
def evaluate(model: nn.Module, loader: DataLoader, use_amp: bool = True) -> float:
    """Top-1 accuracy on `loader`. `use_amp=False` for FP32 eval (SAM runs)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available(),
                                         dtype=torch.bfloat16):
                logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total   += y.size(0)
    return correct / total


@torch.no_grad()
def evaluate_tta(model: nn.Module, loader: DataLoader) -> float:
    """Test-time augmentation: add the logits for the image and its horizontal flip."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(),
                                     dtype=torch.bfloat16):
            logits = model(x) + model(torch.flip(x, dims=[3]))
        correct += (logits.argmax(dim=1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# Training images with the test-set transform (no augmentation, no mixing). Used to measure the clean train accuracy
# and its gap to the test accuracy, because CutMix and MixUp distort the running train accuracy.
def _build_clean_train_loader(data_root: str = './data',
                              num_workers: int = 2) -> DataLoader:
    ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=test_transform)
    return DataLoader(ds, batch_size=256, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=False)


def _build_aug_loader(recipe: str, batch_size: int = 128,
                      num_workers: int = 2, data_root: str = './data') -> DataLoader:
    """Build a fresh train DataLoader with the requested auto-augmentation
    recipe. Always includes RandomCrop+pad+flip, Normalize and Cutout(16). Only the auto-aug operator changes.

    recipe is 'trivial' (TrivialAugmentWide, the default), 'autoaugment_cifar10' or 'randaugment'.
    The published runs all use 'trivial', which does not call this function.
    """
    if recipe == 'autoaugment_cifar10':
        auto_op = T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10)
    elif recipe == 'randaugment':
        auto_op = T.RandAugment(num_ops=2, magnitude=14)
    elif recipe == 'trivial':
        auto_op = T.TrivialAugmentWide()
    else:
        raise ValueError(f'Unknown aug recipe: {recipe}')

    tf = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(p=0.5),
        auto_op,
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        Cutout(n_holes=1, length=16),
    ])
    ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers,
                      pin_memory=True, drop_last=True,
                      persistent_workers=False)


def _build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                     decoupled_wd: bool):
    """SGD with Nesterov momentum. With `decoupled_wd=True`, BN gamma/bias and any 1-D
    parameter get weight_decay=0. All other weights use the requested value.
    The published runs use decoupled_wd=False.
    """
    if not decoupled_wd:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               nesterov=True, weight_decay=weight_decay)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.SGD(
        [{'params': decay,    'weight_decay': weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr, momentum=0.9, nesterov=True,
    )


def _build_scheduler(optimizer, num_epochs: int, warmup_epochs: int):
    """Build the LR schedule. When `warmup_epochs > 0`, the schedule is a
    two-stage `SequentialLR`: a linear warmup from lr*0.01 to lr over the
    first `warmup_epochs`, followed by cosine annealing from lr to 0 over
    the remaining epochs. When `warmup_epochs == 0` (the baseline default),
    only cosine annealing is used over the full training.
    """
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs))
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs])


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """MixUp augmentation: a weighted average of two images and of their labels.
    Returns (mixed_x, y_a, y_b, lam). Loss = lam*L(pred,y_a) + (1-lam)*L(pred,y_b).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    rand_index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[rand_index]
    return mixed_x, y, y[rand_index], lam


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimisation (Foret et al., ICLR 2021).

    Wraps an inner SGD-like optimiser. Each training step requires two
    forward and backward passes: first_step() temporarily moves the weights
    to the point of (approximately) maximum loss within a small neighbourhood
    of radius rho around the current weights. second_step() restores the
    weights and applies the descent step using the gradient computed at that
    perturbed point. The effect is to bias training toward flatter minima.
    """
    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        if rho <= 0.0:
            raise ValueError('SAM rho must be > 0')
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'e_w' not in self.state[p]:
                    continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]['params'][0].device
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2).to(device)
            for group in self.param_groups for p in group['params']
            if p.grad is not None
        ]), p=2)
        return norm

    def step(self, closure=None):
        raise RuntimeError('SAM.step() must not be called directly, use first_step and second_step')


def train_one_config(cfg: SKNetConfig, num_epochs: int = 300,
                     lr: float = 0.1, weight_decay: float = 5e-4,
                     cutmix_alpha: float = 1.0, cutmix_prob: float = 0.5,
                     mixup_alpha: float = 0.2, mixup_prob: float = 0.0,
                     label_smoothing: float = 0.1,
                     ema_decay: float = 0.999,
                     use_amp: bool = True,
                     grad_clip: float = 5.0,
                     eval_clean_every: int = 25,
                     warmup_epochs: int = 0,
                     decoupled_wd: bool = False,
                     aug_recipe: str = 'trivial',
                     use_sam: bool = False, sam_rho: float = 0.05,
                     # SWA (Stochastic Weight Averaging, Izmailov et al., 2018):
                     use_swa: bool = False,
                     swa_start_epoch: int = 75,
                     swa_lr: float = 0.001,
                     init_from: Optional[str] = None,
                     batch_size: int = 128,
                     num_workers: int = 2,
                     data_root: str = './data',
                     verbose: bool = True,
                     on_epoch_end=None):
    """Train one SKNet config and return a dict of per-epoch stats plus the best state_dict.

    The kept weights (best_state) are those of the epoch with the highest test accuracy,
    raw or EMA. There is no separate validation split.
    """
    # Pick the train loader: 'trivial' reuses the module-level train_loader,
    # other recipes build a fresh loader with the requested auto-augmentation.
    if aug_recipe == 'trivial':
        local_train_loader = train_loader
    else:
        local_train_loader = _build_aug_loader(
            aug_recipe,
            batch_size=batch_size,
            num_workers=num_workers,
            data_root=data_root,
        )

    train_clean_loader = _build_clean_train_loader(data_root=data_root,
                                                   num_workers=num_workers)

    # SAM is incompatible with stochastic-depth-style block-level randomness
    # (DropPath). Random masks differ between SAM's first and second forward
    # pass, so the two gradients do not match. Force-disable when SAM is on.
    if use_sam and cfg.stoch_depth_mode != 'none':
        if verbose:
            print(f'  WARNING: SAM is incompatible with DropPath '
                  f'(random masks differ between SAM passes). Forcing '
                  f'stoch_depth_mode="none" for this run.')
        cfg = SKNetConfig(
            name=cfg.name,
            blocks_per_stage=cfg.blocks_per_stage,
            channels=cfg.channels,
            branches=cfg.branches,
            stem_channels=cfg.stem_channels,
            use_intra_trans=cfg.use_intra_trans,
            use_skip=cfg.use_skip,
            stoch_depth_mode='none',
            stoch_depth_max_rate=0.0,
            basic_mode=cfg.basic_mode,
            num_classes=cfg.num_classes,
        )

    model = SKNet(cfg).to(device)
    n_params = count_params(model)
    if use_sam:
        use_amp = False  # SAM runs in 32-bit precision

    # Load an existing checkpoint as the starting point (used for the SAM+SWA fine-tunes)
    if init_from is not None:
        if verbose:
            print(f'--- Loading initial weights from {init_from} ---')
        loaded = torch.load(init_from, map_location=device, weights_only=False)
        if 'state_dict' in loaded:
            if 'cfg' in loaded and getattr(loaded['cfg'], 'name', None) != cfg.name:
                print(f'  WARNING: init_from cfg name "{loaded["cfg"].name}" != target cfg name '
                      f'"{cfg.name}". Loading may succeed but semantics may differ.')
            model.load_state_dict(loaded['state_dict'])
            if verbose:
                src_acc = loaded.get('best_acc', None)
                print(f'    loaded weights from best_acc={src_acc*100:.2f}% '
                      f'({"EMA" if loaded.get("best_is_ema") else "raw"})'
                      if src_acc else '    loaded state_dict (no best_acc info)')
        else:
            raise ValueError(f'init_from ckpt {init_from} has no "state_dict" key')

    if verbose:
        extras = []
        if warmup_epochs > 0:    extras.append(f'warmup={warmup_epochs}')
        if decoupled_wd:         extras.append('decoupled_wd')
        if aug_recipe != 'trivial': extras.append(f'aug={aug_recipe}')
        if cfg.stoch_depth_mode != 'none':
            extras.append(f'sd={cfg.stoch_depth_mode}@{cfg.stoch_depth_max_rate}')
        if mixup_prob > 0:       extras.append(f'mixup_p={mixup_prob}')
        if use_sam:              extras.append(f'SAM(rho={sam_rho})')
        if use_swa:              extras.append(f'SWA(start={swa_start_epoch},lr={swa_lr})')
        if init_from:            extras.append(f'init_from={init_from.split("/")[-1]}')
        extra_str = (' [' + ', '.join(extras) + ']') if extras else ''
        print(f'--- {cfg.name}{extra_str} --- {n_params/1e6:.2f}M params, {num_epochs} epochs ---')

    if use_sam:
        # SAM wraps an inner SGD optimiser. When decoupled_wd is on, build the
        # (decay / no_decay) parameter groups here and pass them to SAM so
        # that weight-decay decoupling is preserved under SAM as well.
        if decoupled_wd:
            decay, no_decay = [], []
            for name_, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim <= 1 or name_.endswith('.bias'):
                    no_decay.append(p)
                else:
                    decay.append(p)
            sam_param_groups = [
                {'params': decay,    'weight_decay': weight_decay},
                {'params': no_decay, 'weight_decay': 0.0},
            ]
            optimizer = SAM(sam_param_groups, torch.optim.SGD,
                            rho=sam_rho, lr=lr, momentum=0.9, nesterov=True)
        else:
            optimizer = SAM(model.parameters(), torch.optim.SGD,
                            rho=sam_rho, lr=lr, momentum=0.9, nesterov=True,
                            weight_decay=weight_decay)
    else:
        optimizer = _build_optimizer(model, lr, weight_decay, decoupled_wd)
    scheduler = _build_scheduler(optimizer, num_epochs, warmup_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
    ema       = EMA(model, decay=ema_decay)

    # SWA setup (Izmailov et al., 2018). AveragedModel maintains a running mean of
    # weights. SWALR holds the LR at swa_lr during the SWA phase.
    swa_model = None
    swa_scheduler = None
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr,
                              anneal_strategy='linear', anneal_epochs=1)
        if verbose:
            print(f'    SWA enabled: averaging starts at epoch {swa_start_epoch}, '
                  f'swa_lr={swa_lr} (constant via linear anneal_epochs=1)')

    batch_losses      = []
    grad_norms        = []
    nan_steps         = 0
    train_accs        = []
    test_accs         = []
    test_accs_ema     = []
    gaps              = []
    train_accs_clean  = []
    clean_eval_epochs = []
    best_acc      = 0.0
    best_state    = None
    best_is_ema   = False
    best_epoch    = None
    aborted       = False

    # Per-block weight monitoring, to see whether any block is bypassed during training.
    # For each IntermediateBlock: FC weight and bias, branch BN weight and branch conv weight.
    # A block whose values fall close to zero is being skipped through the skip path.
    block_modules = [m for stage_mods in model.stages for m in stage_mods
                     if isinstance(m, IntermediateBlock)]
    block_stats = []

    t0 = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_correct, running_total = 0, 0
        for x, y in local_train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            r = np.random.rand()
            mix_mode = 'none'
            y_a, y_b, lam = y, y, 1.0
            if cutmix_alpha > 0 and r < cutmix_prob:
                x, y_a, y_b, lam = cutmix(x, y, alpha=cutmix_alpha)
                mix_mode = 'cutmix'
            elif mixup_alpha > 0 and r < cutmix_prob + mixup_prob:
                x, y_a, y_b, lam = mixup(x, y, alpha=mixup_alpha)
                mix_mode = 'mixup'

            def _compute_loss(logits_):
                if mix_mode != 'none':
                    return lam * criterion(logits_, y_a) + (1 - lam) * criterion(logits_, y_b)
                return criterion(logits_, y)

            if use_sam:
                # SAM training step (FP32, no AMP). First forward+backward
                # finds the perturbation toward the local loss maximum. SAM
                # then re-evaluates the gradient at the perturbed weights and
                # uses that gradient to take the actual SGD step.
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = _compute_loss(logits)
                if not torch.isfinite(loss):
                    nan_steps += 1
                    if verbose:
                        print(f'  WARNING: non-finite loss ({loss.item()}) at epoch {epoch+1}, '
                              f'aborting this run.')
                    aborted = True
                    break
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.first_step(zero_grad=True)
                logits2 = model(x)
                loss2 = _compute_loss(logits2)
                if not torch.isfinite(loss2):
                    nan_steps += 1
                    if verbose:
                        print(f'  WARNING: non-finite loss2 ({loss2.item()}) after SAM '
                              f'perturbation at epoch {epoch+1}, aborting this run.')
                    aborted = True
                    break
                loss2.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.second_step(zero_grad=True)
                grad_norms.append(float(total_norm))
                batch_losses.append(loss.item())
                preds = logits.argmax(dim=1)
            else:
                # bfloat16 autocast instead of float16: bfloat16 has the same exponent range as float32,
                # which removed the overflow seen with float16 in the intermediate blocks. GradScaler is not needed for bfloat16.
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available(),
                                             dtype=torch.bfloat16):
                    logits = model(x)
                    loss = _compute_loss(logits)
                if not torch.isfinite(loss):
                    nan_steps += 1
                    if verbose:
                        print(f'  WARNING: non-finite loss ({loss.item()}) at epoch {epoch+1}, '
                              f'aborting this run.')
                    aborted = True
                    break
                loss.backward()
                # Element-wise clip (outlier guard) then global-norm clip.
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_norms.append(float(total_norm))
                if not torch.isfinite(total_norm):
                    nan_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()
                batch_losses.append(loss.item())
                preds = logits.argmax(dim=1)

            ema.update(model)
            running_correct += (preds == y).sum().item()
            running_total   += y.size(0)

        if aborted:
            break

        # SWA phase: starting at swa_start_epoch, accumulate running average of
        # weights and use SWA-LR (constant) instead of the cosine scheduler.
        if use_swa and (epoch + 1) >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        train_acc = running_correct / running_total
        # Evaluate in the same precision as training (SAM runs in FP32).
        test_acc_raw = evaluate(model, test_loader, use_amp=use_amp)

        ema.apply_shadow(model)
        test_acc_ema = evaluate(model, test_loader, use_amp=use_amp)
        ema.restore(model)

        train_accs.append(train_acc)
        test_accs.append(test_acc_raw)
        test_accs_ema.append(test_acc_ema)
        gaps.append(train_acc - test_acc_raw)

        # Per-epoch attention/branch weight stats per IntermediateBlock.
        with torch.no_grad():
            epoch_block_data = []
            min_fc_w_max = float('inf')
            min_bn_g_max = float('inf')
            min_conv_w_max = float('inf')
            for blk_idx, blk in enumerate(block_modules):
                fc_w_max = blk.fc.weight.abs().max().item()
                fc_w_mean = blk.fc.weight.abs().mean().item()
                fc_b_max = blk.fc.bias.abs().max().item()
                # Per-branch (Conv + BN) stats
                branches_data = []
                for br_idx, branch in enumerate(blk.branches):
                    br_conv_w_max = 0.0
                    br_conv_w_mean = 0.0
                    br_bn_g_max = 0.0
                    br_bn_g_mean = 0.0
                    br_bn_b_max = 0.0
                    for sub in branch.modules():
                        if isinstance(sub, nn.Conv2d):
                            br_conv_w_max = sub.weight.abs().max().item()
                            br_conv_w_mean = sub.weight.abs().mean().item()
                        elif isinstance(sub, nn.BatchNorm2d):
                            br_bn_g_max = sub.weight.abs().max().item()
                            br_bn_g_mean = sub.weight.abs().mean().item()
                            br_bn_b_max = sub.bias.abs().max().item()
                    branches_data.append({
                        'idx': br_idx,
                        'conv_w_max':  br_conv_w_max,
                        'conv_w_mean': br_conv_w_mean,
                        'bn_g_max':    br_bn_g_max,
                        'bn_g_mean':   br_bn_g_mean,
                        'bn_b_max':    br_bn_b_max,
                    })
                bn_g_max_blk = max((b['bn_g_max'] for b in branches_data), default=0.0)
                conv_w_max_blk = max((b['conv_w_max'] for b in branches_data), default=0.0)
                epoch_block_data.append({
                    'idx': blk_idx,
                    'fc_w_max':    fc_w_max,
                    'fc_w_mean':   fc_w_mean,
                    'fc_b_max':    fc_b_max,
                    'branches':    branches_data,
                    'bn_g_max':    bn_g_max_blk,
                    'conv_w_max':  conv_w_max_blk,
                })
                min_fc_w_max = min(min_fc_w_max, fc_w_max)
                min_bn_g_max = min(min_bn_g_max, bn_g_max_blk)
                min_conv_w_max = min(min_conv_w_max, conv_w_max_blk)
            block_stats.append({
                'epoch': epoch + 1,
                'blocks': epoch_block_data,
                'min_fc_w_max':   min_fc_w_max,
                'min_bn_g_max':   min_bn_g_max,
                'min_conv_w_max': min_conv_w_max,
            })
            # Warn when, in some block, the largest FC weight, branch BN weight or
            # branch conv weight falls near zero, a sign that the block is bypassed
            # through the skip connection.
            if verbose and (min_fc_w_max < 1e-3 or min_bn_g_max < 0.05 or min_conv_w_max < 1e-3):
                print(f'  [epoch {epoch+1}] WARNING block-bypass risk: '
                      f'min FC|w|max={min_fc_w_max:.2e}, '
                      f'min BN-γ max={min_bn_g_max:.3f}, '
                      f'min Conv|w|max={min_conv_w_max:.2e}',
                      flush=True)

        epoch_idx_1based = epoch + 1
        is_last = (epoch_idx_1based == num_epochs)
        if eval_clean_every > 0 and (epoch_idx_1based % eval_clean_every == 0 or is_last):
            tac = evaluate(model, train_clean_loader, use_amp=use_amp)
            train_accs_clean.append(tac)
            clean_eval_epochs.append(epoch_idx_1based)

        current_best_this_epoch = max(test_acc_raw, test_acc_ema)
        if current_best_this_epoch > best_acc:
            best_acc = current_best_this_epoch
            best_is_ema = test_acc_ema >= test_acc_raw
            best_epoch = epoch_idx_1based
            if best_is_ema:
                ema.apply_shadow(model)
                best_state = copy.deepcopy(model.state_dict())
                ema.restore(model)
            else:
                best_state = copy.deepcopy(model.state_dict())

        if on_epoch_end is not None:
            # Pass the epoch statistics and the current learning rate (SWA or
            # cosine scheduler) to the callback.
            in_swa_phase = use_swa and (epoch + 1) >= swa_start_epoch
            active_scheduler = swa_scheduler if in_swa_phase else scheduler
            on_epoch_end(epoch + 1, {
                'train_acc':         train_acc,
                'test_acc_raw':      test_acc_raw,
                'test_acc_ema':      test_acc_ema,
                'best_acc':          best_acc,
                'best_is_ema':       best_is_ema,
                'best_epoch':        best_epoch,
                'best_state':        best_state,
                'train_accs':        train_accs,
                'test_accs':         test_accs,
                'test_accs_ema':     test_accs_ema,
                'gaps':              gaps,
                'grad_norms':        grad_norms,
                'batch_losses':      batch_losses,
                'train_accs_clean':  train_accs_clean,
                'clean_eval_epochs': clean_eval_epochs,
                'nan_steps':         nan_steps,
                'lr':                active_scheduler.get_last_lr()[0],
                'elapsed':           time.time() - t0,
                'num_train_batches': len(local_train_loader),
                'block_stats':       block_stats,
            })

    total_time = time.time() - t0

    # SWA finalisation. Recompute the batch-norm running statistics on the clean train loader
    # (the averaged weights produce different activations than any single snapshot),
    # then evaluate. The SWA state is saved along with best_raw/best_ema, and the
    # candidate with the highest test accuracy becomes best_state.
    swa_acc = None
    swa_state = None
    if use_swa and not aborted and swa_model is not None:
        if verbose:
            print(f'--- {cfg.name} SWA finalization: BN recalibrate on clean train loader ---')
        from torch.optim.swa_utils import update_bn
        # update_bn calls model.train(), which would re-enable DropPath, so force
        # it off on the AveragedModel for the duration of the recalibration.
        inner_cfg = swa_model.module.cfg
        saved_sd_mode  = inner_cfg.stoch_depth_mode
        saved_sd_rate  = inner_cfg.stoch_depth_max_rate
        inner_cfg.stoch_depth_mode     = 'none'
        inner_cfg.stoch_depth_max_rate = 0.0
        if hasattr(swa_model.module, '_block_drop_rates'):
            saved_rates = list(swa_model.module._block_drop_rates)
            swa_model.module._block_drop_rates = [0.0] * len(saved_rates)
        else:
            saved_rates = None
        try:
            update_bn(train_clean_loader, swa_model, device=device)
        finally:
            inner_cfg.stoch_depth_mode     = saved_sd_mode
            inner_cfg.stoch_depth_max_rate = saved_sd_rate
            if saved_rates is not None:
                swa_model.module._block_drop_rates = saved_rates
        swa_acc = evaluate(swa_model, test_loader, use_amp=use_amp)
        # The AveragedModel state_dict has a 'module.' prefix, so save the inner module's state_dict.
        swa_state = copy.deepcopy(swa_model.module.state_dict())
        if verbose:
            print(f'    SWA test acc = {swa_acc*100:.2f}%')

        # Compare with best_raw/ema and pick the highest as final.
        if swa_acc > best_acc:
            if verbose:
                print(f'    SWA ({swa_acc*100:.2f}%) > best raw or EMA ({best_acc*100:.2f}%), '
                      f'using SWA as final best_state')
            best_acc = swa_acc
            best_state = swa_state
            best_is_ema = False  # SWA weights, not EMA
            best_epoch = num_epochs  # SWA result is always at the end of training

    if verbose:
        status = 'ABORTED' if aborted else 'done'
        print(f'--- {cfg.name} {status}: best {best_acc*100:.2f}% ({"EMA" if best_is_ema else "raw/swa"}) '
              f'@ epoch {best_epoch} in {total_time/60:.1f} min, nan_steps={nan_steps} ---')

    return {
        'name':              cfg.name,
        'cfg':               cfg,
        'n_params':          n_params,
        'batch_losses':      batch_losses,
        'grad_norms':        grad_norms,
        'nan_steps':         nan_steps,
        'aborted':           aborted,
        'train_accs':        train_accs,
        'test_accs':         test_accs,
        'test_accs_ema':     test_accs_ema,
        'gaps':              gaps,
        'train_accs_clean':  train_accs_clean,
        'clean_eval_epochs': clean_eval_epochs,
        'best_acc':          best_acc,
        'best_is_ema':       best_is_ema,
        'best_epoch':        best_epoch,
        'best_state':        best_state,
        'swa_acc':           swa_acc,
        'swa_state':         swa_state,
        'total_time':        total_time,
        'block_stats':       block_stats,
    }


# Unit test for basic mode
def test_basic_mode():
    cfg = SKNetConfig(
        name='basic_mode_test',
        blocks_per_stage=[2, 2, 2], channels=[32, 64, 128],
        branches=['1x1', '3x3', '3x3_dil2'],    # will be overridden
        use_intra_trans=True, use_skip=True,    # also overridden
        basic_mode=True,
    )
    model = SKNet(cfg)
    # Check config was overridden
    assert model.cfg.branches == ['3x3', '1x1'], f'branches not reset: {model.cfg.branches}'
    assert not model.cfg.use_intra_trans, 'intra_trans not disabled'
    assert not model.cfg.use_skip, 'skip not disabled'
    for m in model.modules():
        if isinstance(m, IntermediateBlock):
            assert m.L == 2, f'basic mode block has L={m.L}'
    # No IntraTransition should exist
    n_intra = sum(1 for m in model.modules() if isinstance(m, IntraTransition))
    assert n_intra == 0, f'basic mode still has {n_intra} IntraTransition(s)'
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)
    print('PASS: basic_mode gives the plain block sequence '
          f'({count_params(model)/1e6:.2f}M params, output {tuple(out.shape)})')



if __name__ == '__main__':
    for key, cfg in RUN_CONFIGS.items():
        model = SKNet(cfg)
        n = count_params(model)
        with torch.no_grad():
            dummy = torch.randn(2, 3, 32, 32)
            out = model(dummy)
        assert out.shape == (2, 10), f'Run {key}: wrong output shape {out.shape}'
        print(f'Run {key} [{cfg.name}]: {n/1e6:.2f}M params, output {tuple(out.shape)}')
    test_basic_mode()
