"""CNN for CIFAR-10 built from one attention block type: model, data pipeline and training loop.

The code is in sknet/lib.py. The names below are re-exported for convenience.
Module-level state that the training loop reads (device, train_loader,
test_loader) has to be set on sknet.lib itself, as train.py does.
"""
from .lib import (
    EMA,
    RUN_CONFIGS,
    SAM,
    IntermediateBlock,
    IntraTransition,
    SKNet,
    SKNetConfig,
    StageTransition,
    count_params,
    evaluate,
    evaluate_tta,
    train_one_config,
)
