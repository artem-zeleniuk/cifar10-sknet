"""Original module name of the library in sknet/lib.py.

The published checkpoints were saved while the library was one file called
sknet_lib.py, so their pickled configuration refers to sknet_lib.SKNetConfig.
train.py and tta_eval.py also import the library under this name.

This file makes the name sknet_lib point to the module sknet.lib itself, not to
a copy of its names. That matters because train.py assigns module-level
variables (L.device, L.train_loader, L.test_loader) that train_one_config
reads. With "from sknet.lib import *" these values would be set in a second
namespace and the training loop would not see them.
"""
import sys

from sknet import lib as _lib

sys.modules[__name__] = _lib
