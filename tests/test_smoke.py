"""Quick checks on the CPU, about 10 seconds and no downloads. Run from the repository root with
python tests/test_smoke.py or python -m pytest tests -q.

They build every configuration in RUN_CONFIGS and run one forward and backward pass, compare the parameter
counts with those printed by the training runs, run test_basic_mode of the library, check that the module name
sknet_lib loads a configuration pickled as in the published checkpoints, that train.py accepts every command in
experiments.md, and that evaluate.py has a SHA-256 for each ensemble member and runs on a small random model.
"""
import dataclasses
import os
import pickle
import re
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import sknet.lib as lib  # noqa: E402

# Parameter counts as printed by the training runs.
EXPECTED_PARAMS = {
    '1.1': 51019424,
    '2.1': 41063558,
    '2.2': 34653652,
    'I1': 59118950,
    'I2': 59118950,
}


def _forward_backward(key):
    cfg = lib.RUN_CONFIGS[key]
    torch.manual_seed(0)
    model = lib.SKNet(cfg)
    model.train()
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([3, 7])
    logits = model(x)
    assert logits.shape == (2, 10), (key, tuple(logits.shape))
    loss = F.cross_entropy(logits, y)
    assert torch.isfinite(loss), (key, loss.item())
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, (key, name)
        assert torch.isfinite(p.grad).all(), (key, name)
    # With DropPath a whole block can be dropped for both samples of this tiny
    # batch, so the check that every block receives a gradient is skipped there.
    if 'droppath' not in cfg.stoch_depth_mode:
        for i, blk in enumerate(m for m in model.modules() if isinstance(m, lib.IntermediateBlock)):
            assert blk.fc.weight.grad.abs().max() > 0, (key, 'block', i, 'gets no gradient')
    return model


def test_every_config_forward_backward():
    for key in lib.RUN_CONFIGS:
        _forward_backward(key)


def test_parameter_counts_match_logs():
    for key, n in EXPECTED_PARAMS.items():
        assert lib.count_params(lib.SKNet(lib.RUN_CONFIGS[key])) == n, key


def test_basic_mode():
    lib.test_basic_mode()


def test_sknet_lib_is_the_library_module():
    import sknet_lib
    assert sknet_lib is lib
    # train.py assigns module-level variables through this name.
    old = sknet_lib.test_loader
    try:
        sknet_lib.test_loader = 'sentinel'
        assert lib.test_loader == 'sentinel'
    finally:
        sknet_lib.test_loader = old


def test_config_pickled_under_old_module_name():
    # The checkpoints of May 2026 hold a configuration of class sknet_lib.SKNetConfig
    # with one more field, shake_shake=False, written by the historical library.
    cfg = dataclasses.replace(lib.RUN_CONFIGS['I2'])
    cfg.__dict__['shake_shake'] = False
    data = pickle.dumps(cfg, protocol=2)  # torch.save uses pickle protocol 2
    new_ref = b'c' + lib.SKNetConfig.__module__.encode() + b'\nSKNetConfig\n'
    assert new_ref in data
    loaded = pickle.loads(data.replace(new_ref, b'csknet_lib\nSKNetConfig\n'))
    assert type(loaded) is lib.SKNetConfig
    assert loaded == lib.RUN_CONFIGS['I2'] and loaded.shake_shake is False
    assert lib.count_params(lib.SKNet(loaded)) == EXPECTED_PARAMS['I2']
    # Older configurations without the two DropPath fields (and without
    # shake_shake, added above) still build, because the dataclass defaults
    # fill them in.
    for field in ('stoch_depth_mode', 'stoch_depth_max_rate', 'shake_shake'):
        loaded.__dict__.pop(field)
    assert lib.count_params(lib.SKNet(loaded)) == EXPECTED_PARAMS['I2']


def _commands_in_experiments_md():
    text = open(os.path.join(ROOT, 'experiments.md'), encoding='utf-8').read()
    return [line.strip() for line in text.splitlines() if line.strip().startswith('python train.py ')]


def test_experiment_commands_parse():
    import train
    commands = _commands_in_experiments_md()
    assert len(commands) >= 10, len(commands)
    argv = sys.argv
    try:
        for cmd in commands:
            sys.argv = shlex.split(cmd)[1:]
            args = train.parse_args()  # exits with an error on an unknown flag
            assert args.run in lib.RUN_CONFIGS
    finally:
        sys.argv = argv


def test_evaluate_script():
    import evaluate
    sums = evaluate.read_sha256sums(os.path.join(ROOT, 'SHA256SUMS'))
    for _, fname, key in evaluate.MEMBERS:
        assert re.fullmatch(r'[0-9a-f]{64}', sums[fname]), fname
        assert key in lib.RUN_CONFIGS
    cfg = lib.SKNetConfig(name='tiny', blocks_per_stage=[1, 1, 1], channels=[8, 16, 32])
    loader = [(torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2]))]
    p_plain, p_flip, y = evaluate.softmax_outputs(lib.SKNet(cfg), loader, torch.device('cpu'))
    assert p_plain.shape == p_flip.shape == (3, 10) and torch.equal(y, loader[0][1])
    assert torch.allclose(p_flip.sum(1), torch.ones(3))


if __name__ == '__main__':
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith('test_') and callable(f)]
    for name, f in tests:
        f()
        print('ok', name)
    print(f'{len(tests)} tests passed')
