"""Evaluate the four published ensemble members and their ensemble on the CIFAR-10 test set.

One command downloads the four checkpoints from Hugging Face (about 1.5 GB),
checks each file against its SHA-256 in SHA256SUMS before loading it, and
prints the test accuracy of every model and of the ensemble, without and with
horizontal-flip averaging:

    python evaluate.py

The evaluation is the one of Section 13 of the notebook: the ensemble averages
the softmax outputs of the four models, and flip averaging averages the softmax
outputs for each image and its mirror image. On a GPU the forward pass runs in
bfloat16 autocast, as in the notebook. A run on the CPU works but is slow.

    python evaluate.py --dry-run    # no download: random weights and images, checks the code path

Options: --weights-dir (default weights/), --data-root (default ./data),
--batch-size, --num-workers, --device.
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request

import torch
import torch.nn.functional as F
import torchvision

import sknet_lib as L  # registers the module name that the pickled checkpoints refer to

ROOT = os.path.dirname(os.path.abspath(__file__))
HF_URL = 'https://huggingface.co/Artem3000-dev/cifar-10-sknet/resolve/main/'

# Ensemble members in the order of the notebook: (name, file on Hugging Face, entry of RUN_CONFIGS).
MEMBERS = [
    ('ft_g0', 'ft_g0_ckpt.pt', '1.1'),
    ('ft_1_1', 'ft_1_1_ckpt.pt', '1.1'),
    ('I1_historical', 'I1_historical_ckpt.pt', 'I1'),
    ('ft_I2', 'ft_I2_ckpt.pt', 'I2'),
]

# Saved outputs of the notebook (Google Colab, Tesla T4), Section 13: (plain, flip averaging).
NOTEBOOK = {
    'ft_g0': (98.26, 98.25),
    'ft_1_1': (97.96, 98.05),
    'I1_historical': (97.99, 98.09),
    'ft_I2': (98.31, 98.36),
    'ensemble': (98.42, 98.37),
}


def read_sha256sums(path):
    sums = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                sums[os.path.basename(parts[1])] = parts[0]
    return sums


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(fname, weights_dir, expected):
    """Return the path of a verified local copy of fname, downloading it if needed."""
    path = os.path.join(weights_dir, fname)
    if os.path.exists(path):
        if sha256_of(path) != expected:
            sys.exit(f'{path} exists but its SHA-256 does not match SHA256SUMS. Remove it and run again.')
        print(f'  {fname}: found, SHA-256 matches')
        return path
    os.makedirs(weights_dir, exist_ok=True)
    part = path + '.part'
    print(f'  {fname}: downloading from {HF_URL} ...', flush=True)
    urllib.request.urlretrieve(HF_URL + fname, part)
    if sha256_of(part) != expected:
        sys.exit(f'{part}: SHA-256 does not match SHA256SUMS. The file was not loaded.')
    os.replace(part, path)
    print(f'  {fname}: downloaded, SHA-256 matches')
    return path


@torch.no_grad()
def softmax_outputs(model, loader, device):
    """Softmax outputs for the test images and for their mirror images, and the labels."""
    model.eval()
    p_plain, p_flip, labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            l1 = model(x).float()
            l2 = model(torch.flip(x, dims=[3])).float()
        p1 = F.softmax(l1, dim=1)
        p2 = F.softmax(l2, dim=1)
        p_plain.append(p1.cpu())
        p_flip.append(((p1 + p2) / 2).cpu())
        labels.append(y.cpu())
    return torch.cat(p_plain), torch.cat(p_flip), torch.cat(labels)


def accuracy(probs, labels):
    return 100.0 * (probs.argmax(1) == labels).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description='Evaluate the published ensemble on the CIFAR-10 test set.')
    ap.add_argument('--weights-dir', default=os.path.join(ROOT, 'weights'))
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dry-run', action='store_true',
                    help='no download: random weights and 8 random images, to check the code path')
    args = ap.parse_args()
    device = torch.device(args.device)

    sums = read_sha256sums(os.path.join(ROOT, 'SHA256SUMS'))
    missing = [f for _, f, _ in MEMBERS if f not in sums]
    if missing:
        sys.exit(f'SHA256SUMS has no entry for {missing}')

    if args.dry_run:
        print('Dry run: random weights and random images. The accuracies below are meaningless.')
        g = torch.Generator().manual_seed(0)
        loader = [(torch.randn(4, 3, 32, 32, generator=g), torch.randint(0, 10, (4,), generator=g))
                  for _ in range(2)]
    else:
        if device.type != 'cuda':
            print('No GPU: the evaluation runs on the CPU in 32-bit precision and takes a long time.')
        print(f'Checkpoints in {args.weights_dir}:')
        paths = {name: fetch(fname, args.weights_dir, sums[fname]) for name, fname, _ in MEMBERS}
        test_set = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True,
                                                transform=L.test_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    rows, all_plain, all_flip, labels = [], [], [], None
    for name, fname, key in MEMBERS:
        t0 = time.time()
        if args.dry_run:
            model = L.SKNet(L.RUN_CONFIGS[key])
        else:
            ck = torch.load(paths[name], map_location='cpu', weights_only=False)  # verified above
            model = L.SKNet(ck['cfg'])
            model.load_state_dict(ck['state_dict'])
        model = model.to(device)
        p_plain, p_flip, y = softmax_outputs(model, loader, device)
        if labels is None:
            labels = y
        assert torch.equal(labels, y)
        all_plain.append(p_plain)
        all_flip.append(p_flip)
        rows.append((name, accuracy(p_plain, labels), accuracy(p_flip, labels)))
        print(f'  evaluated {name} in {time.time() - t0:.0f} s', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    rows.append(('ensemble', accuracy(torch.stack(all_plain).mean(0), labels),
                 accuracy(torch.stack(all_flip).mean(0), labels)))

    print()
    print(f'Test accuracy on {len(labels)} images (in brackets: saved notebook output, Tesla T4)')
    print(f'{"model":<15}{"plain":>18}{"flip averaging":>22}')
    for name, acc_plain, acc_flip in rows:
        nb_plain, nb_flip = NOTEBOOK[name]
        print(f'{name:<15}{acc_plain:>9.2f}% ({nb_plain:.2f}%){acc_flip:>13.2f}% ({nb_flip:.2f}%)')


if __name__ == '__main__':
    main()
