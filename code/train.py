"""Train the frozen candidate scorer and write the inference bundle.

Protocol
--------
* Evidence: hand-crafted groups (A vote structure, C text surface, R relative
  text, Q question interaction, V exact-arithmetic verification, G grounding,
  Z within-group twins) plus one dense block, the group-centred 1024-d
  embedding, under per-block L2 — the dense block is shrunk roughly 400x
  harder than the sparse statistics.
* Fit on the **hard subset of train** (groups with >= 2 distinct parsed
  answers).  The whole test bank is hard by construction, so training on the
  easy groups only dilutes the objective.
* Estimator: a single linear scorer under group-softmax CE, fitted with
  L-BFGS from two starting points and keeping the lower-loss solution.
* Model selection uses grouped out-of-fold accuracy on train-hard, confirmed
  on val-hard (bootstrap CI).  Test references are never read here.

Usage (from code/):
    python train.py --output runs/model_bundle --folds 10
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from mp2lib.baselines import load_split, hard_mask, accuracy
from mp2lib.features import build_base_features, feature_names
from mp2lib.model import (
    fit_linear_lbfgs, fit_linear, fit_mlp, scores_from_model, save_bundle,
)

SEED = 7506
HAND_GROUPS = ('A', 'C', 'R', 'Q', 'V', 'G', 'Z')
EMB_BLOCKS = ('D',)                # group-centred embedding; raw E adds nothing
L2_HAND = 1e-3
L2_EMB = 4e-1                      # swept on train-hard OOF, see experiments.py


def bootstrap_ci(picks, labels, n_boot=2000, seed=0):
    picks = np.asarray(picks, dtype=np.int64)
    labels = np.asarray(labels, dtype=bool)
    correct = labels[np.arange(len(labels)), picks]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1)
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(correct.mean()), float(lo), float(hi)


def assemble(base, emb, groups=HAND_GROUPS, emb_blocks=EMB_BLOCKS):
    """Concatenate hand groups with the dense embedding blocks."""
    from mp2lib.features import _select_groups
    parts = [_select_groups(base, tuple(groups))]
    dims = [parts[0].shape[-1]]
    e = np.asarray(emb, dtype=np.float64)
    if 'E' in emb_blocks:
        parts.append(e)
        dims.append(e.shape[-1])
    if 'D' in emb_blocks:
        parts.append(e - e.mean(axis=1, keepdims=True))
        dims.append(e.shape[-1])
    return np.concatenate(parts, axis=-1), dims


def reg_vector(dims, l2_hand=L2_HAND, l2_emb=L2_EMB):
    return np.concatenate([np.full(d, l2_hand if i == 0 else l2_emb)
                           for i, d in enumerate(dims)])


def fit_member(X, y, dims, seed, kind='lbfgs'):
    if kind == 'lbfgs':
        return fit_linear_lbfgs(X, y, l2_vector=reg_vector(dims), seed=seed)
    return fit_linear(X, y, seed=seed, l2=L2_HAND)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--kind', default='linear',
                    choices=('linear', 'mlp', 'adam'))
    ap.add_argument('--groups', default=''.join(HAND_GROUPS))
    ap.add_argument('--emb-blocks', default=''.join(EMB_BLOCKS))
    ap.add_argument('--l2-emb', type=float, default=L2_EMB)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--train-subset', default='hard',
                    choices=('hard', 'full'))
    ap.add_argument('--skip-oof', action='store_true',
                    help='fit the final model only (for quick refits)')
    args = ap.parse_args()

    t_start = time.perf_counter()
    groups = tuple(args.groups.upper())
    emb_blocks = tuple(args.emb_blocks.upper())
    l2_emb = args.l2_emb

    tr_bank, tr_arrays, tr_labels = load_split('train')
    if args.train_subset == 'hard':
        keep = np.flatnonzero(hard_mask(tr_bank))
        tr_bank = [tr_bank[i] for i in keep]
        tr_arrays = {'embeddings': tr_arrays['embeddings'][keep]}
        tr_labels = tr_labels[keep]
    tr_emb = tr_arrays['embeddings']
    n = len(tr_bank)
    print(f'train: {args.train_subset} n={n}  groups={args.groups}  '
          f'emb_blocks={args.emb_blocks}  l2_emb={l2_emb}')

    va_bank, va_arrays, va_labels = load_split('validation')
    va_hard = hard_mask(va_bank)

    tr_base = build_base_features(tr_bank, tr_arrays)
    va_base = build_base_features(va_bank, va_arrays)

    def assemble_all(base, emb):
        return assemble(base, emb, groups, emb_blocks)

    oof_acc = None
    if not args.skip_oof:
        rng = np.random.default_rng(args.seed)
        fold_ids = np.array_split(rng.permutation(n), args.folds)
        oof = np.zeros((n, 4))
        for k, ev in enumerate(fold_ids):
            fit_idx = np.setdiff1d(np.arange(n), ev)
            Xf, dims = assemble_all(tr_base[fit_idx], tr_emb[fit_idx])
            Xe, _ = assemble_all(tr_base[ev], tr_emb[ev])
            if args.kind == 'linear':
                model = fit_member(Xf, tr_labels[fit_idx], dims, args.seed + 100 * k)
            else:
                model = fit_linear(Xf, tr_labels[fit_idx], l2=L2_HAND,
                                   seed=args.seed + 100 * k)
            oof[ev] = scores_from_model(Xe, model)
        oof_acc = accuracy(oof.argmax(axis=1), tr_labels)
        print(f'  OOF (hard) accuracy: {100*oof_acc:.2f}%')

    # --- final fit on all training data ---
    Xtr, dims = assemble_all(tr_base, tr_emb)
    reg = reg_vector(dims, l2_emb=l2_emb)
    if args.kind == 'linear':
        model = fit_linear_lbfgs(Xtr, tr_labels, l2_vector=reg, seed=args.seed)
    elif args.kind == 'adam':
        model = fit_linear(Xtr, tr_labels, l2=L2_HAND, seed=args.seed)
    else:
        model = fit_mlp(Xtr, tr_labels, seed=args.seed)

    # --- validation report ---
    Xva, _ = assemble_all(va_base, va_arrays['embeddings'])
    va_scores = scores_from_model(Xva, model)
    va_picks = va_scores.argmax(axis=1)
    va_acc = accuracy(va_picks, va_labels)
    va_hard_acc, lo, hi = bootstrap_ci(va_picks[va_hard], va_labels[va_hard])
    print(f'  validation full: {100*va_acc:.2f}%')
    print(f'  validation hard: {100*va_hard_acc:.2f}%  [{100*lo:.2f},{100*hi:.2f}]')

    config = {
        'protocol': 'dase7506-mp2-open-hard-verifier-v5',
        'model_kind': model['kind'],
        'groups': args.groups,
        'emb_blocks': args.emb_blocks,
        'l2_hand': L2_HAND,
        'l2_emb': l2_emb,
        'n_features': int(Xtr.shape[-1]),
        'feature_names': feature_names(groups) +
                         [f'emb_{b}_{k}' for b in emb_blocks for k in range(1024)],
        'train_subset': args.train_subset,
        'train_n': n,
        'folds': args.folds,
        'seed': args.seed,
        'oof_hard_accuracy': oof_acc,
        'val_accuracy': float(va_acc),
        'val_hard_accuracy': float(va_hard_acc),
        'val_hard_ci': [lo, hi],
        'train_seconds': time.perf_counter() - t_start,
        'python': platform.python_version(),
        'numpy': np.__version__,
    }
    save_bundle(args.output, model, config)
    size = sum(f.stat().st_size for f in args.output.iterdir() if f.is_file())
    print(f'  bundle {args.output} ({size:,} bytes, limit {16*2**20:,})')


if __name__ == '__main__':
    main()
