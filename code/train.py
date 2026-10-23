"""Train the frozen candidate scorer and write the inference bundle.

Protocol
--------
* Features: answer agreement (A), embedding geometry (B), answer-cluster
  coherence (S), text surface (C), within-group relative text (R), and
  question interaction (Q), plus a PCA of the group-centred, L2-normalised
  candidate embeddings (P).
* Fit on the **hard subset of train** (groups with >= 2 distinct parsed
  answers).  The whole test bank is hard by construction, so training on the
  easy groups only dilutes the objective.
* Model selection uses 5-fold grouped OOF accuracy on train-hard, confirmed
  on val-hard (bootstrap CI).  Test references are never read here.

Usage (from code/):
    python train.py --output runs/model_bundle --kind linear --pca 256 --seed 0
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from mp2lib.baselines import load_split, hard_mask, accuracy
from mp2lib.features import build_features, feature_names
from mp2lib.model import (
    fit_linear, fit_linear_ensemble, fit_mlp, scores_from_model, save_bundle,
)
from mp2lib.pca import fit_pca


def bootstrap_ci(picks, labels, n_boot=2000, seed=0):
    picks = np.asarray(picks, dtype=np.int64)
    labels = np.asarray(labels, dtype=bool)
    correct = labels[np.arange(len(labels)), picks]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1)
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(correct.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--kind', default='linear',
                    choices=('linear', 'ensemble', 'mlp'))
    ap.add_argument('--pca', type=int, default=256)
    ap.add_argument('--groups', default='ABSCRQ')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--l2', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=96)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--n-models', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-subset', default='hard',
                    choices=('hard', 'full'))
    args = ap.parse_args()

    t_start = time.perf_counter()
    groups = tuple(args.groups.upper())

    tr_bank, tr_arrays, tr_labels = load_split('train')
    if args.train_subset == 'hard':
        keep = np.flatnonzero(hard_mask(tr_bank))
        tr_bank = [tr_bank[i] for i in keep]
        tr_arrays = {'embeddings': tr_arrays['embeddings'][keep]}
        tr_labels = tr_labels[keep]
    tr_emb = tr_arrays['embeddings']
    n = len(tr_bank)
    print(f'train: {args.train_subset} n={n}  groups={args.groups}  pca={args.pca}')

    va_bank, va_arrays, va_labels = load_split('validation')
    va_hard = hard_mask(va_bank)

    # --- 5-fold OOF on the training subset ---
    rng = np.random.default_rng(args.seed)
    fold_ids = np.array_split(rng.permutation(n), args.folds)
    oof = np.zeros((n, 4))
    for k, ev in enumerate(fold_ids):
        fit_idx = np.setdiff1d(np.arange(n), ev)
        pca = fit_pca(tr_emb[fit_idx], args.pca) if args.pca else None
        Xf = build_features([tr_bank[i] for i in fit_idx],
                            {'embeddings': tr_emb[fit_idx]}, groups, pca)
        Xe = build_features([tr_bank[i] for i in ev],
                            {'embeddings': tr_emb[ev]}, groups, pca)
        model = _fit(args, Xf, tr_labels[fit_idx], seed=args.seed + 100 * k)
        oof[ev] = scores_from_model(Xe, model)
    oof_acc = accuracy(oof.argmax(axis=1), tr_labels)
    print(f'  OOF (hard) accuracy: {100*oof_acc:.2f}%')

    # --- final fit on all training data ---
    pca = fit_pca(tr_emb, args.pca) if args.pca else None
    Xtr = build_features(tr_bank, tr_arrays, groups, pca)
    model = _fit(args, Xtr, tr_labels, seed=args.seed)

    # --- validation report ---
    Xva = build_features(va_bank, va_arrays, groups, pca)
    va_picks = scores_from_model(Xva, model).argmax(axis=1)
    va_acc = accuracy(va_picks, va_labels)
    va_hard_acc, lo, hi = bootstrap_ci(va_picks[va_hard], va_labels[va_hard])
    print(f'  validation full: {100*va_acc:.2f}%')
    print(f'  validation hard: {100*va_hard_acc:.2f}%  [{100*lo:.2f},{100*hi:.2f}]')

    names = feature_names(groups, args.pca)
    config = {
        'protocol': 'dase7506-mp2-open-hard-verifier-v5',
        'model_kind': args.kind,
        'groups': args.groups,
        'n_pca': args.pca,
        'n_features': int(Xtr.shape[-1]),
        'feature_names': names,
        'train_subset': args.train_subset,
        'train_n': n,
        'folds': args.folds,
        'epochs': args.epochs,
        'lr': args.lr,
        'l2': args.l2,
        'hidden': args.hidden if args.kind == 'mlp' else None,
        'dropout': args.dropout if args.kind == 'mlp' else None,
        'n_models': args.n_models if args.kind == 'ensemble' else 1,
        'seed': args.seed,
        'oof_hard_accuracy': float(oof_acc),
        'val_accuracy': float(va_acc),
        'val_hard_accuracy': float(va_hard_acc),
        'val_hard_ci': [lo, hi],
        'train_seconds': time.perf_counter() - t_start,
        'python': platform.python_version(),
        'numpy': np.__version__,
    }
    save_bundle(args.output, model, config, pca)
    size = sum(f.stat().st_size for f in args.output.iterdir() if f.is_file())
    print(f'  bundle {args.output} ({size:,} bytes, limit {16*2**20:,})')


def _fit(args, X, y, seed):
    if args.kind == 'ensemble':
        return fit_linear_ensemble(X, y, n_models=args.n_models, seed=seed,
                                   l2=args.l2, lr=args.lr, epochs=args.epochs)
    if args.kind == 'linear':
        return fit_linear(X, y, l2=args.l2, lr=args.lr, epochs=args.epochs,
                          seed=seed)
    return fit_mlp(X, y, hidden=args.hidden, l2=args.l2, lr=1e-3,
                   epochs=args.epochs, dropout=args.dropout, seed=seed)


if __name__ == '__main__':
    main()
