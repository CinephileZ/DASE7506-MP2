"""Train linear softmax-CE scorer, evaluate on validation, save bundle.

Usage (all relative to code/):
    python train.py --output runs/model_bundle

Options:
    --split        : training split   [train]
    --folds        : number of internal folds [5]
    --epochs       : adam steps per fold [2000]
    --lr           : initial learning rate [0.05]
    --l2           : L2 regularisation strength [1e-3]
    --features     : feature-group subset [abc]
    --seed         : random seed [0]
    --val-split    : validation split for model selection [validation]
    --train-subset : 'full' or 'hard' for internal training [full]

After training, also evaluates on validation and saves the bundle.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mp2lib.baselines import (
    load_split, hard_mask, accuracy,
)
from mp2lib.features import build_features
from mp2lib.model import (
    fit_softmax_ce, predict_choices, save_bundle, scores_from_weights,
)


def _make_feature_filter(features_set):
    """Return indices for requested feature group(s)."""
    from mp2lib.features import _FEATURES_A, _FEATURES_B, _FEATURES_C
    names = features_set.lower()
    idxs = []
    if 'a' in names:
        idxs += range(len(_FEATURES_A))
    if 'b' in names:
        idxs += range(len(_FEATURES_A), len(_FEATURES_A) + len(_FEATURES_B))
    if 'c' in names:
        start = len(_FEATURES_A) + len(_FEATURES_B)
        idxs += range(start, start + len(_FEATURES_C))
    return sorted(set(idxs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='train', choices=('train', 'validation'))
    parser.add_argument('--val-split', default='validation', choices=('train', 'validation'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--features', default='abc')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-subset', default='full', choices=('full', 'hard'))
    args = parser.parse_args()

    # --- Load training data ---
    print(f"Loading train: {args.split} ({args.train_subset} subset)")
    bank, arrays, labels = load_split(args.split)
    feature_set_indices = _make_feature_filter(args.features)
    F = len(feature_set_indices)
    # Feature names for logging/reporting
    from mp2lib.features import _FEATURES_A, _FEATURES_B, _FEATURES_C
    fnames_all = list(_FEATURES_A) + list(_FEATURES_B) + list(_FEATURES_C)
    feat_names_selected = [fnames_all[i] for i in feature_set_indices]

    hard = hard_mask(bank)
    print(f"Training n={len(bank)}, features={F} (subset {'+'.join(args.features.upper())})")
    print(f"  hard={int(hard.sum())}, easy={int((~hard).sum())}")
    print(f"  correct slots: {labels.sum()} (out of {len(bank)*4})")

    # --- Optional hard-only training subset ---
    if args.train_subset == 'hard':
        hard_indices = np.flatnonzero(hard)
        bank = [bank[i] for i in hard_indices]
        arrays = {'embeddings': arrays['embeddings'][hard_indices]}
        labels = labels[hard_indices]
        hard = np.ones(len(bank), dtype=bool)
        print(f"After hard subset: n={len(bank)}")

    N = len(bank)
    feats = build_features(bank, arrays)[:, :, feature_set_indices]

    # --- Internal CV on training data ---
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    fold_ids = np.array_split(perm, args.folds)
    oof_preds = np.zeros((N, 4), dtype=np.float64)

    print("\n--- Internal 5-fold CV ---")
    for fi, fold_eval in enumerate(fold_ids):
        fold_train = np.setdiff1d(np.arange(N), fold_eval)
        result = fit_softmax_ce(feats[fold_train], labels[fold_train],
                                l2=args.l2, lr=args.lr,
                                epochs=args.epochs, seed=args.seed + fi)
        s = scores_from_weights(feats[fold_eval], result)
        picks = s.argmax(axis=1)
        acc_full = accuracy(picks, labels[fold_eval])
        hard_eval = hard[fold_eval]
        acc_hard = accuracy(picks[hard_eval], labels[fold_eval][hard_eval]) if hard_eval.any() else float('nan')
        print(f"  Fold {fi+1}: full={100*acc_full:6.2f}%  hard={100*acc_hard:6.2f}%")
        oof_preds[fold_eval] = s

    oof_picks = oof_preds.argmax(axis=1)
    oof_full = accuracy(oof_picks, labels)
    oof_hard = accuracy(oof_picks[hard], labels[hard]) if hard.any() else float('nan')
    print(f"\n  OOF: full={100*oof_full:6.2f}%  hard={100*oof_hard:6.2f}%")


    # --- Validation evaluation (development / selection) ---
    val_bank, val_arrays, val_labels = load_split(args.val_split)
    val_hard = hard_mask(val_bank)
    val_feats = build_features(val_bank, val_arrays)[:, :, feature_set_indices]

    print("\n--- Refitting on all training data ---")
    final_result = fit_softmax_ce(feats, labels, l2=args.l2, lr=args.lr,
                                  epochs=args.epochs * 2, seed=args.seed)
    val_scores = scores_from_weights(val_feats, final_result)
    val_picks = val_scores.argmax(axis=1)
    val_acc = accuracy(val_picks, val_labels)
    val_hard_acc = accuracy(val_picks[val_hard], val_labels[val_hard]) if val_hard.any() else float('nan')

    print(f"  Validation: full={100*val_acc:.2f}%  hard={100*val_hard_acc:.2f}%")

    # Save bundle
    config = {
        'split': args.split,
        'val_split': args.val_split,
        'folds': args.folds,
        'epochs': args.epochs,
        'l2': args.l2,
        'lr': args.lr,
        'features': args.features,
        'feature_count': F,
        'feature_names': feat_names_selected,
        'seed': args.seed,
        'train_subset': args.train_subset,
        'oof_accuracy': float(oof_full),
        'oof_hard_accuracy': float(oof_hard),
        'val_accuracy': float(val_acc),
        'val_hard_accuracy': float(val_hard_acc),
    }
    save_bundle(args.output, final_result, config)
    asset_size = (args.output / 'model.npz').stat().st_size
    print(f"\nSaved bundle to {args.output} ({asset_size:,} bytes, limit=16 MiB)")
    print(f"Feature count: {F}")

    if args.split != args.val_split:
        # Print feature importance (top 10 by abs weight)
        ws = np.abs(final_result['weights'])
        top_k = min(10, F)
        top_idx = np.argsort(ws)[-top_k:]
        print(f"\nTop-{top_k} feature importances (abs(weights))")
        for i in sorted(top_idx):
            print(f"  {feat_names_selected[i]:<22s} {final_result['weights'][i]:+.6f}")


if __name__ == '__main__':
    main()