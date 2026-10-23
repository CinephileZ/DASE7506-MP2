"""Experiment driver: feature-set and model sweep with OOF CV.

Protocol
--------
* Development metric is **OOF accuracy on the hard subset of train** (5-fold,
  grouped by question) — ~2114 groups, the same regime as the test bank.
* **val-hard** (355 groups) is the confirmation set; bootstrap CIs are printed
  because it is small.
* Test references are never touched here.

Usage (from code/):
    python experiments.py --list
    python experiments.py --exp lin_ac
    python experiments.py --sweep
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mp2lib.baselines import load_split, hard_mask
from mp2lib.features import build_features, build_base_features, feature_names
from mp2lib.model import fit_linear, fit_mlp, scores_from_model
from mp2lib.pca import fit_pca

CACHE = Path('runs/feature_cache.npz')


# ---------------------------------------------------------------------------
# data / cache
# ---------------------------------------------------------------------------

def load_data():
    out = {}
    for split in ('train', 'validation'):
        bank, arrays, labels = load_split(split)
        out[split] = (bank, arrays, labels, hard_mask(bank))
    return out


def base_features(split, bank, arrays):
    """Base features (groups A,B,S,C,R,Q) with an on-disk cache."""
    key = f'{split}_base'
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as f:
            if key in f.files:
                return f[key]
    mat = build_base_features(bank, arrays)
    store = {}
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as f:
            store = {k: f[k] for k in f.files}
    store[key] = mat
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **store)
    return mat


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def accuracy(picks, labels):
    return float(labels[np.arange(len(labels)), np.asarray(picks, int)].mean())


def bootstrap_ci(picks, labels, n_boot=2000, seed=0):
    picks = np.asarray(picks, dtype=np.int64)
    labels = np.asarray(labels, dtype=bool)
    correct = labels[np.arange(len(labels)), picks]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1)
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(correct.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# one experiment
# ---------------------------------------------------------------------------

def run_experiment(data, groups, n_pca=0, kind='linear', folds=5, seed=0,
                   model_kw=None, hard_only=True, n_models=1):
    model_kw = dict(model_kw or {})
    n_models = int(model_kw.pop('n_models', 1))
    tr_bank, tr_arrays, tr_labels, tr_hard = data['train']
    va_bank, va_arrays, va_labels, va_hard = data['validation']

    tr_base = base_features('train', tr_bank, tr_arrays)
    va_base = base_features('validation', va_bank, va_arrays)

    # restrict to the hard subset (test-like regime)
    if hard_only:
        tri = np.flatnonzero(tr_hard)
        tr_base, tr_labels_g = tr_base[tri], tr_labels[tri]
        tr_emb = tr_arrays['embeddings'][tri]
    else:
        tr_labels_g = tr_labels
        tr_emb = tr_arrays['embeddings']

    n = len(tr_labels_g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds_idx = np.array_split(perm, folds)

    fit = fit_linear if kind == 'linear' else fit_mlp
    oof = np.zeros((n, 4))
    for k, ev in enumerate(folds_idx):
        trn = np.setdiff1d(np.arange(n), ev)
        pca = fit_pca(tr_emb[trn], n_pca, seed=seed) if n_pca else None
        Xtr = _assemble(tr_base[trn], tr_emb[trn], groups, pca)
        Xev = _assemble(tr_base[ev], tr_emb[ev], groups, pca)
        oof[ev] = np.mean([
            scores_from_model(Xev, fit(Xtr, tr_labels_g[trn], seed=seed + 100 * k + m, **model_kw))
            for m in range(n_models)
        ], axis=0)

    oof_picks = oof.argmax(axis=1)
    oof_acc = accuracy(oof_picks, tr_labels_g)

    # refit on all training data, evaluate on validation
    pca = fit_pca(tr_emb, n_pca, seed=seed) if n_pca else None
    Xtr = _assemble(tr_base, tr_emb, groups, pca)
    Xva = _assemble(va_base, va_arrays['embeddings'], groups, pca)
    va_scores = np.mean([
        scores_from_model(Xva, fit(Xtr, tr_labels_g, seed=seed + m, **model_kw))
        for m in range(n_models)
    ], axis=0)
    va_picks = va_scores.argmax(axis=1)
    va_acc = accuracy(va_picks, va_labels)
    va_hard_acc, lo, hi = bootstrap_ci(va_picks[va_hard], va_labels[va_hard])

    return {
        'groups': ''.join(groups), 'n_pca': n_pca, 'kind': kind, 'n_features': int(Xva.shape[-1]),
        'oof_hard': oof_acc, 'val_full': va_acc, 'val_hard': va_hard_acc,
        'val_hard_ci': (lo, hi),
        'va_picks': va_picks, 'va_scores': va_scores,
        'oof_picks': oof_picks, 'n_train_hard': n,
    }


def _assemble(base, emb, groups, pca):
    from mp2lib.features import _select_groups
    parts = [_select_groups(base, tuple(groups))]
    if pca is not None:
        parts.append(pca.transform(emb))
    return np.concatenate(parts, axis=-1)


ALL = ('A', 'B', 'S', 'C', 'R', 'Q')

EXPERIMENTS = {
    # name: (groups, n_pca, kind, extra kwargs)
    'lin_a':      (('A',), 0, 'linear', {}),
    'lin_ac':     (('A', 'C'), 0, 'linear', {}),
    'lin_acsrq':  (('A', 'C', 'S', 'R', 'Q'), 0, 'linear', {}),
    'lin_all':    (ALL, 0, 'linear', {}),
    'lin_all_pc': (ALL, 32, 'linear', {}),
    'mlp_all_pc': (ALL, 32, 'mlp', {'hidden': 64}),
    'mlp_all_pc64': (ALL, 64, 'mlp', {'hidden': 96}),
}

# systematic grids -----------------------------------------------------------

def pca_grid():
    for k in (0, 8, 16, 32, 64, 96, 128, 192, 256):
        yield f'pca{k}_lin', (ALL, k, 'linear', {})
        yield f'pca{k}_mlp', (ALL, k, 'mlp', {'hidden': 96, 'epochs': 1200})


def linear_reg_grid():
    for l2 in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        yield f'lin_l2_{l2:g}', (ALL, 64, 'linear', {'l2': l2})


def mlp_reg_grid():
    for hidden in (32, 64, 128):
        for l2 in (3e-4, 1e-3, 3e-3, 1e-2):
            for dr in (0.0, 0.2):
                yield (f'mlp_h{hidden}_l2_{l2:g}_dr{dr:g}',
                       (ALL, 64, 'mlp', {'hidden': hidden, 'l2': l2,
                                         'dropout': dr, 'epochs': 1200}))


def big_pca_grid():
    for k in (256, 320, 384, 512, 768):
        yield f'big{k}_lin', (ALL, k, 'linear', {})


def full_train_grid():
    for k in (128, 256):
        yield (f'fulltrain_pca{k}_lin',
               (ALL, k, 'linear', {'_full_train': True}))


def ensemble_grid():
    for m in (1, 3, 5, 9):
        yield f'ens{m}_pca256', (ALL, 256, 'linear', {'n_models': m})


def mlp2_grid():
    for hidden in (64, 128):
        for l2 in (1e-3, 1e-2):
            for dr in (0.0, 0.3):
                yield (f'm2_h{hidden}_l2_{l2:g}_dr{dr:g}',
                       (ALL, 256, 'mlp', {'hidden': hidden, 'l2': l2,
                                          'dropout': dr, 'epochs': 3000,
                                          'lr': 1e-3}))


def ablate_grid():
    """Feature-group ablation at the chosen PCA size (report RQ1)."""
    K = 256
    for name, groups, k in (
        ('abl_A',            ('A',), 0),
        ('abl_A_pca',        ('A',), K),
        ('abl_A_C_R_Q',      ('A', 'C', 'R', 'Q'), 0),
        ('abl_A_C_R_Q_pca',  ('A', 'C', 'R', 'Q'), K),
        ('abl_A_B_S_pca',    ('A', 'B', 'S'), K),
        ('abl_full_pca',     ALL, K),
    ):
        yield name, (groups, k, 'linear', {})


def gsel_grid():
    """Groups x PCA size around the current optimum."""
    for gl in ('ACRQ', 'ABCRQ', 'ASCRQ', 'ABSCRQ'):
        for k in (192, 256, 320):
            yield (f'g_{gl}_p{k}', (tuple(gl), k, 'linear', {}))


def rq_grid():
    """RQ3: does a candidate have to be judged against its peers?"""
    for name, groups, k in (
        ('rq_point_C',        ('C',), 0),            # text only, no context
        ('rq_point_CQ',       ('C', 'Q'), 0),        # text + question only
        ('rq_ctx_A',          ('A',), 0),            # peer answers only
        ('rq_ctx_ACRQ_p256',  ('A', 'C', 'R', 'Q'), 256),   # full
        ('rq_emb_only_p256',  (), 256),              # peer embeddings only
    ):
        yield name, (groups, k, 'linear', {})


MEMBERS = {
    'l128': ('ACRQ', 128, 'linear'),
    'l256': ('ACRQ', 256, 'linear'),
    'l384': ('ACRQ', 384, 'linear'),
    'lnop': ('ABSCRQ', 0, 'linear'),
    'm256': ('ACRQ', 256, 'mlp'),
}


def run_mix(data, member_keys, seed=0, folds=5):
    """Average z-scored scores of several heterogeneous members (OOF + val)."""
    from mp2lib.model import fit_mlp, fit_linear, scores_from_model
    tr_bank, tr_arrays, tr_labels, tr_hard = data['train']
    va_bank, va_arrays, va_labels, va_hard = data['validation']
    tri = np.flatnonzero(tr_hard)
    tr_base = base_features('train', tr_bank, tr_arrays)[tri]
    tr_lab = tr_labels[tri]
    tr_emb = tr_arrays['embeddings'][tri]
    va_base = base_features('validation', va_bank, va_arrays)
    va_emb = va_arrays['embeddings']

    n = len(tr_lab)
    fold_ids = np.array_split(np.random.default_rng(seed).permutation(n), folds)
    oof = np.zeros((n, 4))
    for k, ev in enumerate(fold_ids):
        fit_idx = np.setdiff1d(np.arange(n), ev)
        acc = np.zeros((len(ev), 4))
        for key in member_keys:
            groups, npca, kind = MEMBERS[key]
            pca = fit_pca(tr_emb[fit_idx], npca) if npca else None
            Xf = _assemble(tr_base[fit_idx], tr_emb[fit_idx], groups, pca)
            Xe = _assemble(tr_base[ev], tr_emb[ev], groups, pca)
            if kind == 'linear':
                m = fit_linear(Xf, tr_lab[fit_idx], seed=seed + 100 * k)
            else:
                m = fit_mlp(Xf, tr_lab[fit_idx], hidden=128, l2=1e-3, lr=1e-3,
                            epochs=3000, seed=seed + 100 * k)
            s = scores_from_model(Xe, m)
            acc += (s - s.mean()) / max(s.std(), 1e-9)
        oof[ev] = acc / len(member_keys)
    oof_acc = accuracy(oof.argmax(axis=1), tr_lab)

    va = np.zeros((len(va_bank), 4))
    for key in member_keys:
        groups, npca, kind = MEMBERS[key]
        pca = fit_pca(tr_emb, npca) if npca else None
        Xt = _assemble(tr_base, tr_emb, groups, pca)
        Xv = _assemble(va_base, va_emb, groups, pca)
        if kind == 'linear':
            m = fit_linear(Xt, tr_lab, seed=seed)
        else:
            m = fit_mlp(Xt, tr_lab, hidden=128, l2=1e-3, lr=1e-3, epochs=3000,
                        seed=seed)
        s = scores_from_model(Xv, m)
        va += (s - s.mean()) / max(s.std(), 1e-9)
    va /= len(member_keys)
    picks = va.argmax(axis=1)
    hard_acc, lo, hi = bootstrap_ci(picks[va_hard], va_labels[va_hard])
    return oof_acc, accuracy(picks, va_labels), hard_acc, lo, hi


def mix_grid():
    yield 'mix_l256', (('ACRQ',), 256, 'mix', {'members': ('l256',)})
    yield 'mix_l128_l256_l384', (('ACRQ',), 256, 'mix',
                                 {'members': ('l128', 'l256', 'l384')})
    yield 'mix_l256_lnop', (('ACRQ',), 256, 'mix',
                            {'members': ('l256', 'lnop')})
    yield 'mix_all5', (('ACRQ',), 256, 'mix',
                       {'members': ('l128', 'l256', 'l384', 'lnop', 'm256')})


GRIDS = {'pca': pca_grid, 'lin': linear_reg_grid, 'mlp': mlp_reg_grid,
         'big': big_pca_grid, 'ens': ensemble_grid, 'full': full_train_grid,
         'mlp2': mlp2_grid, 'ablate': ablate_grid, 'gsel': gsel_grid,
         'rq': rq_grid, 'mix': mix_grid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--exp', action='append', default=[])
    ap.add_argument('--grid', action='append', default=[],
                    help=f'named grid: {sorted(GRIDS)}')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--seeds', type=int, default=1,
                    help='average OOF/val over this many seeds')
    args = ap.parse_args()

    if args.list:
        for k, v in EXPERIMENTS.items():
            print(f'{k:16s} groups={v[0]} pca={v[1]} kind={v[2]} {v[3]}')
        for g in GRIDS:
            print(f'grid {g}: {len(list(GRIDS[g]()))} configs')
        return

    todo = dict(EXPERIMENTS) if args.sweep else {}
    for name in args.exp:
        todo[name] = EXPERIMENTS[name]
    for g in args.grid:
        todo.update(dict(GRIDS[g]()))

    data = load_data()
    print(f'train-hard n={int(data["train"][3].sum())}  '
          f'val-hard n={int(data["validation"][3].sum())}  seeds={args.seeds}')
    print(f'{"exp":22s} {"F":>4s} {"oof_hard":>9s} {"val_full":>9s} {"val_hard":>9s}  CI95')
    results = {}
    for name, (groups, n_pca, kind, kw) in todo.items():
        t0 = time.perf_counter()
        kw = dict(kw)
        if kind == 'mix':
            res = [run_mix(data, kw['members'], seed=s) for s in range(args.seeds)]
            oof = float(np.mean([r[0] for r in res]))
            full = float(np.mean([r[1] for r in res]))
            hard = float(np.mean([r[2] for r in res]))
            lo = float(np.mean([r[3] for r in res]))
            hi = float(np.mean([r[4] for r in res]))
            nfeat = 0
        else:
            hard_only = not kw.pop('_full_train', False)
            rows = [run_experiment(data, groups, n_pca, kind, seed=s, model_kw=kw,
                                   hard_only=hard_only)
                    for s in range(args.seeds)]
            oof = float(np.mean([r['oof_hard'] for r in rows]))
            full = float(np.mean([r['val_full'] for r in rows]))
            hard = float(np.mean([r['val_hard'] for r in rows]))
            lo = float(np.mean([r['val_hard_ci'][0] for r in rows]))
            hi = float(np.mean([r['val_hard_ci'][1] for r in rows]))
            nfeat = rows[0]['n_features']
        dt = time.perf_counter() - t0
        print(f'{name:22s} {nfeat:4d} {100*oof:8.2f}% {100*full:8.2f}% '
              f'{100*hard:8.2f}%  [{100*lo:.2f},{100*hi:.2f}]  ({dt:.1f}s)')
        results[name] = {
            'groups': ''.join(groups), 'n_pca': n_pca, 'kind': kind,
            'model_kw': kw, 'n_features': nfeat,
            'oof_hard': oof, 'val_full': full, 'val_hard': hard,
            'val_hard_ci': (lo, hi), 'seeds': args.seeds,
        }

    out = Path('runs/experiments.json')
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(results)
    out.write_text(json.dumps(prev, indent=2, default=float) + '\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
