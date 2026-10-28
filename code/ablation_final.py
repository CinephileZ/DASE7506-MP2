"""Evidence-source and estimator-complexity ablations for the report.

Three of the guide's research questions, one shared protocol — the frozen
bundle's own (``runs/model_bundle_v2``): hard train subset (n=2114), grouped
10-fold OOF at seed 7506, L-BFGS two-start linear fits with per-block L2
(hand 1e-3 / embedding 4e-1), validation-only reporting.

  RQ1  "Does reading the reasoning actually help?"  — an evidence ladder:
       parsed answers only -> +question interaction -> +text surface ->
       +arithmetic verification -> +supplied embedding -> frozen combination.
  RQ2  "What makes a wrong answer look convincing?" — the surface step
       isolates length/format/style, the verification step isolates exact
       arithmetic checking, and the frozen weights' mass is split across
       style / verification / vote features.
  RQ5  "When does more computation pay off?" — training-free majority vote
       vs the frozen linear scorer vs an MLP on the same features (the GBDT
       probe is recorded in runs/experiments.json and cited, not re-run).

Every arm is analysis-only: nothing is adopted, the frozen bundle is
untouched, and the test bank is never read.  Single seed 7506, no averaging.

Usage (from code/):
    python ablation_final.py                  # ladder (fast arms)
    python ablation_final.py --arms mlp_full  # the slow MLP arm alone
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from mp2lib.baselines import load_split, hard_mask, accuracy, baseline_vote
from mp2lib.features import (
    build_base_features, _BASE_GROUPS, _GROUP_NAMES, _Z_SOURCE,
)
from mp2lib.model import fit_linear_lbfgs, fit_mlp, scores_from_model
from train import SEED, L2_HAND, L2_EMB, bootstrap_ci

FOLDS = 10
RUNS = Path('runs')
EXPERIMENTS = RUNS / 'experiments.json'
RESULTS = RUNS / 'ablation_final.json'
BUNDLE = RUNS / 'model_bundle_v2'

# ---------------------------------------------------------------------------
# column geometry of build_base_features output ([N, 4, 120])
# ---------------------------------------------------------------------------

_SPANS = {}
_pos = 0
for _g in _BASE_GROUPS:
    _w = len(_GROUP_NAMES[_g])
    _SPANS[_g] = (_pos, _pos + _w)
    _pos += _w
assert len(_Z_SOURCE) == len(_GROUP_NAMES['Z'])

# _Z_SOURCE order is C, Q, V, G — spans of each source inside the Z block
_ZSPANS = {}
_pos = 0
for _g in ('C', 'Q', 'V', 'G'):
    _w = len(_GROUP_NAMES[_g])
    _ZSPANS[_g] = (_pos, _pos + _w)
    _pos += _w


def _hand_cols(groups, z_sources=()):
    idx = []
    for g in groups:
        idx += list(range(*_SPANS[g]))
    if z_sources:
        z0 = _SPANS['Z'][0]
        for g in z_sources:
            idx += [z0 + i for i in range(*_ZSPANS[g])]
    return idx


ARMS = {
    'vote': None,                       # training-free majority vote
    'A':       dict(cols=_hand_cols('A')),
    'AQ':      dict(cols=_hand_cols('AQ')),
    'surface': dict(cols=_hand_cols('ACRQ', ('C', 'Q'))),
    'hand':    dict(cols=_hand_cols('ACRQVGZ')),
    'embD':    dict(cols=[], use_d=True),
    'full':    dict(cols=_hand_cols('ACRQVGZ'), use_d=True),
    'mlp_full': dict(cols=_hand_cols('ACRQVGZ'), use_d=True, estimator='mlp'),
}
ARM_ORDER = ['vote', 'A', 'AQ', 'surface', 'hand', 'embD', 'full', 'mlp_full']

EVIDENCE = {
    'vote': 'majority vote over parsed answers (training-free)',
    'A': 'parsed final answers only (vote structure)',
    'AQ': '+ question/answer interaction',
    'surface': '+ text-surface statistics C,R and twins z(C,Q)',
    'hand': '+ arithmetic verification V, grounding G, twins z(V,G)',
    'embD': 'group-centred embedding block only',
    'full': 'hand features + embedding (the frozen method)',
    'mlp_full': 'frozen features, MLP estimator instead of linear',
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_data():
    tr_bank, tr_arrays, tr_labels = load_split('train')
    keep = np.flatnonzero(hard_mask(tr_bank))
    tr_bank = [tr_bank[i] for i in keep]
    tr_emb = tr_arrays['embeddings'][keep]
    tr_labels = tr_labels[keep]
    va_bank, va_arrays, va_labels = load_split('validation')
    va_hard = hard_mask(va_bank)
    print(f'train-hard n={len(tr_bank)}  val n={len(va_bank)} '
          f'(hard {int(va_hard.sum())})  seed {SEED}  folds {FOLDS}', flush=True)

    t0 = time.perf_counter()
    tr_base = build_base_features(tr_bank, {'embeddings': tr_emb})
    va_base = build_base_features(va_bank, va_arrays)
    print(f'features built in {time.perf_counter() - t0:.1f}s  '
          f'tr {tr_base.shape}  va {va_base.shape}', flush=True)
    return {
        'tr_bank': tr_bank, 'tr_emb': tr_emb, 'tr_labels': tr_labels,
        'tr_base': tr_base, 'va_bank': va_bank, 'va_emb': va_arrays['embeddings'],
        'va_labels': va_labels, 'va_base': va_base, 'va_hard': va_hard,
    }


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def run_vote(data):
    tr_picks = baseline_vote(data['tr_bank'])
    va_picks = np.asarray(baseline_vote(data['va_bank']))
    val_full = accuracy(va_picks, data['va_labels'])
    val_hard, lo, hi = bootstrap_ci(va_picks[data['va_hard']],
                                    data['va_labels'][data['va_hard']])
    return {
        'estimator': 'majority vote (training-free)',
        'n_features': 0,
        'oof_hard': accuracy(tr_picks, data['tr_labels']),
        'val_full': val_full, 'val_hard': val_hard,
        'val_hard_ci': [lo, hi], 'fit_seconds': 0.0,
        'seed': SEED, 'folds': FOLDS,
        'numpy': np.__version__, 'python': platform.python_version(),
    }


def run_learned(name, spec, data):
    cols = spec['cols']
    use_d = spec.get('use_d', False)
    estimator = spec.get('estimator', 'linear')

    def assemble(base, emb):
        parts = [base[:, :, cols]] if cols else []
        if use_d:
            e = np.asarray(emb, dtype=np.float64)
            parts.append(e - e.mean(axis=1, keepdims=True))
        if not parts:
            raise ValueError('arm with no evidence')
        return np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]

    Xtr = assemble(data['tr_base'], data['tr_emb'])
    Xva = assemble(data['va_base'], data['va_emb'])
    reg = np.concatenate([np.full(len(cols), L2_HAND)] +
                         ([np.full(1024, L2_EMB)] if use_d else []))

    n = len(Xtr)
    rng = np.random.default_rng(SEED)
    fold_ids = np.array_split(rng.permutation(n), FOLDS)
    oof = np.zeros((n, 4))
    t0 = time.perf_counter()
    for k, ev in enumerate(fold_ids):
        fi = np.setdiff1d(np.arange(n), ev)
        if estimator == 'linear':
            m = fit_linear_lbfgs(Xtr[fi], data['tr_labels'][fi],
                                 l2_vector=reg, seed=SEED + 100 * k)
        else:
            m = fit_mlp(Xtr[fi], data['tr_labels'][fi], seed=SEED + 100 * k)
        oof[ev] = scores_from_model(Xtr[ev], m)
        print(f'    fold {k + 1}/{FOLDS} ({time.perf_counter() - t0:.0f}s)',
              flush=True)
    oof_acc = accuracy(oof.argmax(axis=1), data['tr_labels'])

    if estimator == 'linear':
        m = fit_linear_lbfgs(Xtr, data['tr_labels'], l2_vector=reg, seed=SEED)
    else:
        m = fit_mlp(Xtr, data['tr_labels'], seed=SEED)
    va_scores = scores_from_model(Xva, m)
    picks = va_scores.argmax(axis=1)
    val_full = accuracy(picks, data['va_labels'])
    val_hard, lo, hi = bootstrap_ci(picks[data['va_hard']],
                                    data['va_labels'][data['va_hard']])
    return {
        'estimator': estimator,
        'n_features': int(Xtr.shape[-1]),
        'oof_hard': oof_acc,
        'val_full': val_full, 'val_hard': val_hard,
        'val_hard_ci': [lo, hi],
        'fit_seconds': time.perf_counter() - t0,
        'seed': SEED, 'folds': FOLDS,
        'numpy': np.__version__, 'python': platform.python_version(),
    }


# ---------------------------------------------------------------------------
# frozen-weight inspection (RQ2)
# ---------------------------------------------------------------------------

def weight_report():
    """Where the frozen scorer's decision mass sits, by evidence family."""
    with np.load(BUNDLE / 'model.npz', allow_pickle=False) as f:
        w = np.asarray(f['weights'])
    cfg = json.loads((BUNDLE / 'config.json').read_text())
    names = cfg['feature_names']
    assert len(names) == len(w)

    spans, pos = {}, 0
    for g in cfg['groups']:
        wd = len(_GROUP_NAMES[g])
        spans[g] = (pos, pos + wd)
        pos += wd
    spans['D'] = (pos, len(w))

    def cols_of(groups, z_sources=()):
        idx = []
        for g in groups:
            idx += list(range(*spans[g]))
        if z_sources:
            z0 = spans['Z'][0]
            for g in z_sources:
                idx += [z0 + i for i in range(*_ZSPANS[g])]
        return idx

    families = {
        'votes (A)': cols_of('A'),
        'style (C,R,z_C)': cols_of('CR', ('C',)),
        'question (Q,z_Q)': cols_of('Q', ('Q',)),
        'verification (V,G,z_V,z_G)': cols_of('VG', ('V', 'G')),
    }
    hand_mass = float(np.abs(w[:spans['D'][0]]).sum())
    lines = ['frozen-weight |w| mass by evidence family (L1, standardised):']
    for fam, idx in families.items():
        mass = float(np.abs(w[idx]).sum())
        lines.append(f'  {fam:32s} {mass:8.3f}  '
                     f'{100 * mass / hand_mass:5.1f}% of hand mass')
    lines.append(f'  {"dense block D":32s} {float(np.abs(w[spans["D"][0]:]).sum()):8.3f}  '
                 f'(separate block, own regularisation)')

    top = np.argsort(-np.abs(w[:spans['D'][0]]))[:15]
    lines.append('top-15 hand weights:')
    for i in top:
        lines.append(f'  {names[i]:24s} {w[i]:+.3f}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def merge_results(results):
    all_res = (json.loads(RESULTS.read_text())
               if RESULTS.exists() else {})
    all_res.update(results)
    RESULTS.write_text(json.dumps(all_res, indent=2) + '\n')
    exp = json.loads(EXPERIMENTS.read_text())
    for name, res in results.items():
        exp[f'ablf_{name}'] = res
    EXPERIMENTS.write_text(json.dumps(exp, indent=2) + '\n')


def check_frozen(full_res):
    cfg = json.loads((BUNDLE / 'config.json').read_text())
    pairs = [('oof_hard', full_res['oof_hard'], cfg['oof_hard_accuracy']),
             ('val_full', full_res['val_full'], cfg['val_accuracy']),
             ('val_hard', full_res['val_hard'], cfg['val_hard_accuracy'])]
    lines = []
    ok = True
    for key, got, want in pairs:
        d = abs(got - want)
        ok &= d < 1e-6
        lines.append(f'  {key}: arm {got:.6f} vs frozen {want:.6f} '
                     f'(delta {d:.2e})')
    print(('frozen reproduction: PASS\n' if ok else 'frozen reproduction: MISMATCH\n')
          + '\n'.join(lines), flush=True)
    return ('PASS' if ok else 'MISMATCH'), '\n'.join(lines)


def write_md(frozen_check):
    res = json.loads(RESULTS.read_text())
    exp = json.loads(EXPERIMENTS.read_text())
    size = sum(f.stat().st_size for f in BUNDLE.iterdir() if f.is_file())
    gbm = exp.get('postfreeze_gbm_lambdarank', {}).get('oof_hard')

    out = [
        '# MP2 report ablations (frozen protocol, seed 7506)',
        '',
        'All arms share the frozen bundle\'s protocol: hard train subset',
        f'(n=2114), grouped {FOLDS}-fold OOF, L-BFGS two-start linear fits,',
        f'per-block L2 (hand {L2_HAND} / embedding {L2_EMB}), seed {SEED}.',
        'Reporting is train-hard OOF + validation only; the test bank is',
        'never read, nothing was adopted, and the frozen bundle is unchanged.',
        '',
        '| arm | evidence | F | OOF-hard | val-full | val-hard | 95% CI | fit s |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for name in ARM_ORDER:
        if name not in res:
            continue
        r = res[name]
        lo, hi = r['val_hard_ci']
        out.append(
            f"| {name} | {EVIDENCE[name]} | {r['n_features']} "
            f"| {100 * r['oof_hard']:.2f}% | {100 * r['val_full']:.2f}% "
            f"| {100 * r['val_hard']:.2f}% "
            f"| [{100 * lo:.2f}, {100 * hi:.2f}] | {r['fit_seconds']:.0f} |")
    out += [
        '',
        f'Frozen-bundle reproduction by the `full` arm: **{frozen_check[0]}**',
        '(`' + frozen_check[1].replace('\n', '`<br>`') + '`)',
        '',
        '## RQ2 — where the frozen decision mass sits',
        '',
        '```', weight_report(), '```',
        '',
        '## RQ5 — resources',
        '',
        f'* majority vote: 0 learned parameters, 0 B of assets.',
        f'* frozen linear: {size:,} B of inference assets (limit 16 MiB); '
        f'fit in ~47 s on this machine (train.py run).',
        '* MLP on the same features: fit_seconds in the table; asset would '
        'grow by the two dense layers but stays far under the limit.',
        (f'* GBDT probe (recorded, not re-run): LambdaRank LightGBM on the '
         f'pairwise view peaked at {100 * gbm:.1f}% OOF-hard — below linear.'
         if gbm else ''),
        '',
        '## Arm-to-question mapping',
        '',
        '* **RQ1 (does reading the reasoning help):** vote/A/AQ use only the '
        'parsed answers and the question; `hand` adds reading the solution '
        'text (surface + verification); `embD` is the supplied dense '
        'representation alone; `full` is the frozen combination.',
        '* **RQ2 (what makes a wrong answer convincing):** the '
        '`AQ -> surface` step prices pure length/format/style; '
        '`surface -> hand` prices exact-arithmetic verification and '
        'grounding on top of style; the weight-mass table shows what the '
        'frozen model actually relies on.',
        '* **RQ5 (when does more computation pay off):** vote vs `full` '
        '(linear) vs `mlp_full`, plus the recorded GBDT probe.',
    ]
    (RUNS / 'ablation_final.md').write_text('\n'.join(out) + '\n')
    print(f'wrote {RUNS / "ablation_final.md"}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', default='vote,A,AQ,surface,hand,embD,full')
    args = ap.parse_args()

    data = load_data()
    results = {}
    for name in [a.strip() for a in args.arms.split(',')]:
        print(f'--- arm {name} ---', flush=True)
        spec = ARMS[name]
        res = run_vote(data) if spec is None else run_learned(name, spec, data)
        results[name] = res
        lo, hi = res['val_hard_ci']
        print(f'  {name}: F={res["n_features"]} '
              f'oof={100 * res["oof_hard"]:.2f}% '
              f'val_full={100 * res["val_full"]:.2f}% '
              f'val_hard={100 * res["val_hard"]:.2f}% '
              f'[{100 * lo:.2f},{100 * hi:.2f}] ({res["fit_seconds"]:.1f}s)',
              flush=True)

    merge_results(results)
    frozen_check = ('n/a', 'full arm not run')
    if 'full' in results:
        frozen_check = check_frozen(results['full'])
    write_md(frozen_check)


if __name__ == '__main__':
    main()
