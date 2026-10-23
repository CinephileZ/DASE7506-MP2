"""Error analysis for the current best scorer (development only).

Breaks validation accuracy down by vote structure, compares the learned
scorer with majority voting, and reports calibration of the group-softmax
confidence.  Used to write the report's failure analysis and RQ4.
"""

import argparse

import numpy as np

from mp2lib.baselines import baseline_vote, accuracy
from experiments import ALL, load_data, run_experiment, bootstrap_ci
from dev_eval import group_stats


def bucket_report(name, picks, labels, hard, bucket, scores=None):
    print(f'\n== {name}')
    for label, mask in bucket.items():
        if mask.sum() == 0:
            continue
        acc, lo, hi = bootstrap_ci(np.asarray(picks)[mask], labels[mask])
        line = (f'  {label:28s} n={int(mask.sum()):4d}  acc={100*acc:6.2f}% '
                f'[{100*lo:.2f},{100*hi:.2f}]')
        if scores is not None and mask.sum():
            conf = scores[mask].max(axis=1).mean()
            line += f'  mean_conf={conf:.3f}'
        print(line)


def permutation_importance(data, groups, n_pca, kind, n_models, seed=0,
                           n_repeat=5):
    """Drop in hard-subset validation accuracy when a feature block is shuffled
    across the four candidates within each question.

    Shuffling *within* a question preserves every group-level constant and the
    marginal distribution of the block, so the drop isolates how much the
    decision actually relies on that block.
    """
    from mp2lib.features import _GROUP_NAMES, _select_groups
    from mp2lib.model import fit_linear, fit_linear_ensemble, scores_from_model
    from mp2lib.pca import fit_pca
    from experiments import base_features, _assemble

    tr_bank, tr_arrays, tr_labels, tr_hard = data['train']
    va_bank, va_arrays, va_labels, va_hard = data['validation']
    tri = np.flatnonzero(tr_hard)
    tr_base, tr_lab = base_features('train', tr_bank, tr_arrays)[tri], tr_labels[tri]
    va_base = base_features('validation', va_bank, va_arrays)
    tr_emb, va_emb = tr_arrays['embeddings'][tri], va_arrays['embeddings']

    pca = fit_pca(tr_emb, n_pca) if n_pca else None
    fit = fit_linear if kind == 'linear' else fit_linear_ensemble
    kw = {'l2': 1e-3, 'lr': 0.05, 'epochs': 3000}
    if kind == 'ensemble':
        kw['n_models'] = n_models
    model = fit(_assemble(tr_base, tr_emb, groups, pca), tr_lab, seed=seed, **kw)

    Xv = _assemble(va_base, va_emb, groups, pca)
    base_picks = scores_from_model(Xv, model).argmax(axis=1)[va_hard]
    va_lab_h = va_labels[va_hard]
    base_acc = accuracy(base_picks, va_lab_h)
    print(f'baseline val-hard accuracy: {100*base_acc:.2f}%')

    # locate column ranges of each block inside the assembled matrix
    blocks = {}
    pos = 0
    for g in groups:
        w = len(_GROUP_NAMES[g])
        blocks[g] = (pos, pos + w)
        pos += w
    if n_pca:
        blocks['P'] = (pos, pos + n_pca)

    rng = np.random.default_rng(seed)
    print(f'\n{"block":6s} {"dims":>5s} {"acc after shuffle":>18s} {"drop (pp)":>10s}')
    for name, (a, b) in blocks.items():
        drops = []
        for _ in range(n_repeat):
            Xs = Xv.copy()
            for n in range(Xs.shape[0]):
                Xs[n, :, a:b] = Xv[n, rng.permutation(4), a:b]
            acc_s = accuracy(scores_from_model(Xs, model).argmax(axis=1)[va_hard],
                             va_lab_h)
            drops.append(base_acc - acc_s)
        print(f'{name:6s} {b - a:5d} {100*(base_acc - np.mean(drops)):17.2f}% '
              f'{100*np.mean(drops):9.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pca', type=int, default=256)
    ap.add_argument('--kind', default='linear')
    ap.add_argument('--n-models', type=int, default=3)
    ap.add_argument('--perm', action='store_true')
    args = ap.parse_args()

    data = load_data()

    if args.perm:
        permutation_importance(data, ALL, args.pca, args.kind, args.n_models)
        return
    va_bank, va_arrays, va_labels, va_hard = data['validation']
    distinct, top_vote, valid = group_stats(va_bank)

    kw = {'n_models': args.n_models}
    if args.kind == 'mlp':
        kw = {'hidden': 96, 'n_models': args.n_models}
    r = run_experiment(data, ALL, args.pca, args.kind, seed=0, model_kw=kw)
    picks = r['va_picks']
    scores = r['va_scores']

    buckets = {
        'all validation': np.ones(len(va_bank), bool),
        'hard (>=2 distinct)': va_hard,
        'hard: strict top>=3': va_hard & (top_vote >= 3),
        'hard: tied top<=2': va_hard & (top_vote <= 2),
        'hard: distinct=2': va_hard & (distinct == 2),
        'hard: distinct=3': va_hard & (distinct == 3),
        'hard: distinct=4': va_hard & (distinct == 4),
        'hard: has unparsed': va_hard & (valid < 4),
        'easy (<2 distinct)': ~va_hard,
    }

    vote = np.asarray(baseline_vote(va_bank, 'shortest'))
    bucket_report(f'{args.kind} pca={args.pca} (model)', picks, va_labels,
                  va_hard, buckets, scores)
    bucket_report('majority vote (shortest)', vote, va_labels, va_hard, buckets)

    # agreement / disagreement between model and vote on hard groups
    print('\n== model vs vote on hard groups')
    h = va_hard
    agree = h & (picks == vote)
    dis = h & (picks != vote)
    print(f'  agree   n={int(agree.sum()):4d}  model={100*accuracy(picks[agree], va_labels[agree]):6.2f}%  '
          f'vote={100*accuracy(vote[agree], va_labels[agree]):6.2f}%')
    print(f'  disagree n={int(dis.sum()):4d}  model={100*accuracy(picks[dis], va_labels[dis]):6.2f}%  '
          f'vote={100*accuracy(vote[dis], va_labels[dis]):6.2f}%')

    # calibration (RQ4)
    print('\n== confidence calibration on hard groups (softmax of scores)')
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    conf = p.max(axis=1)
    correct = va_labels[np.arange(len(va_labels)), picks]
    for lo, hi in [(0.25, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        m = h & (conf >= lo) & (conf < hi)
        if m.sum():
            print(f'  conf[{lo:.2f},{hi:.2f}) n={int(m.sum()):4d} '
                  f'acc={100*correct[m].mean():6.2f}%  mean_conf={conf[m].mean():.3f}')
    print(f'  ECE(hard) = {_ece(conf[h], correct[h]):.4f}')


def _ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


if __name__ == '__main__':
    main()
