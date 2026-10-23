"""Development harness: EDA, model comparison and bootstrap CIs.

Not part of the submission pipeline.  All development decisions are made on
train (OOF CV) + validation; test references are never loaded here.

Usage (from code/):
    python dev_eval.py eda
    python dev_eval.py ladder
"""

import argparse
import numpy as np

from mp2lib.baselines import (
    load_split, hard_mask, accuracy,
    baseline_vote, baseline_vote_centroid,
)
from common import parse_answer


# --- evaluation helpers -----------------------------------------------------

def bootstrap_ci(picks, labels, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for selection accuracy."""
    picks = np.asarray(picks, dtype=np.int64)
    labels = np.asarray(labels, dtype=bool)
    correct = labels[np.arange(len(labels)), picks]
    n = len(correct)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    accs = correct[idx].mean(axis=1)
    lo, hi = np.percentile(accs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(correct.mean()), float(lo), float(hi)


def group_stats(bank):
    """Per-group vote-structure statistics."""
    n = len(bank)
    distinct = np.zeros(n, dtype=np.int64)
    top_vote = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=np.int64)
    for g, row in enumerate(bank):
        parsed = [parse_answer(c['text']) for c in row['candidates']]
        counts = {}
        for p in parsed:
            if p is not None:
                counts[p] = counts.get(p, 0) + 1
        distinct[g] = len(counts)
        top_vote[g] = max(counts.values()) if counts else 0
        valid[g] = sum(1 for p in parsed if p is not None)
    return distinct, top_vote, valid


def eda():
    for split in ('train', 'validation'):
        bank, arrays, labels = load_split(split)
        hard = hard_mask(bank)
        distinct, top_vote, valid = group_stats(bank)
        print(f'\n===== {split}: n={len(bank)}  hard={int(hard.sum())} =====')

        picks_vote = np.asarray(baseline_vote(bank, 'shortest'))
        picks_cent = np.asarray(baseline_vote_centroid(bank, arrays))

        for name, m in (('ALL', np.ones(len(bank), bool)), ('HARD', hard)):
            print(f'-- {name} subset (n={int(m.sum())})')
            print(f'   vote/shortest  {100*accuracy(picks_vote[m], labels[m]):6.2f}%')
            print(f'   vote/centroid  {100*accuracy(picks_cent[m], labels[m]):6.2f}%')

        # breakdown of hard groups by (distinct answers, top vote size)
        print('-- HARD breakdown: distinct / top_vote -> n, vote_acc, correct_slots')
        for d in range(2, 5):
            for t in range(1, 5):
                m = hard & (distinct == d) & (top_vote == t)
                if m.sum() == 0:
                    continue
                acc = accuracy(picks_vote[m], labels[m])
                slots = labels[m].sum(axis=1).mean()
                print(f'   d={d} t={t}: n={int(m.sum()):5d}  vote_acc={100*acc:6.2f}%  '
                      f'mean_correct_slots={slots:.2f}')

        # tie structure among hard groups
        print('-- HARD tie structure')
        tied = hard & (top_vote <= 2)
        strict = hard & (top_vote >= 3)
        for name, m in (('tied(top<=2)', tied), ('strict(top>=3)', strict)):
            if m.sum():
                print(f'   {name:15s} n={int(m.sum()):5d} vote_acc={100*accuracy(picks_vote[m], labels[m]):6.2f}%')


def ladder():
    for split in ('train', 'validation'):
        bank, arrays, labels = load_split(split)
        hard = hard_mask(bank)
        print(f'\n===== {split} =====')
        methods = {
            'random': np.random.default_rng(0).integers(0, 4, len(bank)),
            'fixed0': np.zeros(len(bank), dtype=int),
            'vote/min_index': baseline_vote(bank, 'min_index'),
            'vote/shortest': baseline_vote(bank, 'shortest'),
            'vote/longest': baseline_vote(bank, 'longest'),
            'vote/centroid': baseline_vote_centroid(bank, arrays),
        }
        for name, picks in methods.items():
            picks = np.asarray(picks, dtype=np.int64)
            full = np.ones(len(bank), bool)
            a_f, lo_f, hi_f = bootstrap_ci(picks[full], labels[full])
            a_h, lo_h, hi_h = bootstrap_ci(picks[hard], labels[hard])
            print(f'  {name:16s} full={100*a_f:6.2f}% [{100*lo_f:.2f},{100*hi_f:.2f}]  '
                  f'hard={100*a_h:6.2f}% [{100*lo_h:.2f},{100*hi_h:.2f}]')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('eda', 'ladder'))
    args = ap.parse_args()
    {'eda': eda, 'ladder': ladder}[args.cmd]()
