"""Phase 1: baseline ladder report on train/validation, full and hard subsets."""

import numpy as np

from mp2lib.baselines import (
    load_split, hard_mask, accuracy,
    baseline_random, baseline_fixed, baseline_vote, baseline_vote_centroid,
)


def report(name, picks, labels, fold_name):
    print(f'  {name:<22s} {fold_name:<8s} {100*accuracy(picks, labels):6.2f}%')


def main():
    for split in ('train', 'validation'):
        bank, arrays, labels = load_split(split)
        hard = hard_mask(bank)
        print(f'== {split}: n={len(bank)}, hard={int(hard.sum())}, easy={int((~hard).sum())}')

        methods = {
            'random': baseline_random(bank, seed=0),
            'fixed index 0': baseline_fixed(bank, 0),
            'vote/min_index': baseline_vote(bank, 'min_index'),
            'vote/shortest': baseline_vote(bank, 'shortest'),
            'vote/longest': baseline_vote(bank, 'longest'),
            'vote/max_index': baseline_vote(bank, 'max_index'),
            'vote/centroid': baseline_vote_centroid(bank, arrays),
        }
        full = np.ones(len(bank), dtype=bool)
        for name, picks in methods.items():
            picks = np.asarray(picks, dtype=np.int64)
            report(name, picks[full], labels[full], 'full')
            report(name, picks[hard], labels[hard], 'hard')


if __name__ == '__main__':
    main()