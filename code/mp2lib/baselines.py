"""Data loading, label construction and baselines for development."""

from collections import Counter

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl


def load_split(split):
    """Load bank, arrays and gold-match labels for a provided split.

    Only called for train/validation during development. Test references
    are never loaded anywhere in mp2lib.
    """
    bank, arrays = load_inputs(split)
    refs = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    gold = [parse_answer(r['answer']) for r in refs]
    labels = np.array([
        [parse_answer(c['text']) is not None and parse_answer(c['text']) == g
         for c in row['candidates']]
        for row, g in zip(bank, gold)
    ], dtype=bool)
    return bank, arrays, labels


def hard_mask(bank):
    """True for questions with >= 2 distinct parsed answers (test-like).

    Test groups all satisfy this by construction; train/validation contain
    an easier mix, so development decisions should focus on this subset.
    """
    m = np.zeros(len(bank), dtype=bool)
    for i, row in enumerate(bank):
        parsed = {parse_answer(c['text']) for c in row['candidates']}
        parsed.discard(None)
        m[i] = len(parsed) >= 2
    return m


def accuracy(picks, labels):
    """Selection accuracy for a list/array of candidate indices."""
    picks = np.asarray(picks, dtype=np.int64)
    return float(labels[np.arange(len(labels)), picks].mean())


def _vote_groups(row):
    """parsed answer -> list of candidate indices."""
    groups = {}
    for i, c in enumerate(row['candidates']):
        p = parse_answer(c['text'])
        if p is not None:
            groups.setdefault(p, []).append(i)
    return groups


def baseline_random(bank, seed=0):
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, 4)) for _ in bank]


def baseline_fixed(bank, index=0):
    return [index] * len(bank)


def baseline_vote(bank, tiebreak='min_index'):
    """Majority vote over parsed answers with a configurable tiebreak.

    tiebreak:
      min_index  — lowest candidate index among top-voted answers
      max_index  — highest candidate index among top-voted answers
      shortest   — shortest text among top-voted candidates
      longest    — longest text among top-voted candidates
      centroid   — highest cosine similarity to the group embedding centroid
    """
    picks = []
    for qi, row in enumerate(bank):
        groups = _vote_groups(row)
        if not groups:
            picks.append(0)
            continue
        counts = {a: len(v) for a, v in groups.items()}
        top = max(counts.values())
        candidates = [i for ans, v in groups.items() if counts[ans] == top for i in v]
        if tiebreak == 'min_index':
            picks.append(min(candidates))
        elif tiebreak == 'max_index':
            picks.append(max(candidates))
        elif tiebreak == 'shortest':
            picks.append(min(candidates, key=lambda i: len(row['candidates'][i]['text'])))
        elif tiebreak == 'longest':
            picks.append(max(candidates, key=lambda i: len(row['candidates'][i]['text'])))
        else:
            raise ValueError(f'unknown tiebreak {tiebreak}')
    return picks


def baseline_vote_centroid(bank, arrays, tiebreak='shortest'):
    """Majority vote; ties broken by embedding closeness to group centroid."""
    picks = []
    emb = arrays['embeddings']
    for qi, row in enumerate(bank):
        groups = _vote_groups(row)
        if not groups:
            picks.append(0)
            continue
        counts = {a: len(v) for a, v in groups.items()}
        top = max(counts.values())
        candidates = [i for ans, v in groups.items() if counts[ans] == top for i in v]
        if len(candidates) == 1:
            picks.append(candidates[0])
            continue
        g = emb[qi]
        centroid = g.mean(axis=0)
        sims = {
            i: float(g[i] @ centroid / max(np.linalg.norm(g[i]) * np.linalg.norm(centroid), 1e-12))
            for i in candidates
        }
        best = max(sims.values())
        tied = [i for i in candidates if sims[i] == best]
        if len(tied) > 1 and tiebreak == 'shortest':
            picks.append(min(tied, key=lambda i: len(row['candidates'][i]['text'])))
        else:
            picks.append(tied[0])
    return picks