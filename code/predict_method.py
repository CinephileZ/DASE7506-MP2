"""MP2 candidate selection — linear softmax-CE scorer.

Implements ``run(bank, arrays, checkpoint)`` which is called by
``submission.py -> predict()`` during the official evaluation pipeline.

Feature groups used (by ablation results):
  A — Answer-agreement vote structure (7 dims)
  C — Text surface signals (17 dims)
  B (embedding) was found harmful during validation and omitted.
"""

import json
import re
from pathlib import Path

import numpy as np


# --- regex helpers ----------------------------------------------------------

_NUMBER_RE = re.compile(r'[-+]?(?:\d[\d,]*(?:\.\d+)?)')
_BOLD_RE = re.compile(r'\*\*')
_SENTENCE_RE = re.compile(r'[.!?\n]+')
_LATEX_RE = re.compile(r'\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)', re.DOTALL)

_HESITATION_WORDS = [
    'wait', 'check', 'actually', 'hmm', 'maybe', 'perhaps',
    'let me try', 'i think', 'probably', 'might be', 'surely',
]
_CONCLUSION_WORDS = [
    'thus', 'therefore', 'hence', 'consequently',
    'finally', 'the answer is', 'so the answer',
]


# --- answer parsing (mirrors common.parse_answer) ---------------------------

def _parse_answer(text):
    """Extract final numeric answer from a candidate solution text."""
    from fractions import Fraction
    t = str(text)
    tail = t.rsplit('####', 1)[1] if '####' in t else None
    if tail is None:
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', t)
        tail = boxed[-1] if boxed else ''
    m = re.search(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', tail)
    if not m:
        return None
    try:
        return str(Fraction(m.group().replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return None


# --- Group-A: answer-agreement features (7 dim) -----------------------------

def _build_group_a(bank):
    """Return per-candidate answer-agreement matrix [N, 4, 7]."""
    N = len(bank)
    out = np.zeros((N, 4, 7), dtype=np.float64)
    for g in range(N):
        texts = [c['text'] for c in bank[g]['candidates']]
        parsed = [_parse_answer(t) for t in texts]
        lengths = [len(t) for t in texts]

        vote_map = {}
        for i, p in enumerate(parsed):
            if p is not None:
                vote_map.setdefault(p, []).append(i)
        valid_count = sum(1 for p in parsed if p is not None)
        max_vote = max((len(v) for v in vote_map.values()), default=0)
        unique_answers = len(vote_map)

        # Shortest-text tiebreak among tied top-answer candidates
        tie_shortest = set()
        tied_winners = [a for a, v in vote_map.items()
                        if len(v) == max_vote and max_vote > 1]
        if tied_winners:
            ti = [i for ans in tied_winners for i in vote_map[ans]]
            sl = min(lengths[i] for i in ti)
            tie_shortest = {i for i in ti if lengths[i] == sl}

        for c in range(4):
            p = parsed[c]
            vc = len(vote_map[p]) if p is not None else 0
            is_max = p is not None and vc == max_vote
            idx = 0
            out[g, c, idx] = float(p is not None); idx += 1             # parsed_ok
            out[g, c, idx] = float(vc); idx += 1                        # vote_count
            out[g, c, idx] = vc / max(valid_count, 1); idx += 1         # vote_ratio
            out[g, c, idx] = float(vc >= 3); idx += 1                   # strict_majority
            out[g, c, idx] = float(is_max); idx += 1                    # is_tied_winner
            out[g, c, idx] = float(unique_answers); idx += 1            # group_unique
            out[g, c, idx] = float(c in tie_shortest and is_max)       # tie_shortest
    return out


# --- Group-C: text surface features (17 dim) ----------------------------------

_TEXT_FEAT_NAMES = [
    'length', 'word_count', 'sentence_count', 'num_count',
    'op_count', 'latex_count', 'has_boxed', 'has_hashhash',
    'bold_count', 'step_keyword_count', 'numeric_density',
    'avg_sentence_len', 'ops_per_sentence', 'hesitation_hits',
    'conclusion_hits', 'int_num_count', 'float_val_count',
]


def _extract_one_text(text):
    stripped = text.strip()
    words = stripped.split()
    sentences = max(1, len(_SENTENCE_RE.findall(stripped)))
    nums = _NUMBER_RE.findall(stripped)
    ops = re.findall(r'[+\-*/%]', stripped)
    latex = _LATEX_RE.findall(stripped)
    lower = stripped.lower()

    fv, iv = [], []
    for ns in nums:
        try:
            v = float(ns.replace(',', ''))
            fv.append(v)
            if '/' not in ns and '.' not in ns:
                iv.append(int(v))
        except (ValueError, ZeroDivisionError):
            pass

    return {
        'length': len(text),
        'word_count': len(words),
        'sentence_count': sentences,
        'num_count': len(nums),
        'op_count': len(ops),
        'latex_count': len(latex),
        'has_boxed': float('\\boxed{' in text or '\\\\boxed{' in text),
        'has_hashhash': float('####' in text),
        'bold_count': len(_BOLD_RE.findall(stripped)),
        'step_keyword_count': sum(1 for kw in ['thus','therefore','hence',
                                                'consequently','finally',
                                                'lastly','first,','second,',
                                                'next,','then ','step']
                                  if kw in lower),
        'numeric_density': len(nums) / max(len(words), 1),
        'avg_sentence_len': len(words) / max(sentences, 1),
        'ops_per_sentence': len(ops) / max(sentences, 1),
        'hesitation_hits': sum(1 for w in _HESITATION_WORDS if w in lower),
        'conclusion_hits': sum(1 for w in _CONCLUSION_WORDS if w in lower),
        'int_num_count': len(iv),
        'float_val_count': len(fv) - len(iv),
    }


def _build_group_c(bank):
    """Return per-candidate text features [N, 4, 17]."""
    n_names = len(_TEXT_FEAT_NAMES)
    mat = np.zeros((len(bank), 4, n_names), dtype=np.float64)
    for g in range(len(bank)):
        row = bank[g]
        for ci in range(4):
            feat = _extract_one_text(row['candidates'][ci]['text'])
            for j, fn in enumerate(_TEXT_FEAT_NAMES):
                mat[g, ci, j] = feat[fn]
    return mat


# --- Combined feature matrix -------------------------------------------------

def build_features(bank, arrays):
    """Build full feature matrix [N, 4, F_total].

    Uses A+C only; embedding features (B) were dropped by ablation.
    """
    return np.concatenate([_build_group_a(bank), _build_group_c(bank)],
                          axis=-1).astype(np.float64)


# --- scoring -----------------------------------------------------------------

def _apply_scores(features, result):
    """Standardize features using stored stats, then dot product."""
    mean = result['mean']
    std = np.maximum(result['std'], 1e-6)
    flat = (features.reshape(-1, features.shape[-1]).astype(np.float64) - mean) / std
    return (flat @ result['weights'] + result['bias']).reshape(features.shape[:2])


# --- fallback: majority vote with longest-text tiebreak --------------------

def _vote_fallback(bank):
    picks = []
    for row in bank:
        parsed = [_parse_answer(c['text']) for c in row['candidates']]
        groups = {}
        for i, p in enumerate(parsed):
            if p is not None:
                groups.setdefault(p, []).append(i)
        if not groups:
            picks.append(0)
            continue
        top = max(len(v) for v in groups.values())
        wins = [i for ans, v in groups.items() if len(v) == top for i in v]
        if len(wins) == 1:
            picks.append(wins[0])
        else:
            picks.append(max(wins,
                            key=lambda i: len(row['candidates'][i]['text'])))
    return picks


# --- public API --------------------------------------------------------------

def run(bank, arrays, checkpoint):
    """Select one candidate index per question.

    Args:
        bank       : list[dict] — questions with 4 candidates each
        arrays     : dict with 'embeddings' key [N, 4, 1024]
        checkpoint : pathlib.Path to bundle dir or None

    Returns:
        list[int] — exactly one int in {0, 1, 2, 3} per question,
                    ordered matching *bank*.
    """
    cp = Path(checkpoint) if checkpoint is not None else None
    mf = (cp / 'model.npz') if cp else Path('model.npz')
    cf = (cp / 'config.json') if cp else Path('config.json')

    if not mf.exists() or not cf.exists():
        return _vote_fallback(bank)

    data = np.load(mf, allow_pickle=False)
    cfg = json.loads(cf.read_text())

    weights = data['weights']
    bias = float(data['bias'])
    mean = data['mean']
    std = data['std']

    result = {
        'weights': weights, 'bias': bias,
        'mean': mean, 'std': std,
    }

    features = build_features(bank, arrays)
    scores = _apply_scores(features, result)
    picks = scores.argmax(axis=1)
    return [int(p) for p in picks]
