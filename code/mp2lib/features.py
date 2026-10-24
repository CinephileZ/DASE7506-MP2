"""Feature extraction for MP2 candidate selection.

Design note
-----------
Scoring is *within-group*: a softmax (or any monotone) model over the four
candidate scores is invariant to adding a feature that is constant inside a
group.  Group-level constants (e.g. number of distinct answers) therefore
carry no decision information and are deliberately excluded; every feature
below varies across the four candidates of a group.

Groups
------
  A  answer agreement   — vote structure over parsed final answers
  B  embedding geometry — raw norms, distance to group centroid, pairwise
                          cosine row statistics
  S  answer-cluster coherence — how close a candidate sits to the centroid of
                          the candidates that share its answer, versus the
                          centroid of those that do not
  C  text surface       — absolute length / format / density signals
  R  relative text      — the same signals standardised inside the group
  Q  question interaction — relation between the final answer and the question

Group P (PCA of the group-centred, L2-normalised embedding) is appended by
:func:`mp2lib.pca.PCAModel` and is not computed here.
"""

import re
from fractions import Fraction

import numpy as np

from common import parse_answer
from mp2lib.verify import audit

# --- regex constants --------------------------------------------------------

_NUMBER_RE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
_OPERATOR_RE = re.compile(r'[+\-*/%]')
_LATEX_RE = re.compile(r'\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)', re.DOTALL)
_SENTENCE_RE = re.compile(r'[.!?\n]+')
_BOLD_RE = re.compile(r'\*\*')

_HESITATION_WORDS = [
    'wait', 'check', 'actually', 'hmm', 'maybe', 'perhaps',
    'let me try', 'i think', 'probably', 'might be', 'surely',
]
_CONCLUSION_WORDS = [
    'thus', 'therefore', 'hence', 'consequently',
    'finally', 'the answer is', 'so the answer',
]
_STEP_WORDS = ['thus', 'therefore', 'hence', 'consequently', 'finally',
               'lastly', 'first,', 'second,', 'next,', 'then ', 'step']

# --- feature name lists (stable ordering) -----------------------------------

_FEATURES_A = [
    'parsed_ok',          # final answer parses
    'vote_count',         # candidates in the group sharing this answer
    'vote_ratio',         # vote_count / number of parseable candidates
    'vote_frac4',         # vote_count / 4
    'is_top_vote',        # holds the (possibly tied) largest answer share
    'strict_majority',    # vote_count >= 3
    'vote_margin',        # vote_count - largest share held by a *different* answer
    'is_singleton',       # unique supporter while other answers exist
    'rank_by_vote',       # 0 = largest share (smallest index on ties)
    'tie_break_shortest', # among tied top answers, shortest text
    'value_position',     # where this answer sits in the group's sorted values
    'value_ratio_to_peak',# answer / largest absolute answer in the group
]

_FEATURES_B = [
    'norm_rel',           # L2 norm / group mean norm
    'dist_centroid',      # L2 distance to group centroid
    'dist_centroid_rel',  # distance / group mean distance
    'cos_centroid',       # cosine to group centroid
    'sim_row_min',        # pairwise cosine row statistics (diagonal excluded)
    'sim_row_max',
    'sim_row_mean',
    'sim_row_std',
]

_FEATURES_S = [
    'cos_same_cluster',   # cosine to centroid of same-answer candidates
    'cos_diff_cluster',   # cosine to centroid of different-answer candidates
    'cluster_margin',     # cos_same_cluster - cos_diff_cluster
    'has_same_support',   # at least one other candidate shares this answer
]

_FEATURES_C = [
    'length',
    'word_count',
    'sentence_count',
    'num_count',
    'op_count',
    'latex_count',
    'boxed_marker',
    'has_hashhash',
    'format_priority',     # 2 = boxed, 1 = ####, 0 = neither
    'bold_count',
    'step_keyword_count',
    'numeric_density',     # num_count / word_count
    'avg_sentence_len',    # word_count / sentence_count
    'ops_per_sentence',    # op_count / sentence_count
    'hesitation_hits',
    'conclusion_hits',
    'distinct_num_count',  # distinct numeric values in the text
    'eq_count',            # number of '=' signs
    'list_line_count',     # lines opening with a list marker
    'display_math_present',
    'tail_line_len',       # characters on the final line
    'line_break_rate',     # newlines / words
    'answer_integral',     # the final answer is a whole number
    'answer_below_zero',
]

_FEATURES_R = [
    'length_rel', 'word_count_rel', 'sentence_count_rel', 'num_count_rel',
    'op_count_rel', 'latex_count_rel', 'length_rank', 'num_count_rank',
]

_FEATURES_Q = [
    'answer_in_question',    # parsed answer appears among the question's numbers
    'answer_mentions',       # times the final answer value occurs in the text
    'last_num_is_answer',    # the final number equals the parsed answer
    'answer_after_last_op',  # the answer is the result of the final operation
]

_FEATURES_V = [
    'calc_stated',         # equations the reader could check
    'calc_holds',          # of those, ones that recompute exactly
    'calc_fails',          # of those, ones that do not
    'calc_holds_frac',
    'calc_has_failure',    # at least one stated step does not hold
    'calc_failure_frac',
    'answer_stated_hits',  # stated results equal to the candidate's answer
    'answer_stated_any',   # some stated result is the answer
    'answer_stated_last',  # the final stated result is the answer
    'calc_stated_rel',     # equations relative to the group mean
]

# Grounding: does the solution actually consume the question's given numbers,
# and is the final answer just a number restated from the question (a classic
# shortcut) or within its plausible range?
_FEATURES_G = [
    'ground_used_ratio',       # share of the question's numbers used by the trace
    'ground_missing_count',    # question numbers the trace never mentions
    'ground_answer_restated',  # the final answer is one of the given numbers
    'ground_answer_in_range',  # the answer stays within the question's numbers
    'ground_mentions',         # total occurrences of question numbers
    'ground_used_relative',    # used ratio relative to the group
]

# Within-group standardised twins of every numeric statistic above: when one
# candidate must be chosen, "the longest of the four" is a different question
# from "long", and the twin makes that comparison available to the scorer.
_Z_SOURCE = _FEATURES_C + _FEATURES_Q + _FEATURES_V + _FEATURES_G
_FEATURES_Z = [f'z_{name}' for name in _Z_SOURCE]

_BASE_GROUPS = ('A', 'B', 'S', 'C', 'R', 'Q', 'V', 'G', 'Z')

_GROUP_NAMES = {
    'A': _FEATURES_A, 'B': _FEATURES_B, 'S': _FEATURES_S,
    'C': _FEATURES_C, 'R': _FEATURES_R, 'Q': _FEATURES_Q,
    'V': _FEATURES_V, 'G': _FEATURES_G, 'Z': _FEATURES_Z,
}


def feature_names(groups=_BASE_GROUPS, n_pca=0):
    names = []
    for g in groups:
        names += list(_GROUP_NAMES[g])
    names += [f'pca_{k}' for k in range(n_pca)]
    return names


# ---------------------------------------------------------------------------
# per-candidate scalar extraction
# ---------------------------------------------------------------------------

def _nums_with_values(text):
    """Return (raw_tokens, float_values) for every number-like token."""
    toks = _NUMBER_RE.findall(text)
    vals = []
    for t in toks:
        try:
            if '/' in t:
                num, den = t.split('/')
                den_v = float(den)
                if den_v == 0:
                    vals.append(float('nan'))
                else:
                    vals.append(float(num) / den_v)
            else:
                vals.append(float(t.replace(',', '')))
        except ValueError:
            vals.append(float('nan'))
    return toks, vals


def _candidate_scalars(text, parsed=None):
    stripped = text.strip()
    words = stripped.split()
    sentences = max(1, len(_SENTENCE_RE.findall(stripped)))
    toks, vals = _nums_with_values(stripped)
    ops = _OPERATOR_RE.findall(stripped)
    latex = _LATEX_RE.findall(stripped)
    lower = stripped.lower()

    has_boxed = float('\\boxed{' in text)
    has_hashhash = float('####' in text)
    distinct = len({round(v, 6) for v in vals if np.isfinite(v)})
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    ans_int = ans_neg = 0.0
    if parsed is not None:
        try:
            fv = float(Fraction(parsed))
            ans_int = float(fv.is_integer())
            ans_neg = float(fv < 0)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass

    return {
        'length': float(len(text)),
        'word_count': float(len(words)),
        'sentence_count': float(sentences),
        'num_count': float(len(toks)),
        'op_count': float(len(ops)),
        'latex_count': float(len(latex)),
        'boxed_marker': has_boxed,
        'has_hashhash': has_hashhash,
        'format_priority': 2.0 * has_boxed + 1.0 * has_hashhash,
        'bold_count': float(len(_BOLD_RE.findall(stripped))),
        'step_keyword_count': float(sum(1 for kw in _STEP_WORDS if kw in lower)),
        'numeric_density': len(toks) / max(len(words), 1),
        'avg_sentence_len': len(words) / sentences,
        'ops_per_sentence': len(ops) / sentences,
        'hesitation_hits': float(sum(1 for w in _HESITATION_WORDS if w in lower)),
        'conclusion_hits': float(sum(1 for w in _CONCLUSION_WORDS if w in lower)),
        'distinct_num_count': float(distinct),
        'eq_count': float(stripped.count('=')),
        'list_line_count': float(sum(1 for ln in lines if ln.lstrip()[:1] in ('-', '*'))),
        'display_math_present': float('$$' in text),
        'tail_line_len': float(len(lines[-1]) if lines else 0),
        'line_break_rate': float(stripped.count('\n') / max(len(words), 1)),
        'answer_integral': ans_int,
        'answer_below_zero': ans_neg,
        '_vals': vals,
    }


def _group_relative_ranks(values):
    """Descending rank of each entry (0 = largest), ties broken by index."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0] * len(values)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


# ---------------------------------------------------------------------------
# group-level assembly
# ---------------------------------------------------------------------------

def _build_group(bank, arrays):
    """Return [N, 4, F_base] for groups A, B, S, C, R, Q."""
    n = len(bank)
    emb = np.asarray(arrays['embeddings'], dtype=np.float64)
    fA, fB, fS, fC, fR, fQ, fV, fG = [], [], [], [], [], [], [], []

    nA, nB, nS, nC, nR, nQ, nV, nG = (len(_GROUP_NAMES[g]) for g in
                                      ('A', 'B', 'S', 'C', 'R', 'Q', 'V', 'G'))

    A = np.zeros((n, 4, nA))
    B = np.zeros((n, 4, nB))
    S = np.zeros((n, 4, nS))
    C = np.zeros((n, 4, nC))
    R = np.zeros((n, 4, nR))
    Q = np.zeros((n, 4, nQ))
    V = np.zeros((n, 4, nV))
    G = np.zeros((n, 4, nG))

    for g in range(n):
        row = bank[g]
        texts = [c['text'] for c in row['candidates']]
        parsed = [parse_answer(t) for t in texts]
        sc = [_candidate_scalars(t, p) for t, p in zip(texts, parsed)]
        lengths = [s['length'] for s in sc]

        # --- A: vote structure
        vote_map = {}
        for i, p in enumerate(parsed):
            if p is not None:
                vote_map.setdefault(p, []).append(i)
        counts = {a: len(v) for a, v in vote_map.items()}
        valid = sum(1 for p in parsed if p is not None)
        max_vote = max(counts.values()) if counts else 0
        tied_top = [a for a, c in counts.items() if c == max_vote and max_vote > 1]
        tie_short = set()
        if tied_top:
            idx = [i for a in tied_top for i in vote_map[a]]
            smin = min(lengths[i] for i in idx)
            tie_short = {i for i in idx if lengths[i] == smin}

        votes = [counts.get(p, 0) if p is not None else 0 for p in parsed]
        ranks = _group_relative_ranks(votes)
        # answer-value ordering: some error modes land on a "total" or an
        # endpoint value, which this exposes without any text-level evidence
        fvals = []
        for p in parsed:
            try:
                fvals.append(float(Fraction(p)) if p is not None else None)
            except (ValueError, ZeroDivisionError):
                fvals.append(None)
        finite_vals = sorted({v for v in fvals if v is not None})
        max_abs = max((abs(v) for v in finite_vals), default=0.0)
        for c in range(4):
            p = parsed[c]
            vc = votes[c]
            other = max([cnt for a, cnt in counts.items() if a != p], default=0)
            fv = fvals[c]
            if fv is None or len(finite_vals) < 2:
                vrank, rel_max = 0.0, 0.0
            else:
                vrank = finite_vals.index(fv) / (len(finite_vals) - 1)
                rel_max = fv / max(max_abs, 1e-9)
            A[g, c] = [
                float(p is not None),
                float(vc),
                vc / max(valid, 1),
                vc / 4.0,
                float(p is not None and vc == max_vote),
                float(vc >= 3),
                float(vc - other),
                float(vc == 1 and len(counts) >= 2),
                float(ranks[c]),
                float(c in tie_short and p is not None and vc == max_vote),
                float(vrank),
                float(rel_max),
            ]

        # --- B: embedding geometry
        e = emb[g]
        norms = np.linalg.norm(e, axis=-1)
        centroid = e.mean(axis=0)
        dist = np.linalg.norm(e - centroid, axis=-1)
        denom_n = np.maximum(norms, 1e-12)
        cos_c = np.array([_cos(e[i], centroid) for i in range(4)])
        enorm = e / denom_n[:, None]
        sims = enorm @ enorm.T
        np.fill_diagonal(sims, np.nan)
        mean_norm = max(norms.mean(), 1e-12)
        mean_dist = max(dist.mean(), 1e-12)
        for c in range(4):
            B[g, c] = [
                norms[c] / mean_norm,
                dist[c],
                dist[c] / mean_dist,
                cos_c[c],
                float(np.nanmin(sims[c])),
                float(np.nanmax(sims[c])),
                float(np.nanmean(sims[c])),
                float(np.nanstd(sims[c])),
            ]

        # --- S: answer-cluster coherence
        for c in range(4):
            same = [j for j in range(4) if j != c and parsed[j] is not None
                    and parsed[j] == parsed[c]]
            diff = [j for j in range(4) if parsed[j] != parsed[c]]
            if c in same:
                same = [j for j in same if j != c]
            cs = _cos(e[c], e[same].mean(axis=0)) if same else 0.0
            cd = _cos(e[c], e[diff].mean(axis=0)) if diff else 0.0
            S[g, c] = [cs, cd, cs - cd, float(len(same) > 0)]

        # --- C: absolute text surface
        for c in range(4):
            s = sc[c]
            C[g, c] = [s[k] for k in _FEATURES_C]

        # --- R: within-group relative text
        series = {k: np.array([sc[c][k] for c in range(4)]) for k in
                  ('length', 'word_count', 'sentence_count', 'num_count',
                   'op_count', 'latex_count')}
        rel = {}
        for k, v in series.items():
            mu = v.mean()
            sd = max(v.std(), 1e-6)
            rel[k] = (v - mu) / sd
        rank_len = _group_relative_ranks(list(series['length']))
        rank_num = _group_relative_ranks(list(series['num_count']))
        for c in range(4):
            R[g, c] = [
                rel['length'][c], rel['word_count'][c], rel['sentence_count'][c],
                rel['num_count'][c], rel['op_count'][c], rel['latex_count'][c],
                float(rank_len[c]), float(rank_num[c]),
            ]

        # --- Q: question interaction
        q_tokens, q_vals = _nums_with_values(str(row.get('question', '')))
        q_set = {round(v, 6) for v in q_vals if np.isfinite(v)}
        for c in range(4):
            p = parsed[c]
            if p is None:
                Q[g, c] = [0.0, 0.0, 0.0, 0.0]
                continue
            try:
                pv = float(Fraction(p))
                in_q = float(round(pv, 6) in q_set)
            except (ValueError, ZeroDivisionError):
                pv, in_q = float('nan'), 0.0
            vals = sc[c]['_vals']
            mentions = float(sum(1 for v in vals if np.isfinite(v)
                                 and np.isfinite(pv) and abs(v - pv) < 1e-6))
            finite = [v for v in vals if np.isfinite(v)]
            last_is = float(bool(finite) and np.isfinite(pv) and abs(finite[-1] - pv) < 1e-6)
            Q[g, c] = [in_q, mentions, last_is, float(len(q_tokens))]

        # --- V: exact-arithmetic verification of the stated steps
        audits = [audit(t, p) for t, p in zip(texts, parsed)]
        steps_arr = np.array([a.n_steps for a in audits], dtype=np.float64)
        steps_mu = steps_arr.mean()
        steps_sd = max(steps_arr.std(), 1e-6)
        for c in range(4):
            a = audits[c]
            V[g, c] = [
                float(a.n_steps),
                float(a.n_ok),
                float(a.n_bad),
                (a.n_ok / a.n_steps) if a.n_steps else 0.0,
                float(a.n_bad > 0),
                (a.n_bad / a.n_steps) if a.n_steps else 0.0,
                float(a.reach_hits),
                float(a.reach_any),
                float(a.reach_last),
                (steps_arr[c] - steps_mu) / steps_sd,
            ]

        # --- G: grounding in the question's given numbers
        # Each number in the question is looked for in the trace one by one,
        # so a number that appears only as part of a larger value (12 inside
        # 120) does not count as consumed.
        q_str = str(row.get('question', ''))
        q_values = [v for v in _nums_with_values(q_str)[1] if np.isfinite(v)]
        q_unique = []
        for v in q_values:
            if not any(abs(v - u) < 1e-6 for u in q_unique):
                q_unique.append(v)
        consumed = []
        for c in range(4):
            body = [v for v in sc[c]['_vals'] if np.isfinite(v)]
            hits = 0
            for qv in q_unique:
                if any(abs(v - qv) < 1e-6 for v in body):
                    hits += 1
            consumed.append(hits)
        consumed_arr = np.array(consumed, dtype=np.float64)
        consumed_mu = consumed_arr.mean()
        consumed_sd = max(consumed_arr.std(), 1e-6)
        q_peak = max((abs(v) for v in q_unique), default=None)
        for c in range(4):
            p = parsed[c]
            try:
                pv = float(Fraction(p)) if p is not None else None
            except (ValueError, ZeroDivisionError):
                pv = None
            nq = len(q_unique)
            share = consumed[c] / nq if nq else 0.0
            idle = nq - consumed[c]
            restated = float(pv is not None
                             and any(abs(pv - qv) < 1e-6 for qv in q_unique))
            in_range = float(pv is not None and q_peak is not None
                             and abs(pv) <= q_peak + 1e-9)
            repeats = float(sum(1 for v in sc[c]['_vals'] if np.isfinite(v)
                                and any(abs(v - qv) < 1e-6 for qv in q_unique)))
            G[g, c] = [share, float(idle), restated, in_range, repeats,
                       (consumed_arr[c] - consumed_mu) / consumed_sd]

    # --- Z: within-group standardised twins of the C/Q/V/G statistics
    src = np.concatenate([C, Q, V, G], axis=-1)          # [N, 4, len(_Z_SOURCE)]
    mu = src.mean(axis=1, keepdims=True)
    sd = np.maximum(src.std(axis=1, keepdims=True), 1e-6)
    Z = (src - mu) / sd

    return np.concatenate([A, B, S, C, R, Q, V, G, Z], axis=-1)


def build_base_features(bank, arrays):
    """Return [N, 4, F] with groups A,B,S,C,R,Q (no PCA)."""
    mat = _build_group(bank, arrays)
    if not np.isfinite(mat).all():
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat.astype(np.float64)


def _select_groups(features, groups):
    idx = []
    pos = 0
    for g in _BASE_GROUPS:
        width = len(_GROUP_NAMES[g])
        if g in groups:
            idx += list(range(pos, pos + width))
        pos += width
    return features[:, :, idx]


def build_features(bank, arrays, groups=_BASE_GROUPS, pca=None):
    """Assemble the model input matrix.

    Args:
        bank   : list of question dicts
        arrays : dict with 'embeddings' [N, 4, 1024]
        groups : iterable over 'A','B','S','C','R','Q'
        pca    : optional fitted :class:`mp2lib.pca.PCAModel`

    Returns:
        ndarray [N, 4, F]
    """
    base = build_base_features(bank, arrays)
    parts = [_select_groups(base, tuple(groups))]
    if pca is not None:
        parts.append(pca.transform(arrays['embeddings']))
    return np.concatenate(parts, axis=-1)
