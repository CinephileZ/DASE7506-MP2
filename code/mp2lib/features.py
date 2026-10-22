"""Feature extraction for MP2 candidate selection.

Three feature groups (matching plan ablation units):
  A — Answer consistency: voting-based signals per candidate.
  B — Embedding: group-centred, normalised embedding statistics.
  C — Text surface: length, format markers, step/number density.
"""

import re

import numpy as np

from common import parse_answer

# --- regex constants --------------------------------------------------------

_NUMBER_RE = re.compile(r'[-+]?(?:\d[\d,]*(?:\.\d+)?)')
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

# --- feature name lists (stable ordering across calls) ----------------------

_FEATURES_A = [
    'vote_count',          # how many candidates share this answer
    'vote_ratio',          # vote_count / valid_parsed_in_group
    'strict_majority',     # 1 if vote_count > 2 (out of 4)
    'is_tied_winner',      # 1 if part of tied max-vote answers
    'group_unique_answers',# distinct parsed answers in the group
    'parsed_ok',           # 1 if candidate's text parses successfully
    'tie_break_shortest',  # among tied winners, is shortest text?
]

# B: 8 per-candidate embedding statistics (each column already indexed by candidate)
_FEATURES_B = [
    'norm',
    'dist_centroid',
    'cos_centroid',
    'sim_row_min',
    'sim_row_max',
    'sim_row_mean',
    'sim_row_std',
    'alignment',
]

# Derived embedding names for PCA projection columns (appended separately)
_FEATURES_B_PCA_PREFIX = 'pca_'

_FEATURES_C = [
    'length',            # total chars
    'word_count',        # split() tokens
    'sentence_count',    # approximate sentences
    'num_count',         # numeric values found
    'op_count',          # operators +-*%/
    'latex_count',       # inline equation blocks
    'has_boxed',         # 1 if \\boxed{ present
    'has_hashhash',      # 1 if #### present
    'bold_count',        # ** marker count
    'step_keyword_count',# transition word hits
    'numeric_density',   # num_count / word_count
    'avg_sentence_len',  # word_count / sentence_count
    'ops_per_sentence',  # op_count / sentence_count
    'hesitation_hits',  # count of hesitation words
    'conclusion_hits',   # count of conclusion words
    'int_num_count',     # integer-valued numbers
    'float_val_count',   # non-integer float/count of non-ints
]

# Combined full set (used internally)
_FULL_FEATURE_NAMES = (_FEATURES_A + _FEATURES_B + _FEATURES_C)


def _compute_answer_agreement(bank):
    """Group A: compute answer-agreement features for all questions.

    All signals derive from voting structure over parsed final answers,
    exactly matching the scorer's notion of "same answer" (common.parse_answer).
    """
    N = len(bank)
    A = np.zeros((N, 4, len(_FEATURES_A)), dtype=np.float64)

    for g in range(N):
        texts = [c['text'] for c in bank[g]['candidates']]
        parsed = [parse_answer(t) for t in texts]
        lengths = [len(t) for t in texts]

        # Vote map: parsed answer -> candidate indices supporting it
        vote_map = {}
        for i, p in enumerate(parsed):
            if p is not None:
                vote_map.setdefault(p, []).append(i)

        vote_counts = {a: len(v) for a, v in vote_map.items()}
        valid_count = sum(1 for p in parsed if p is not None)
        max_vote = max(vote_counts.values()) if vote_counts else 0
        tied_winners = [a for a, vc in vote_counts.items() if vc == max_vote and max_vote > 1]
        unique_answers = len(vote_map)

        # Shortest-text tiebreak: among candidates supporting tied top answers
        tie_break_shortest_set = set()
        if tied_winners:
            tied_indices = []
            for ans_key in tied_winners:
                tied_indices.extend(vote_map[ans_key])
            shortest_len = min(lengths[i] for i in tied_indices)
            tie_break_shortest_set = {i for i in tied_indices if lengths[i] == shortest_len}

        for c in range(4):
            p = parsed[c]
            is_valid = p is not None
            vote_cnt = len(vote_map[p]) if is_valid else 0
            is_max = is_valid and vote_cnt == max_vote

            idx = 0
            A[g, c, idx] = float(is_valid); idx += 1                  # parsed_ok
            A[g, c, idx] = float(vote_cnt); idx += 1                  # vote_count
            A[g, c, idx] = vote_cnt / max(valid_count, 1); idx += 1   # vote_ratio
            A[g, c, idx] = float(vote_cnt >= 3); idx += 1             # strict_majority
            A[g, c, idx] = float(is_max); idx += 1                    # is_tied_winner
            A[g, c, idx] = float(unique_answers); idx += 1            # group_unique_answers
            A[g, c, idx] = float(c in tie_break_shortest_set and is_max)  # tie_break_shortest

    return A


# Feature name mapping for easier lookup
_FEATURE_NAME_TO_IDX_A = {name: i for i, name in enumerate(_FEATURES_A)}


def _compute_embedding_features(arrays):
    """Group B: fully-vectorised embedding statistics per candidate.

    For each group [4 × D]:
      - Norm (raw L2), centroid distance, cosine-to-centroid
      - Pairwise cosine similarities (masked diagonal) → row stats: min/max/mean/std
    Returns array of shape [N, 4, F_emb].
    """
    emb = arrays['embeddings']  # [N, 4, D], float32
    norms = np.linalg.norm(emb, axis=-1)        # [N, 4]
    centroid = emb.mean(axis=1)                  # [N, D]
    centroid_norm = np.linalg.norm(centroid, axis=-1)  # [N]

    dist_centroid = np.linalg.norm(emb - centroid[:, None, :], axis=-1)  # [N, 4]

    cos_centroid = (emb * centroid[:, None, :]).sum(-1) / np.maximum(
        norms * centroid_norm[:, None], 1e-12)  # [N, 4]

    # Pairwise cosine similarities: [N, 4, 4], diagonal masked
    enorm = emb / np.maximum(norms[:, :, None], 1e-12)
    sims = np.einsum('nid,njd->nij', enorm, enorm)
    sims = sims * (1.0 - np.eye(4, dtype=np.float64)[None, :, :])

    sim_row_min = sims.min(axis=-1)   # [N, 4]
    sim_row_max = sims.max(axis=-1)
    sim_row_mean = sims.mean(axis=-1)
    sim_row_std = sims.std(axis=-1)

    # Alignment: norm-weighted inverse distance to centroid
    max_dist = dist_centroid.max(axis=-1, keepdims=True)
    alignment = np.sqrt(norms) * (1.0 - dist_centroid / np.maximum(max_dist, 1e-12))

    # Stack each stat → [N,4]; stack together → [N,4,8]
    return np.stack([norms, dist_centroid, cos_centroid, sim_row_min,
                     sim_row_max, sim_row_mean, sim_row_std, alignment], axis=-1)


def _extract_single_candidate_features(text):
    """Return a dict of scalar features from one candidate solution text."""
    stripped = text.strip()
    words = stripped.split()
    sentences = max(1, len(_SENTENCE_RE.findall(stripped)))

    nums = _NUMBER_RE.findall(stripped)
    ops = _OPERATOR_RE.findall(stripped)
    latex = _LATEX_RE.findall(stripped)

    float_vals = []
    int_vals = []
    for ns in nums:
        try:
            clean = ns.replace(',', '')
            v = float(clean)
            float_vals.append(v)
            if '/' not in ns and '.' not in ns:
                int_vals.append(int(v))
        except (ValueError, ZeroDivisionError):
            pass

    lower = stripped.lower()
    hes = sum(1 for w in _HESITATION_WORDS if w in lower)
    conc = sum(1 for w in _CONCLUSION_WORDS if w in lower)

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
        'step_keyword_count': sum(1 for kw in ['thus','therefore','hence','consequently',
                                                 'finally','lastly','first,', 'second,', 'next,',
                                                 'then ', 'step'] if kw in lower),
        'numeric_density': len(nums) / max(len(words), 1),
        'avg_sentence_len': len(words) / max(sentences, 1),
        'ops_per_sentence': len(ops) / max(sentences, 1),
        'hesitation_hits': hes,
        'conclusion_hits': conc,
        'int_num_count': len(int_vals),
        'float_val_count': len(float_vals) - len(int_vals),
    }


def _compute_text_features(bank):
    """Group C: text-surface features per candidate.

    Returns [N, 4, F_text].
    """
    N = len(bank)
    F_text = len(_FEATURES_C)
    names = _FEATURES_C
    mat = np.zeros((N, 4, F_text), dtype=np.float64)

    for g in range(N):
        for c in range(4):
            feat = _extract_single_candidate_features(bank[g]['candidates'][c]['text'])
            for j, fn in enumerate(names):
                mat[g, c, j] = feat[fn]
    return mat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(bank, arrays):
    """Build complete feature matrix [N, 4, F_total].

    Groups A, B, C concatenated along feature axis.  PCA features (if
    available) are appended separately by the model layer.

    Args:
        bank       : list of question dicts
        arrays     : dict with 'embeddings' key [N, 4, 1024]

    Returns:
        ndarray [N, 4, F_total]  float64
    """
    A = _compute_answer_agreement(bank)   # [N, 4, 6]
    B = _compute_embedding_features(arrays)  # [N, 4, 28]
    C = _compute_text_features(bank)        # [N, 4, 17]
    # Concatenate: 6 + 28 + 17 = 51 base features
    return np.concatenate([A, B, C], axis=-1).astype(np.float64)


def get_feature_names(with_pca=False, n_pca_components=0):
    """Return ordered feature-name list (for debugging / logging)."""
    names = list(_FEATURES_A) + list(_FEATURES_B) + list(_FEATURES_C)
    if with_pca:
        for k in range(n_pca_components):
            names.append(f'{_FEATURES_B_PCA_PREFIX}{k}')
    return names
