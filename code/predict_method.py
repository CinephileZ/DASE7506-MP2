"""Frozen MP2 candidate selector.

``predict(bank, arrays, checkpoint)`` is the interface required by the course
runner (see ``submission.py``).  The method rebuilds the evidence described in
``mp2lib.features`` from the question, the four candidate texts and the
supplied whole-trace embeddings, applies the frozen scorer stored in the
checkpoint bundle, and returns the argmax candidate index per question.

Everything it needs is passed in through ``bank`` / ``arrays`` / ``checkpoint``:
no reference answer is read, no question ID is looked up and no sample count is
assumed, so the same bundle also works on an unseen audit bank.
"""

import json
from pathlib import Path

import numpy as np

from mp2lib.features import build_base_features, _select_groups
from mp2lib.model import scores_from_model
from mp2lib.pca import PCAModel


def _vote_fallback(bank):
    """Majority vote over parsed answers (longest-text tiebreak).

    Used only when no checkpoint bundle is available, so that ``predict``
    still returns a valid choice.
    """
    from common import parse_answer
    picks = []
    for row in bank:
        groups = {}
        for i, c in enumerate(row['candidates']):
            p = parse_answer(c['text'])
            if p is not None:
                groups.setdefault(p, []).append(i)
        if not groups:
            picks.append(0)
            continue
        top = max(len(v) for v in groups.values())
        winners = [i for ans, v in groups.items() if len(v) == top for i in v]
        picks.append(winners[0] if len(winners) == 1 else
                     max(winners, key=lambda i: len(row['candidates'][i]['text'])))
    return picks


def _load_bundle(checkpoint):
    """Read the frozen linear scorer; returns (model, config, pca)."""
    cp = Path(checkpoint)
    model_path, config_path = cp / 'model.npz', cp / 'config.json'
    if not (model_path.exists() and config_path.exists()):
        return None, None, None
    config = json.loads(config_path.read_text())
    if config.get('model_kind') != 'linear':
        raise ValueError(f"unsupported bundle kind {config.get('model_kind')!r}")
    with np.load(model_path, allow_pickle=False) as f:
        raw = {k: f[k] for k in f.files}
    model = {'kind': 'linear', 'weights': raw['weights'],
             'bias': float(raw['bias']), 'mean': raw['mean'], 'std': raw['std']}
    pca = PCAModel.from_arrays(raw) if 'pca_components' in raw else None
    return model, config, pca


def run(bank, arrays, checkpoint):
    model, config, pca = _load_bundle(checkpoint)
    if model is None:
        return [int(i) for i in _vote_fallback(bank)]

    groups = tuple(config['groups'])
    emb_blocks = tuple(config.get('emb_blocks', ''))
    base = build_base_features(bank, arrays)
    parts = [_select_groups(base, groups)]
    if emb_blocks:
        emb = np.asarray(arrays['embeddings'], dtype=np.float64)
        if 'E' in emb_blocks:
            parts.append(emb)
        if 'D' in emb_blocks:
            parts.append(emb - emb.mean(axis=1, keepdims=True))
    if pca is not None:
        parts.append(pca.transform(arrays['embeddings']))
    features = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
    scores = scores_from_model(features, model)
    return [int(i) for i in scores.argmax(axis=1)]
