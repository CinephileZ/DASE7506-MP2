"""Linear scorer fitted with group-softmax cross-entropy.

Decision objective: for each question, softmax over the 4 candidate scores;
the target distribution places uniform mass on every correct candidate
(handles groups with multiple correct answers directly) and optimises
argmax accuracy rather than per-candidate calibration.

Pure NumPy so the inference asset is a single weight vector (< 1 KiB at
51 features) and inference is one matrix multiply.
"""

import json
from pathlib import Path

import numpy as np


def standardize(features):
    """Return (mean, std) over the flattened [N*4, F] feature matrix."""
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def scores_from_weights(features, result):
    """Apply the trained linear scorer: standardize features then score.

    result must carry 'mean', 'std', 'weights', 'bias' (as returned by
    fit_softmax_ce and stored in the bundle).
    """
    mean = result['mean']
    std = result['std']
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    flat = (flat - mean) / std
    return (flat @ result['weights'] + result['bias']).reshape(features.shape[:2])


def fit_softmax_ce(features, labels, l2=1e-3, lr=0.05, epochs=2000, seed=0,
                   standardize_first=True, verbose=False):
    """Fit linear weights by group-softmax cross-entropy with Adam.

    features : [N, 4, F] raw features (standardised internally if requested)
    labels   : [N, 4] bool — True for candidates whose parsed answer is gold

    Returns dict with weights, bias, mean, std (all JSON/NPZ friendly).
    """
    N, C, F = features.shape
    x = features.reshape(-1, F).astype(np.float64)
    if standardize_first:
        mean = x.mean(axis=0)
        std = np.maximum(x.std(axis=0), 1e-6)
        x = (x - mean) / std
    else:
        mean, std = np.zeros(F), np.ones(F)

    xg = x.reshape(N, C, F)
    # Target: uniform over correct candidates (at least one by protocol)
    tgt = labels.astype(np.float64)
    tgt_sum = tgt.sum(axis=1, keepdims=True)
    tgt = tgt / np.maximum(tgt_sum, 1e-12)

    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=F)
    b = 0.0
    mw = np.zeros_like(w)
    vw = np.zeros_like(w)
    mb = vb = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, epochs + 1):
        logits = xg @ w + b                      # [N, C]
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)  # [N, C]

        # dL/dlogits = (p - t) / N ; then chain through features
        grad_logits = (prob - tgt) / N           # [N, C]
        gw = np.einsum('nc,ncf->f', grad_logits, xg) + l2 * w
        gb = grad_logits.sum()

        mw = beta1 * mw + (1 - beta1) * gw
        vw = beta2 * vw + (1 - beta2) * (gw * gw)
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * (gb * gb)
        mwh = mw / (1 - beta1 ** step)
        vwh = vw / (1 - beta2 ** step)
        mbh = mb / (1 - beta1 ** step)
        vbh = vb / (1 - beta2 ** step)
        w -= lr * mwh / (np.sqrt(vwh) + eps)
        b -= lr * mbh / (np.sqrt(vbh) + eps)

        if verbose and (step % 500 == 0 or step == 1):
            loss = -np.log(np.maximum((prob * tgt).sum(axis=1), 1e-12)).mean()
            acc = (prob.argmax(axis=1) == tgt.argmax(axis=1)).mean()
            print(f'  step {step}: loss={loss:.4f} acc={acc:.4f}')

    return {
        'weights': w.astype(np.float32),
        'bias': float(b),
        'mean': mean.astype(np.float32),
        'std': std.astype(np.float32),
    }


def predict_choices(bank, features, bundle):
    """Select one candidate per question via argmax of the linear scorer.

    Ties (identical scores) resolved by lowest index, matching np.argmax.
    """
    scores = scores_from_weights(features, bundle)
    picks = scores.argmax(axis=1)
    return [int(p) for p in picks]


def save_bundle(path, bundle, config):
    """Persist weights and config as a compact .npz + .json pair."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / 'model.npz',
             weights=bundle['weights'], bias=np.float32(bundle['bias']),
             mean=bundle['mean'], std=bundle['std'])
    (path / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


def load_bundle(path):
    """Load weights + config; returns (bundle, config)."""
    path = Path(path)
    with np.load(path / 'model.npz', allow_pickle=False) as f:
        bundle = {k: f[k] for k in ('weights', 'bias', 'mean', 'std')}
    bundle['bias'] = float(bundle['bias'])
    config = json.loads((path / 'config.json').read_text())
    return bundle, config