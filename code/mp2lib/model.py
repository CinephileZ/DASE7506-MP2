"""Candidate scorers fitted with group-softmax cross-entropy.

Decision objective: for each question a softmax is taken over the four
candidate scores, and the target distribution puts uniform mass on every
correct candidate.  This handles groups with several correct candidates
directly and optimises the argmax decision rather than per-candidate
calibration.

Two estimators share the objective:

* :func:`fit_linear`  — a single weight vector (the step-3 baseline);
* :func:`fit_mlp`     — a small pointwise MLP over the same features, which
  adds the interactions a linear scorer cannot express.

Both are pure NumPy, so inference is a few matrix multiplies and the asset is
a single compact ``.npz``.
"""

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def standardize_stats(features):
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), 1e-6)
    return mean, std


# Standardised features are capped so a single outlier cannot dominate the
# dense embedding blocks; the cap sits far out in the tail of the training
# distribution, where it acts as a safety net rather than a data transform.
_CAP = 12.0


def _apply_standardize(features, mean, std):
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    return np.clip((flat - mean) / std, -_CAP, _CAP)


def _targets(labels):
    tgt = labels.astype(np.float64).copy()
    tgt_sum = tgt.sum(axis=1, keepdims=True)
    return tgt / np.maximum(tgt_sum, 1e-12)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# linear scorer
# ---------------------------------------------------------------------------

def fit_linear(features, labels, l2=1e-3, lr=0.05, epochs=3000, seed=0,
               verbose=False, l2_vector=None):
    """Fit a single weight vector by group-softmax CE with Adam.

    ``l2_vector`` overrides ``l2`` with a per-feature penalty vector, which
    lets dense embedding blocks be shrunk much harder than the sparse
    hand-crafted statistics without changing the data.
    """
    n, c, f = features.shape
    mean, std = standardize_stats(features)
    xg = _apply_standardize(features, mean, std).reshape(n, c, f)
    tgt = _targets(labels)
    reg = np.broadcast_to(np.asarray(l2 if l2_vector is None else l2_vector,
                                    dtype=np.float64), (f,))

    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=f)
    b = 0.0
    mw = np.zeros_like(w); vw = np.zeros_like(w)
    mb = vb = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    xf = xg.reshape(n * c, f)

    for step in range(1, epochs + 1):
        prob = _softmax(xg @ w + b)
        gz = (prob - tgt) / n                                  # [N, C]
        gw = gz.reshape(-1) @ xf + reg * w
        gb = gz.sum()
        mw = beta1 * mw + (1 - beta1) * gw
        vw = beta2 * vw + (1 - beta2) * gw * gw
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * gb * gb
        w -= lr * (mw / (1 - beta1 ** step)) / (np.sqrt(vw / (1 - beta2 ** step)) + eps)
        b -= lr * (mb / (1 - beta1 ** step)) / (np.sqrt(vb / (1 - beta2 ** step)) + eps)
        if verbose and (step == 1 or step % 500 == 0):
            loss = -np.log(np.maximum((prob * tgt).sum(1), 1e-12)).mean()
            print(f'  step {step}: loss={loss:.4f}')

    return {'kind': 'linear', 'weights': w, 'bias': float(b),
            'mean': mean, 'std': std}


# ---------------------------------------------------------------------------
# MLP scorer
# ---------------------------------------------------------------------------

def fit_mlp(features, labels, hidden=64, l2=1e-3, lr=3e-3, epochs=1500,
            seed=0, dropout=0.0, verbose=False):
    """Fit a pointwise MLP (F -> hidden -> 1) by group-softmax CE.

    ``dropout`` is applied to the hidden activation during training only.
    """
    n, c, f = features.shape
    mean, std = standardize_stats(features)
    x = _apply_standardize(features, mean, std).reshape(n * c, f)
    tgt = _targets(labels)

    rng = np.random.default_rng(seed)
    scale1 = np.sqrt(2.0 / f)
    scale2 = np.sqrt(2.0 / hidden)
    w1 = rng.normal(0.0, scale1, size=(f, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0.0, scale2, size=(hidden, 1))
    b2 = np.zeros(1)

    params = [w1, b1, w2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, epochs + 1):
        h_pre = x @ w1 + b1
        h = np.maximum(h_pre, 0.0)
        if dropout > 0.0:
            mask = (rng.random(h.shape) >= dropout) / (1.0 - dropout)
            h = h * mask
        else:
            mask = None
        s = (h @ w2 + b2).reshape(n, c)
        prob = _softmax(s)
        gz = ((prob - tgt) / n).reshape(-1, 1)          # [M, 1]
        gw2 = h.T @ gz + l2 * w2
        gb2 = gz.sum(axis=0)
        gh = (gz @ w2.T) * (h_pre > 0)
        if mask is not None:
            gh = gh * mask
        gw1 = x.T @ gh + l2 * w1
        gb1 = gh.sum(axis=0)
        grads = [gw1, gb1, gw2, gb2]

        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            p -= lr * (m[i] / (1 - beta1 ** step)) / (np.sqrt(v[i] / (1 - beta2 ** step)) + eps)

        if verbose and (step == 1 or step % 500 == 0):
            loss = -np.log(np.maximum((prob * tgt).sum(1), 1e-12)).mean()
            print(f'  step {step}: loss={loss:.4f}')

    return {'kind': 'mlp', 'w1': w1, 'b1': b1, 'w2': w2.reshape(-1), 'b2': float(b2[0]),
            'hidden': hidden, 'mean': mean, 'std': std}


def fit_linear_lbfgs(features, labels, l2_vector=None, l2=1e-3, max_iter=600,
                     seed=0):
    """Fit the linear scorer with L-BFGS on the exact full-batch objective.

    Adam's fixed step budget leaves the softmax-CE surface partly unexplored;
    L-BFGS runs to its own convergence criterion, and fitting from two
    starting points (random + zero) keeps the better loss, which makes the
    fit deterministic and insensitive to the initialisation.
    """
    from scipy.optimize import minimize

    n, c, f = features.shape
    mean, std = standardize_stats(features)
    xg = _apply_standardize(features, mean, std).reshape(n, c, f)
    tgt = _targets(labels)
    reg = np.broadcast_to(np.asarray(l2 if l2_vector is None else l2_vector,
                                    dtype=np.float64), (f,))

    def loss_grad(w):
        scores = xg @ w
        scores = scores - scores.max(axis=1, keepdims=True)
        lse = np.log(np.exp(scores).sum(axis=1))
        loss = (lse - (tgt * scores).sum(axis=1)).mean()
        p = np.exp(scores - lse[:, None])
        grad = np.einsum('nc,ncf->f', (p - tgt) / n, xg) + reg * w
        return loss + 0.5 * float((reg * w * w).sum()), grad

    rng = np.random.default_rng(seed)
    best_w, best_loss = None, np.inf
    # Two starting points: the origin, and a random vector scaled like the
    # data (1/sqrt(d)).  L-BFGS can get stuck in a flat region from one of
    # them, so the lower-loss solution is kept.
    for w0 in (np.zeros(f), rng.normal(0.0, 1.0 / np.sqrt(f), size=f)):
        res = minimize(loss_grad, w0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'maxfun': max_iter * 2})
        if res.fun < best_loss:
            best_loss, best_w = float(res.fun), res.x
    return {'kind': 'linear', 'weights': best_w, 'bias': 0.0,
            'mean': mean, 'std': std}


def fit_linear_ensemble(features, labels, n_models=3, seed=0, **kw):
    """Fit ``n_models`` linear scorers with different seeds and average them.

    The members share the standardisation statistics (they see identical
    inputs), so only the weights are stacked.
    """
    members = [fit_linear(features, labels, seed=seed + i, **kw)
               for i in range(n_models)]
    return {'kind': 'ensemble',
            'weights': np.stack([m['weights'] for m in members]),  # [M, F]
            'bias': np.array([m['bias'] for m in members]),        # [M]
            'mean': members[0]['mean'], 'std': members[0]['std']}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def scores_from_model(features, model):
    """Return [N, 4] scores for any estimator kind."""
    x = _apply_standardize(features, model['mean'], model['std'])
    kind = model['kind']
    if kind == 'linear':
        s = x @ model['weights'] + model['bias']
    elif kind == 'ensemble':
        s = (x @ model['weights'].T + model['bias']).mean(axis=1)
    else:
        h = np.maximum(x @ model['w1'] + model['b1'], 0.0)
        s = h @ model['w2'] + model['b2']
    return s.reshape(features.shape[:2])


def predict_choices(features, model):
    """Argmax choice per question (ties → lowest index, matching np.argmax)."""
    return [int(i) for i in scores_from_model(features, model).argmax(axis=1)]


# ---------------------------------------------------------------------------
# bundle IO
# ---------------------------------------------------------------------------

_BUNDLE_KEYS = {
    'linear': ('weights', 'bias', 'mean', 'std'),
    'ensemble': ('weights', 'bias', 'mean', 'std'),
    'mlp': ('w1', 'b1', 'w2', 'b2', 'mean', 'std'),
}


def save_bundle(path, model, config, pca=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(model[k]) for k in _BUNDLE_KEYS[model['kind']]}
    if pca is not None:
        arrays.update(pca.to_arrays())
    np.savez(path / 'model.npz', **arrays)
    (path / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


def load_bundle(path):
    path = Path(path)
    with np.load(path / 'model.npz', allow_pickle=False) as f:
        raw = {k: f[k] for k in f.files}
    config = json.loads((path / 'config.json').read_text())
    kind = config.get('model_kind', 'linear')
    model = {k: raw[k] for k in _BUNDLE_KEYS[kind]}
    model['kind'] = kind
    if kind == 'mlp':
        model['hidden'] = int(model['w1'].shape[1])
        model['b2'] = float(model['b2'])
        model['w2'] = model['w2'].reshape(-1)
    elif kind == 'ensemble':
        model['bias'] = np.asarray(model['bias'], dtype=np.float64)
    else:
        model['bias'] = float(model['bias'])
    from mp2lib.pca import PCAModel
    pca = PCAModel.from_arrays(raw) if 'pca_components' in raw else None
    return model, config, pca
