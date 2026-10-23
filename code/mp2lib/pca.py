"""PCA over group-centred, L2-normalised candidate embeddings.

The supplied whole-trace embeddings are not normalised (L2 ~ 57.6) and the
candidates of one question are almost collinear (cosine ~ 0.98), so the raw
similarity signal is weak.  Removing the group mean and re-normalising each
candidate isolates what is *distinctive* about that candidate's trace, and a
low-rank projection of that residual is a compact input for a nonlinear scorer.

Only the projection matrix is stored (K x 1024 float32), so the inference
asset stays far below the 16 MiB budget.
"""

import numpy as np


class PCAModel:
    def __init__(self, components, n_components):
        self.components = np.asarray(components, dtype=np.float32)  # [K, D]
        self.n_components = int(n_components)

    def transform(self, embeddings):
        """embeddings: [N, 4, D] -> residual PCA codes [N, 4, K]."""
        e = np.asarray(embeddings, dtype=np.float64)
        resid = e - e.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(resid, axis=-1, keepdims=True)
        resid = resid / np.maximum(norms, 1e-12)
        codes = resid @ self.components.T.astype(np.float64)  # [N, 4, K]
        return codes

    def to_arrays(self):
        return {'pca_components': self.components}

    @classmethod
    def from_arrays(cls, arrays):
        comp = arrays['pca_components']
        return cls(comp, comp.shape[0])


def fit_pca(embeddings, n_components=64, seed=0):
    """Fit PCA on the group-centred, L2-normalised candidate embeddings."""
    e = np.asarray(embeddings, dtype=np.float64)
    resid = e - e.mean(axis=1, keepdims=True)
    resid = resid.reshape(-1, resid.shape[-1])
    resid = resid / np.maximum(np.linalg.norm(resid, axis=-1, keepdims=True), 1e-12)
    # Randomised-free: economy SVD on the (rows x dims) centred matrix.
    # Center the residual cloud first so the projection captures variance.
    resid = resid - resid.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(resid, full_matrices=False)
    k = min(n_components, vt.shape[0])
    return PCAModel(vt[:k], k)
