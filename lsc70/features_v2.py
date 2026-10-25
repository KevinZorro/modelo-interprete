"""Candidate feature version v2 for T-05: scale by palm instead of by max norm.

v1 (`normalizacion.vector_features`) divides by the largest wrist-to-landmark distance, so
an extended finger changes the scale of the whole hand. v2 divides by the wrist to
middle-finger-MCP distance (landmark 9), which does not depend on finger pose.

v2 is derived from v1 vectors with no raw landmarks needed: v1 = raw / maxnorm, hence
v1 / ||v1[9]|| = raw / ||raw[9]||. This lets us test it on the cache and on the negatives
(which only store v1). It does NOT replace the contract (docs/contrato-entrada.md); it is
adopted only if the T-05 comparison says so.
"""
import numpy as np

MIDDLE_MCP = 9
MIN_PALM = 1e-6


def palm_scale(X):
    """(n, 63) v1 vectors -> (n, 63) v2 vectors. Rows with a degenerate palm become zeros."""
    X = np.asarray(X, dtype=np.float32)
    pts = X.reshape(len(X), 21, 3)
    palm = np.linalg.norm(pts[:, MIDDLE_MCP], axis=1)
    ok = palm > MIN_PALM
    out = np.zeros_like(pts)
    out[ok] = pts[ok] / palm[ok, None, None]
    return out.reshape(len(X), 63)
