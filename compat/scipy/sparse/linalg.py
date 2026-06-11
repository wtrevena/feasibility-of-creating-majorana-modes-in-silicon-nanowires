"""Shift-invert Lanczos eigsh for BT matrices (pure numpy).

eigsh(H, k, sigma, which="LM") returns the k eigenpairs of Hermitian H
closest to sigma, exactly the call majorana_sim.solve_lowest makes.
Algorithm: block-Thomas LU of (H - sigma*I) with one step of iterative
refinement per solve, Lanczos with full reorthogonalization on
A = (H - sigma*I)^-1, and Ritz-pair acceptance validated DIRECTLY in
H-space: a pair (E, v) is accepted only if ||H v - E v|| <= tol_h * scale,
where scale is the spectral scale of H. This is immune to ghost/spurious
Ritz values of shift-invert near a (numerically) singular shift, because
validation never trusts the Krylov recurrence itself.
Eigenvectors are returned in the ORIGINAL (pre-bmat-permutation) ordering.
"""
import numpy as np

from . import BT


class _BTFactor:
    """Block-Thomas factorization of (H - sigma I), no pivoting, plus one
    iterative-refinement step in solve()."""

    def __init__(self, H, sigma):
        N, c = H.N, H.c
        self.N, self.c = N, c
        self.H = H
        self.sigma = sigma
        self.up = H.up
        Sinv = np.empty((N, c, c), complex)
        LSi = np.empty((N - 1, c, c), complex)
        S = H.diag[0] - sigma * np.eye(c)
        Sinv[0] = np.linalg.inv(S)
        for n in range(1, N):
            LSi[n - 1] = H.lo[n - 1] @ Sinv[n - 1]
            S = H.diag[n] - sigma * np.eye(c) - LSi[n - 1] @ H.up[n - 1]
            Sinv[n] = np.linalg.inv(S)
        self.Sinv, self.LSi = Sinv, LSi

    def _solve_once(self, b):
        N = self.N
        y = b.reshape(N, self.c).copy()
        for n in range(1, N):
            y[n] -= self.LSi[n - 1] @ y[n - 1]
        x = np.empty_like(y)
        x[N - 1] = self.Sinv[N - 1] @ y[N - 1]
        for n in range(N - 2, -1, -1):
            x[n] = self.Sinv[n] @ (y[n] - self.up[n] @ x[n + 1])
        return x.reshape(-1)

    def solve(self, b):
        x = self._solve_once(b)
        r = b - (matvec(self.H, x) - self.sigma * x)
        return x + self._solve_once(r)


def matvec(H, x):
    N, c = H.N, H.c
    X = x.reshape(N, c)
    Y = np.einsum("nij,nj->ni", H.diag, X)
    Y[:-1] += np.einsum("nij,nj->ni", H.up, X[1:])
    Y[1:] += np.einsum("nij,nj->ni", H.lo, X[:-1])
    return Y.reshape(-1)


def _hscale(H):
    """Cheap spectral-scale estimate: max Gershgorin block-row sum."""
    s = np.abs(H.diag).sum(axis=2).max()
    if H.N > 1:
        s += np.abs(H.up).sum(axis=2).max() + np.abs(H.lo).sum(axis=2).max()
    return s


def eigsh(H, k=6, sigma=None, which="LM", maxiter=700, tol_h=1e-10, seed=11):
    if not isinstance(H, BT):
        raise NotImplementedError("shim eigsh handles BT matrices only")
    if sigma is None:
        raise NotImplementedError("shim eigsh requires shift-invert (sigma)")
    n = H.shape[0]
    scale = _hscale(H)
    # offset the shift away from the (numerically singular) MZM eigenvalue:
    # keeps the target eigenvalues extremal in (H - sigma_eff)^-1 while
    # bounding the solve condition number ~1e7.
    sigma_eff = sigma + 1e-7 * scale
    F = _BTFactor(H, sigma_eff)
    rng = np.random.default_rng(seed)
    m_max = min(maxiter, n - 2)
    Q = np.empty((n, m_max + 1), complex)
    q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    q /= np.linalg.norm(q)
    Q[:, 0] = q
    alphas, betas = [], []

    def harvest(m):
        """H-space-validated Ritz pairs, sorted by |E - sigma|; None if <k."""
        T = np.diag(np.array(alphas[:m], float))
        if m > 1:
            ob = np.array(betas[:m - 1], float)
            T += np.diag(ob, 1) + np.diag(ob, -1)
        theta, s = np.linalg.eigh(T)
        idx = np.argsort(-np.abs(theta))[:min(3 * k, m)]
        ok_E, ok_V = [], []
        for i in idx:
            if abs(theta[i]) < 1e-300:
                continue
            v = Q[:, :m] @ s[:, i]
            nv = np.linalg.norm(v)
            if nv < 0.5:
                continue
            v /= nv
            E = sigma_eff + 1.0 / theta[i]
            r = np.linalg.norm(matvec(H, v) - E * v)
            if r <= tol_h * scale:
                ok_E.append(E)
                ok_V.append(v)
        if len(ok_E) < k:
            return None
        ok_E = np.array(ok_E)
        order = np.argsort(np.abs(ok_E - sigma))[:k]
        return np.real(ok_E[order]), np.column_stack([ok_V[i] for i in order])

    got = None
    for j in range(m_max):
        w = F.solve(Q[:, j])
        a = np.real(np.vdot(Q[:, j], w))
        w -= a * Q[:, j]
        if j > 0:
            w -= betas[-1] * Q[:, j - 1]
        for _ in range(2):               # full reorthogonalization, twice
            w -= Q[:, :j + 1] @ (Q[:, :j + 1].conj().T @ w)
        b = np.linalg.norm(w)
        alphas.append(a)
        if b < 1e-14 * max(1.0, np.abs(alphas[0])):
            got = harvest(j + 1)
            break
        betas.append(b)
        Q[:, j + 1] = w / b
        if (j + 1) >= max(2 * k + 4, 24) and (j + 1) % 12 == 0:
            got = harvest(j + 1)
            if got is not None:
                break
    if got is None:
        got = harvest(len(alphas))
    if got is None:
        tol_h *= 100.0            # still ~1e-4 ueV absolute: report, accept
        got = harvest(len(alphas))
    if got is None:
        raise RuntimeError(f"shim eigsh: <{k} H-validated Ritz pairs after "
                           f"{len(alphas)} Lanczos steps")
    vals, vecs = got
    if H.perm is not None:
        out = np.empty_like(vecs)
        out[H.perm, :] = vecs
        vecs = out
    order = np.argsort(vals)
    return vals[order], vecs[:, order]
