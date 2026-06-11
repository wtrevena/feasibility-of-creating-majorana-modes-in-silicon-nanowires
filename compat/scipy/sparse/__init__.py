"""Block-tridiagonal stand-in for the scipy.sparse calls used in majorana_sim.

Representation: BT(N, c) holds diag (N,c,c), up (N-1,c,c), lo (N-1,c,c)
complex blocks for an (N*c) x (N*c) matrix that is block-tridiagonal in
site-major ordering. bmat([[h, Dm], [Dm^H, -h*]]) interleaves the particle
and hole blocks site-by-site (block size 2c) and records the permutation
back to scipy's [particle; hole] ordering so eigenvectors keep the layout
the rest of the code assumes.
"""
import numpy as np


class BT:
    def __init__(self, N, c, diag=None, up=None, lo=None, perm=None):
        self.N, self.c = N, c
        z = lambda n: np.zeros((n, c, c), complex)
        self.diag = z(N) if diag is None else np.asarray(diag, complex)
        self.up = z(N - 1) if up is None else np.asarray(up, complex)
        self.lo = z(N - 1) if lo is None else np.asarray(lo, complex)
        self.perm = perm        # new-index -> original-index (or None)
        self.shape = (N * c, N * c)

    def _like(self, d, u, l):
        return BT(self.N, self.c, d, u, l, self.perm)

    def conj(self):
        return self._like(self.diag.conj(), self.up.conj(), self.lo.conj())

    @property
    def T(self):
        return self._like(self.diag.transpose(0, 2, 1),
                          self.lo.transpose(0, 2, 1),
                          self.up.transpose(0, 2, 1))

    def __neg__(self):
        return self._like(-self.diag, -self.up, -self.lo)

    def __add__(self, o):
        if isinstance(o, LilBT):
            o = o.tocsr()
        if not isinstance(o, BT) or o.N != self.N or o.c != self.c:
            raise TypeError("BT + incompatible")
        return self._like(self.diag + o.diag, self.up + o.up,
                          self.lo + o.lo)

    __radd__ = __add__

    def __mul__(self, s):
        if np.isscalar(s):
            return self._like(self.diag * s, self.up * s, self.lo * s)
        raise TypeError("BT * non-scalar")

    __rmul__ = __mul__

    def tocsr(self):
        return self

    tocsc = tolil = todia = tocoo = tocsr

    def todense(self):
        n = self.N * self.c
        M = np.zeros((n, n), complex)
        c = self.c
        for i in range(self.N):
            M[i*c:(i+1)*c, i*c:(i+1)*c] = self.diag[i]
            if i < self.N - 1:
                M[i*c:(i+1)*c, (i+1)*c:(i+2)*c] = self.up[i]
                M[(i+1)*c:(i+2)*c, i*c:(i+1)*c] = self.lo[i]
        if self.perm is not None:           # express in ORIGINAL ordering
            p = self.perm
            Mo = np.zeros_like(M)
            Mo[np.ix_(p, p)] = M
            return Mo
        return M

    toarray = todense


class _OffDiag:
    """sp.diags(values, +-1) marker: only consumed by kron()."""
    def __init__(self, vals, off):
        self.vals = np.asarray(vals, complex)
        self.off = off
        n = len(self.vals) + 1
        self.shape = (n, n)

    @property
    def T(self):
        return _OffDiag(self.vals, -self.off)


class _DiagVec:
    def __init__(self, vals):
        self.vals = np.asarray(vals, complex)
        self.shape = (len(vals), len(vals))


class _Eye(_DiagVec):
    def __init__(self, n):
        super().__init__(np.ones(n))


class LilBT:
    """lil_matrix replacement: accepts block-aligned 2D-slice writes that lie
    on the (block-tridiagonal) structure; converted to BT by tocsr()."""
    def __init__(self, shape, dtype=complex):
        self.shape = shape
        self.blocks = {}        # (i0, j0) -> array
        self.bs = None

    def __setitem__(self, key, val):
        rs, cs = key
        val = np.asarray(val, complex)
        h, w = val.shape
        if self.bs is None:
            self.bs = h
        if h != self.bs or w != self.bs:
            raise ValueError("LilBT: uniform square blocks only")
        if rs.start % h or cs.start % w:
            raise ValueError("LilBT: block-aligned writes only")
        self.blocks[(rs.start // h, cs.start // w)] = val.copy()

    def tocsr(self):
        c = self.bs
        N = self.shape[0] // c
        bt = BT(N, c)
        for (i, j), v in self.blocks.items():
            if j == i:
                bt.diag[i] += v
            elif j == i + 1:
                bt.up[i] += v
            elif j == i - 1:
                bt.lo[j] += v
            else:
                raise ValueError("LilBT: write outside tridiagonal structure")
        return bt


def lil_matrix(shape, dtype=complex):
    return LilBT(shape, dtype)


def block_diag(blocks, format=None):
    d = np.stack([np.asarray(b, complex) for b in blocks])
    return BT(d.shape[0], d.shape[1], diag=d)


def diags(vals, offsets=0, format=None):
    if offsets == 0:
        return _DiagVec(vals)
    if offsets in (1, -1):
        return _OffDiag(vals, offsets)
    raise NotImplementedError("diags: offsets in {-1,0,1} only")


def eye(n, format=None):
    return _Eye(n)


def kron(a, b, format=None):
    b = np.asarray(b, complex) if not isinstance(b, (BT,)) else b
    if isinstance(a, _OffDiag):
        N = a.shape[0]
        c = b.shape[0]
        tiles = a.vals[:, None, None] * b[None, :, :]
        if a.off == 1:
            return BT(N, c, up=tiles)
        return BT(N, c, lo=tiles)
    if isinstance(a, _DiagVec):           # includes _Eye
        N = a.shape[0]
        c = b.shape[0]
        return BT(N, c, diag=a.vals[:, None, None] * b[None, :, :])
    raise NotImplementedError("kron: first arg must be eye/diags marker")


def bmat(rows, format=None):
    (A, Bm), (C, D) = rows
    parts = [M.tocsr() if isinstance(M, LilBT) else M for M in (A, Bm, C, D)]
    A, Bm, C, D = parts
    if not all(isinstance(M, BT) and M.N == A.N and M.c == A.c
               for M in parts):
        raise NotImplementedError("bmat: 2x2 of equal-structure BT only")
    N, c = A.N, A.c
    c2 = 2 * c

    def two(Pa, Pb, Pc, Pd, n):
        out = np.zeros((n, c2, c2), complex)
        out[:, :c, :c] = Pa
        out[:, :c, c:] = Pb
        out[:, c:, :c] = Pc
        out[:, c:, c:] = Pd
        return out

    diag = two(A.diag, Bm.diag, C.diag, D.diag, N)
    up = two(A.up, Bm.up, C.up, D.up, N - 1)
    lo = two(A.lo, Bm.lo, C.lo, D.lo, N - 1)
    # permutation: new index n*c2 + i  ->  original index
    perm = np.empty(N * c2, int)
    for n in range(N):
        perm[n*c2:n*c2 + c] = n * c + np.arange(c)
        perm[n*c2 + c:(n+1)*c2] = N * c + n * c + np.arange(c)
    return BT(N, c2, diag, up, lo, perm)


from . import linalg  # noqa: E402,F401
