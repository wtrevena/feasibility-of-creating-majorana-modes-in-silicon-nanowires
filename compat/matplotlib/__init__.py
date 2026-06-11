"""matplotlib import stub (sandbox fallback): lets analysis modules import
when only numerical output is needed. Any actual plotting call raises."""
_IS_STUB = True


def use(backend, **kw):
    pass


class _Raise:
    def __getattr__(self, name):
        raise RuntimeError("matplotlib stub: plotting unavailable in this "
                           "sandbox; numerical outputs only")
