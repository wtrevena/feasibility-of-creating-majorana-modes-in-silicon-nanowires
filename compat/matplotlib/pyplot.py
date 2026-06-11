_IS_STUB = True


def __getattr__(name):
    raise RuntimeError("matplotlib stub: plotting unavailable in this "
                       "sandbox; numerical outputs only")
