"""Build the single QKV-ROSA CPU core and expose its Python stream API."""
if __package__:
    from . import rosa_core
else:
    import rosa_core


def build(verbose=False):
    rosa_core.build(verbose=verbose)
    return rosa_core


if __name__ == "__main__":
    print("BUILD OK:", rosa_core.build(verbose=True))
