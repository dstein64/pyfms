import sys

def xrange(*args, **kwargs):
    major_version = sys.version_info.major
    if major_version == 2:
        import __builtin__
        return __builtin__.xrange(*args, **kwargs)
    elif major_version >= 3:
        import builtins
        return builtins.range(*args, **kwargs)
    else:
        raise RuntimeError("Unsupported version of Python.")
