import warnings
from enum import Enum
from contextlib import contextmanager
from functools import wraps
from typing import Dict

_FEATURE_STORE: Dict[str, None] = {}
_ALWAYS_WARN = False

class FeatureType(Enum):
    prototype = "prototype"
    beta = "beta"

class PrototypeFeatureWarning(Warning):
    pass

class BetaFeatureWarning(Warning):
    pass

def disable(name: str):
    r"""Function that can be used to disable warnings for a specific feature.
    """
    _FEATURE_STORE[name] = None

def declare_feature(tp: FeatureType, name: str, extra_msg: str = ""):
    r"""Function that can be called just before a prototype or beta feature
    is used to do proper warning.
    Note that the warning will be thrown only once for each name.
    """
    if not _ALWAYS_WARN and name in _FEATURE_STORE:
        return

    _FEATURE_STORE[name] = None
    msg = f"The feature '{name}' that you are using is a " \
          f"{tp} feature. Be careful as this API might change in " \
          f"the future. Use torch.utils.feature_warning.disable('{name}') to " \
          f"disable this warning if needed. {extra_msg}"
    cls = PrototypeFeatureWarning if tp is FeatureType.prototype else BetaFeatureWarning
    warnings.warn(msg, cls)

def wrap_feature(tp: FeatureType, name: str = None, extra_msg: str = ""):
    r"""Decorator that can be used to mark a class or function as a prototype
    or beta feature.
    Note that the warning will be thrown only once.
    """
    def tmp(obj):
        if name is None:
            if hasattr(obj, "__module__") and obj.__module__:
                feat_name = f"{obj.__module__}.{obj.__name__}"
            else:
                raise RuntimeError("Object passed to wrap_feature does not have a "
                                   "proper __module__ so the 'name' must be provided.")
        else:
            feat_name = name

        @wraps(obj)
        def wrapper(*args, **kwargs):
            declare_feature(tp, feat_name, extra_msg)
            return obj(*args, **kwargs)
        return wrapper
    return tmp

# Mostly used for testing
@contextmanager
def always_warn():
    r"""Force warnings to always be triggered, ignoring disables and previously
    thrown warnings.
    """
    global _ALWAYS_WARN
    _ALWAYS_WARN = True
    try:
        yield
    finally:
        _ALWAYS_WARN = False
