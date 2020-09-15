import torch
from .grad_mode import _DecoratorContextManager

from typing import Any

# Global variable used to make the python API simpler to use
_current_level = -1

def enter_dual_level():
    global _current_level
    new_level = torch._C.enter_dual_level()
    if new_level != _current_level + 1:
        raise RuntimeError("Entering a new forward AD level but the current level "
                           "is not valid. Make sure you did not modified it directly.")
    _current_level = new_level
    return new_level

def exit_dual_level(*, level=None):
    global _current_level
    if level is None:
        level = _current_level
    if level != _current_level:
        raise RuntimeError("Trying to exit a forward AD level that was not the last one "
                           "that was created. This is not supported.")
    torch._C.exit_dual_level(level=level)
    _current_level = level - 1

def make_dual(tensor, tangent, *, level=None):
    if level is None:
        level = _current_level

    if level < 0:
        raise RuntimeError("Trying to create a dual Tensor for forward AD but no level "
                           "exists, make sure to enter_dual_level() first.")

    return torch._C.make_dual(tensor, tangent, level=level)

def unpack_dual(tensor, *, level=None):
    if level is None:
        level = _current_level

    if level < 0:
        return tensor, None

    return torch._C.unpack_dual(tensor, level=level)

class dual_level(_DecoratorContextManager):
    def __init__(self):
        super().__init__()

    def __enter__(self):
        return enter_dual_level()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        exit_dual_level()
