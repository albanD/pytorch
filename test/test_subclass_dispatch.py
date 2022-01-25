import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._pytree import tree_map
from torch._C import _disabled_torch_function_impl

import functools

class WrapperTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        t, kwargs = cls.get_wrapper_properties(*args, **kwargs)
        if "size" not in kwargs:
            size = t.size()
        else:
            size = kwargs["size"]
            del kwargs["size"]
        if "dtype" not in kwargs:
            kwargs["dtype"] = t.dtype
        if "layout" not in kwargs:
            kwargs["layout"] = t.layout
        if "device" not in kwargs:
            kwargs["device"] = t.device
        if "requires_grad" not in kwargs:
            kwargs["requires_grad"] = False
        # Ignore memory_format and pin memory for now as I don't know how to
        # safely access them on a Tensor (if possible??)

        wrapper = torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)
        wrapper._validate_methods()
        return wrapper

    @classmethod
    def get_wrapper_properties(cls, *args, **kwargs):
        raise NotImplementedError("You need to implement get_wrapper_properties")

    def _validate_methods(self):
        # Skip this if not in debug mode?
        # Changing these on the python side is wrong as it would not be properly reflected
        # on the c++ side
        # This doesn't catch attributes set in the __init__
        forbidden_overrides = ["size", "stride", "dtype", "layout", "device", "requires_grad"]
        for el in forbidden_overrides:
            if getattr(self.__class__, el) is not getattr(torch.Tensor, el):
                raise RuntimeError(f"Subclass {self.__class__.__name__} is overwriting the "
                                   f"property {el} but this is not allowed as such change would "
                                   "not be reflected to c++ callers.")

    def __repr__(self, extra=None):
        if extra is not None:
            extra = [""] + extra
        extra_str = ", ".join(extra) if extra else ""
        return f"{self.__class__.__name__}({self.__dict__}{extra_str})"

# General wrapper
class GeneralWrapper(WrapperTensor):
    elem: torch.Tensor

    def __init__(self, elem, *args, **kwargs):
        self.elem = elem

    @classmethod
    def get_wrapper_properties(cls, elem, *args, **kwargs):
        return elem, {}

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def unwrap(cls, e):
        return e.elem if isinstance(e, cls) else e

    @classmethod
    def wrap(cls, extra_args, extra_kwargs, e):
        return cls(e, *extra_args, **extra_kwargs) if isinstance(e, torch.Tensor) else e

    # Basic no-op implementation if needed
    @classmethod
    def _delegate_call(cls, func, types, args, kwargs, extra_args=tuple(), extra_kwargs=None):
        # print(f"Delegating call for {cls.__name__} to {func.__module__}.{func.__name__} with {types}")
        if extra_kwargs is None:
            extra_kwargs = {}

        wrapper = functools.partial(cls.wrap, extra_args, extra_kwargs)

        raw_res = func(*tree_map(cls.unwrap, args), **tree_map(cls.unwrap, kwargs))
        res = tree_map(wrapper, raw_res)
        return res

    # Default no-op operation
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return cls._delegate_call(func, types, args, kwargs)

# Some printing tool
class PrintingTensor(GeneralWrapper):
    def __init__(self, elem, *args, **kwargs):
        super().__init__(elem, *args, **kwargs)
        assert len(args) == 1
        assert len(kwargs) == 0
        assert isinstance(args[0], list)
        self.printer = args[0]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        printer = None
        for a in args:
            if isinstance(a, PrintingTensor):
                printer = a.printer
                break
        assert printer is not None
        printer.append(f"{func.__module__}.{func.__name__}")

        return cls._delegate_call(func, types, args, kwargs, (printer,))

# Used as a dummy wrapper to "hide" Tensors we want to ignore while going through
# the dispatcher. It's special power is to disappear as soon as it gets to dispatch.
# It does that by unpacking itself and never repacking
class BackwardADTrackingTensor(GeneralWrapper):
    def __init__(self, elem, *args, **kwargs):
        super().__init__(elem, *args, **kwargs)
        assert len(args) == 0
        assert len(kwargs) == 0
        # This Tensor should never require gradients
        assert self.requires_grad == False

    @classmethod
    def wrap(cls, extra_args, extra_kwargs, e):
        return e


# Only the wrapped Tensor has autograd meta.
# This is necessary because if the wrapper has some, it would be handled before
# we get to the torch dispatch call
class BackwardADTensor(GeneralWrapper):
    def __init__(self, elem, *args, **kwargs):
        super().__init__(elem, *args, **kwargs)
        assert len(args) == 0
        assert len(kwargs) == 0 or (len(kwargs) == 1 and "requires_grad" in kwargs)
        if kwargs.get("requires_grad", False):
            # User asked for this wrapper to be a leaf so we make sure the wrapped
            # Tensor is a leaf
            assert self.elem.requires_grad is False
            assert self.elem.grad_fn is None
            self.elem.requires_grad = True

    @classmethod
    def unwrap(cls, e):
        # Special logic here where we want to unwrap all of the Tensor of this type
        # while wrapping everything else to make sure they don't "polute" our autograd
        # handling
        if not isinstance(e, torch.Tensor):
            return e
        else:
            return e.elem if isinstance(e, cls) else BackwardADTrackingTensor(e)


class TestSubclassDispatch(TestCase):
    def test_base_print(self):
        print_list = []
        foo = PrintingTensor(torch.rand(2), print_list)

        b = foo * foo
        b += 2
        b.sum()

        expected_print = ['torch._ops.aten.mul', 'torch._ops.aten.add_', 'torch._ops.aten.sum']
        self.assertEqual(print_list, expected_print)


    def test_bw_ad(self):
        BWLevel0 = type("BWLevel0", (BackwardADTensor,), {})
        foo = BWLevel0(torch.rand(2, requires_grad=True))
        bar = BWLevel0(torch.rand(2, requires_grad=True))

        print("prod between subclasses")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)
        self.assertIsNotNone(res.elem.grad_fn)

        bar = torch.rand(2)
        print("prod with plain Tensor")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)
        self.assertIsNotNone(res.elem.grad_fn)

        # Should not need the .elem?
        out = res.sum()
        out.elem.backward(torch.ones([]))
        print(foo.grad)
        self.assertIsNone(foo.grad)
        print(foo.elem.grad)
        self.assertEqual(foo.elem.grad, bar)
        print(bar.grad)
        self.assertIsNone(bar.grad)


    def test_bw_ad_multilvl(self):
        BWLevel0 = type("BWLevel0", (BackwardADTensor,), {})
        BWLevel1 = type("BWLevel1", (BWLevel0,), {})
        foo = BWLevel0(torch.rand(2, requires_grad=True))
        bar = BWLevel1(torch.rand(2, requires_grad=True))

        print("prod")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)
        out = res.sum()
        self.assertIsNotNone(out.elem.grad_fn)

        print("back of lvl0")
        # Should not need the .elem?
        out.elem.backward(torch.ones([]), retain_graph=True)
        print(foo.elem.grad)
        self.assertIsNone(foo.elem.grad)
        print(bar.elem.grad)
        # Unpack all the levels to avoid "Delegating " print spam
        self.assertEqual(bar.elem.grad.elem.elem.elem, foo.elem.elem)

        print("back of lvl1")
        # Should not need the .elem?
        out.elem.elem.elem.backward(torch.ones([]))
        print(foo.elem.grad)
        self.assertEqual(foo.elem.grad.elem, bar.elem.elem)
        print(bar.elem.grad)
        self.assertEqual(bar.elem.grad.elem.elem.elem, foo.elem.elem)

    def test_bw_ad_plain(self):
        BWLevel0 = type("BWLevel0", (BackwardADTensor,), {})
        foo = BWLevel0(torch.rand(2, requires_grad=True))

        bar = torch.rand(2, requires_grad=True)

        print("prod between subclass and plain that requires grad")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)

        out = res.sum()
        self.assertIsNotNone(out.elem.grad_fn)
        # Should not need the .elem?
        out.elem.backward(torch.ones([]))
        print(foo.grad)
        self.assertIsNone(foo.grad)
        print(foo.elem.grad)
        self.assertEqual(foo.elem.grad, bar)
        print(bar.grad)
        self.assertIsNone(bar.grad)

        


if __name__ == '__main__':
    run_tests()
