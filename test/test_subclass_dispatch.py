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
        # del kwargs["device"]
        # wrapper = torch.Tensor._make_subclass(cls, torch.empty(size, device="meta", **kwargs))
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
        print(f"Delegating call for {cls.__name__} to {func.__module__}.{func.__name__} with {types}")
        if extra_kwargs is None:
            extra_kwargs = {}

        wrapper = functools.partial(cls.wrap, extra_args, extra_kwargs)

        ag = tree_map(cls.unwrap, args)
        # print(f" {func.__module__}.{func.__name__}:  {ag}!!!!")
        res = tree_map(wrapper, func(*ag, **tree_map(cls.unwrap, kwargs)))
        # print(f"{func.__module__}.{func.__name__} res {res}::::")
        return res

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

class ADTempWrapperTensor(GeneralWrapper):
    @classmethod
    def wrap(cls, extra_args, extra_kwargs, e):
        return e

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(t, ADTempWrapperTensor) for t in types):
            print(f"Inverted ADTempWrapperTensor order!: {types}")
            return NotImplemented

        return cls._delegate_call(func, types, args, kwargs)

class BackwardADTrackingTensor(GeneralWrapper):
    def __init__(self, elem, *args, **kwargs):
        super().__init__(elem, *args, **kwargs)
        assert len(args) == 0
        assert len(kwargs) == 0 or (len(kwargs) == 1 and "requires_grad" in kwargs)
        assert not kwargs.get("requires_grad", False)

    @classmethod
    def unwrap(cls, e):
        return e.elem if (isinstance(e, cls) or isinstance(e, ADTempWrapperTensor)) else e

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(t, BackwardADTrackingTensor) or issubclass(t, ADTempWrapperTensor) for t in types):
            raise RuntimeError(f"Bad types for tracking Tensor!: {types}")

        return cls._delegate_call(func, types, args, kwargs)

    def __repr__(self):
        grad_str = f"grad_fn={self.grad_fn}" if self.grad_fn else f"requires_grad={self.requires_grad}"
        return super().__repr__([grad_str,])

# Only the wrapped Tensor has autograd meta.
# This is necessary because if the wrapper has some, it would be handled before
# we get to the torch dispatch call
class BackwardADTensor(GeneralWrapper):
    def __init__(self, elem, *args, **kwargs):
        super().__init__(elem, *args, **kwargs)
        assert isinstance(elem, BackwardADTrackingTensor)
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
            return e.elem if isinstance(e, cls) else ADTempWrapperTensor(e, requires_grad=False)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return cls._delegate_call(func, types, args, kwargs)

def is_batched(t):
    try:
        if not torch.is_tensor(t):
            return False
        t.storage()
    except NotImplementedError as e:
        if "BatchedTensorImpl" in str(e):
            return True
    return False

def unpack_batched(t, out_bdim):
    res = torch._remove_batch_dim(t, 0, -1, out_bdim)
    assert not is_batched(res)
    return res

def safe_print(t):
    if is_batched(t):
        print(f"BatchedTensor({unpack_batched(t, 0)}")
    else:
        print(t)

class BatchedTempWrapperTensor(GeneralWrapper):
    @classmethod
    def wrap(cls, extra_args, extra_kwargs, e):
        return e

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(t, BatchedTempWrapperTensor) for t in types):
            print(f"Inverted BatchedTempWrapperTensor order!: {types}")
            return NotImplemented

        return cls._delegate_call(func, types, args, kwargs)

class BatchedTrackingTensor(GeneralWrapper):
    def __init__(self, elem, *args, **kwargs):
        super().__init__(elem, *args, **kwargs)
        assert len(args) == 0
        assert len(kwargs) == 0 or (len(kwargs) == 1 and "requires_grad" in kwargs)
        assert not kwargs.get("requires_grad", False)

    @classmethod
    def unwrap(cls, e):
        return e.elem if (isinstance(e, cls) or isinstance(e, BatchedTempWrapperTensor)) else e

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(t, BatchedTrackingTensor) or issubclass(t, BatchedTempWrapperTensor) for t in types):
            raise RuntimeError(f"Bad types for tracking Tensor!: {types}")

        return cls._delegate_call(func, types, args, kwargs)

    def __repr__(self):
        extra = [f"batched={is_batched(self)}",]
        if is_batched:
            extra.append(f"batch_size={unpack_batched(self, self._dim).size()}")
        return super().__repr__(extra)

class BatchedTensor(GeneralWrapper):
    _dim = -1

    def __init__(self, elem, *args, **kwargs):
        super().__init__(elem, *args, **kwargs)
        assert isinstance(elem, BatchedTrackingTensor)
        assert not is_batched(elem)
        assert len(args) == 0
        assert len(kwargs) == 0
        assert self._dim != -1
        self.elem = torch._add_batch_dim(self.elem, self._dim, 0)

    @classmethod
    def unwrap(cls, e):
        # Special logic here where we want to unwrap all of the Tensor of this type
        # while wrapping everything else to make sure they don't "polute" our vmap
        # handling
        if not isinstance(e, torch.Tensor):
            return e
        else:
            return e.elem if isinstance(e, cls) else BatchedTempWrapperTensor(e)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        res = cls._delegate_call(func, types, args, kwargs)
        return res


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
        foo = BWLevel0(BackwardADTrackingTensor(torch.rand(2)), requires_grad=True)
        bar = BWLevel0(BackwardADTrackingTensor(torch.rand(2)), requires_grad=True)

        print("prod between subclasses")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)

        bar = torch.rand(2)
        print("prod with plain Tensor")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)

        # Should not need the .elem?
        out = res.sum()
        out.elem.backward(torch.ones([]))
        print(foo.grad)
        print(foo.elem.grad)
        print(bar.grad)


    def test_bw_ad_multilvl(self):
        BWLevel0 = type("BWLevel0", (BackwardADTensor,), {})
        BWLevel1 = type("BWLevel1", (BWLevel0,), {})
        foo = BWLevel0(BackwardADTrackingTensor(torch.rand(2)), requires_grad=True)
        bar = BWLevel1(BackwardADTrackingTensor(torch.rand(2)), requires_grad=True)

        print("prod")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)
        out = res.sum()

        print("back of lvl0")
        # Should not need the .elem?
        out.elem.backward(torch.ones([]), retain_graph=True)
        print(foo.elem.grad)
        print(bar.elem.grad)

        print("back of lvl1")
        # Should not need the .elem?
        out.elem.elem.elem.backward(torch.ones([]))
        print(foo.elem.grad)
        print(bar.elem.grad)

    def test_bw_ad_plain(self):
        BWLevel0 = type("BWLevel0", (BackwardADTensor,), {})
        foo = BWLevel0(BackwardADTrackingTensor(torch.rand(2)), requires_grad=True)

        bar = torch.rand(2, requires_grad=True)

        print("prod between subclasses")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)

        out = res.sum()
        # Should not need the .elem?
        out.elem.backward(torch.ones([]))
        print(foo.grad)
        print(foo.elem.grad)
        print(bar.grad)

        # Should not need the .elem?
        out.elem.elem.backward(torch.ones([]))
        print(foo.grad)
        print(foo.elem.grad)
        print(bar.grad)

    def test_batched(self):
        Vmap0 = type("Vmap0", (BatchedTensor,), {"_dim": 0})
        foo = Vmap0(BatchedTrackingTensor(torch.rand(2, 1)))
        # bar = Vmap0(BatchedTrackingTensor(torch.rand(2, 1)))

        print(foo)
        # print(bar)

        # res = foo + bar
        # print(res)
        # out = unpack_batched(res.elem, 0)
        # print(out)

        # foo = Vmap0(torch.rand(2, 1))
        # bar = torch.rand(1)

        # print(foo)
        # print(bar)

        # res = foo + bar
        # print(res)
        # out = unpack_batched(res.elem, 0)
        # print(out)


        # Vmap1 = type("Vmap1", (Vmap0,), {"_dim": 0})
        # foo = Vmap0(torch.rand(2, 3, 1))
        # print(foo)

        # bar = Vmap1(foo)
        # print("baz", bar)
        # safe_print(bar.elem)

        # res = bar.exp()

        # print("Exp", res)



if __name__ == '__main__':
    run_tests()
