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
            assert self.elem.requires_grad is False # Not needed? Can reuse the graph?
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

    def __repr__(self):
        grad_str = f"grad_fn={self.grad_fn}" if self.grad_fn else f"requires_grad={self.requires_grad}"
        return super().__repr__([grad_str,])


last_bad_class = BackwardADTensor
last_bad_class_idx = -1
def grad(fn):
    def wrapped(x):
        global last_bad_class
        global last_bad_class_idx
        prev_last_class = last_bad_class
        last_bad_class_idx += 1
        Cls = type(f"BWLevel{last_bad_class_idx}", (last_bad_class,), {})
        last_bad_class = Cls

        with torch.enable_grad():
            inp = Cls(x, requires_grad=True)

            out = fn(inp)

            assert isinstance(out, torch.Tensor)
            assert out.nelement() == 1

            grad, = torch.autograd.grad(out.elem, inp.elem, allow_unused=True, create_graph=True)

        last_bad_class = prev_last_class
        last_bad_class_idx -= 1
        return grad
    return wrapped

class BatchedTensor(GeneralWrapper):
    def __init__(self, elem, *args, **kwargs):
        super().__init__(elem, *args, **kwargs)
        assert len(args) == 0
        assert len(kwargs) == 0

    @classmethod
    def get_wrapper_properties(cls, elem, *args, **kwargs):
        assert elem.dim() >= 1
        return elem, {"size": elem.size()[1:]}

    def __repr__(self):
        full_size = f"batch_size={self.elem.size(0)},size={self.size()}"
        return super().__repr__([full_size,])

    @classmethod
    def unwrap(cls, e):
        # Special logic here where we want to unwrap all of the Tensor of this type
        # while wrapping everything else to make sure they don't "polute" our autograd
        # handling
        if not isinstance(e, cls):
            return e
        else:
            return e.elem[cls.curr_idx]

    @classmethod
    def wrap(cls, extra_args, extra_kwargs, e):
        return e

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # use cls.curr_idx as hack to easily do the for-loop fallback
        assert not hasattr(cls, "curr_idx")

        batch_size = -1
        for arg in args:
            if isinstance(arg, cls):
                batch_size = arg.elem.size(0)
                break
            if isinstance(arg, list):
                for a in arg:
                    if isinstance(a, cls):
                        batch_size = a.elem.size(0)
                        break
        assert batch_size >= 0

        res = []
        for b in range(batch_size):
            setattr(cls, "curr_idx", b)
            res.append(cls._delegate_call(func, types, args, kwargs))
        delattr(cls, "curr_idx")

        res = torch.stack(res)
        return cls(res)

last_vmap_class = BatchedTensor
last_vmap_class_idx = -1
def vmap(fn):
    def wrapped(*inputs):
        global last_vmap_class
        global last_vmap_class_idx
        prev_last_class = last_vmap_class
        last_vmap_class_idx += 1
        Cls = type(f"VmapLevel{last_vmap_class_idx}", (last_vmap_class,), {})
        last_vmap_class = Cls

        inp = (Cls(x) for x in inputs)

        out = fn(*inp)

        assert isinstance(out, torch.Tensor)

        last_vmap_class = prev_last_class
        last_vmap_class_idx -= 1
        return out.elem
    return wrapped


class TestSubclassDispatch(TestCase):
    def test_base_batched(self):
        foo = BatchedTensor(torch.rand(2, 3))
        print(foo)

        out = foo.select(0, 1)
        print(out)

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
        foo = BWLevel0(torch.rand(2), requires_grad=True)
        bar = BWLevel0(torch.rand(2), requires_grad=True)

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
        out.elem.backward()
        print(foo.grad)
        self.assertIsNone(foo.grad)
        print(foo.elem.grad)
        self.assertEqual(foo.elem.grad, bar)
        print(bar.grad)
        self.assertIsNone(bar.grad)


    def test_bw_ad_multilvl(self):
        BWLevel0 = type("BWLevel0", (BackwardADTensor,), {})
        BWLevel1 = type("BWLevel1", (BWLevel0,), {})
        foo = BWLevel0(torch.rand(2), requires_grad=True)
        bar = BWLevel1(torch.rand(2), requires_grad=True)

        print("prod")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)
        out = res.sum()
        self.assertIsNotNone(out.elem.grad_fn)

        print("back of lvl1")
        # Should not need the .elem?
        out.elem.backward(torch.ones([]), retain_graph=True)
        print(foo.elem.grad)
        self.assertIsNone(foo.elem.grad)
        print(bar.elem.grad)
        self.assertEqual(bar.elem.grad, foo)

        foo.elem.grad = None
        bar.elem.grad = None
        print("back of lvl0")
        # Should not need the second .elem?
        out.elem.elem.backward()
        print(foo.elem.grad)
        self.assertEqual(foo.elem.grad, bar.elem)
        print(bar.elem.grad)
        self.assertIsNone(bar.elem.grad)

    def test_bw_ad_plain(self):
        # tl;dr doesn't work!
        BWLevel0 = type("BWLevel0", (BackwardADTensor,), {})
        foo = BWLevel0(torch.rand(2), requires_grad=True)

        bar = torch.rand(2, requires_grad=True)

        print("prod between subclass and plain that requires grad")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)

        out = res.sum()
        self.assertIsNotNone(out.elem.grad_fn)
        # Should not need the .elem?
        out.elem.backward()
        print(foo.grad)
        self.assertIsNone(foo.grad)
        print(foo.elem.grad)
        self.assertEqual(foo.elem.grad, bar)
        print(bar.grad)
        self.assertIsNone(bar.grad)

        
    def test_bw_ad_values(self):
        x = torch.rand([])
        out = grad(torch.sin)(x)
        self.assertEqual(out, torch.cos(x))

        x = torch.rand([])
        out = grad(grad(torch.sin))(x)
        self.assertEqual(out, -torch.sin(x))

    def test_conj(self):
        x = torch.tensor(1+1j)
        def foo(x):
            assert not x.is_conj()
            y = x.conj()
            assert y.elem.is_conj() # Not properly reflected on the wrapper class?
            return y
        res = grad(foo)(x)
        self.assertEqual(res, torch.ones_like(res))

    def test_indexing(self):
        def f2(value):
            value = value.clone()
            value[value > 0] = 0
            return value.sum()

        x = torch.randn(100)
        result = grad(f2)(x)
        self.assertEqual(result, (x <= 0).type_as(x))

    def test_constructor(self):
        def foo(x):
            return x * torch.tensor(2.)

        x = torch.tensor(3.14)
        grad(foo)(x)

    def test_no_grad(self):
        def f(x):
            with torch.no_grad():
                shift = x ** 2
            return x ** 2 - shift

        x = torch.randn([])
        y = grad(f)(x)
        self.assertEqual(y, 2 * x)
        x = torch.randn([])
        y = grad(grad(f))(x)

    def test_vmap_with_same_map_dim(self):
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        output = vmap(vmap(torch.mul))(x, y)
        self.assertEqual(output, x * y)

        output = vmap(vmap(vmap(torch.mul)))(x, y)
        self.assertEqual(output, x * y)

    def test_vmap_with_different_map_dim(self):
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        output = vmap(lambda x: vmap(lambda y: x * y)(y))(x)
        self.assertEqual(output.shape, (2, 5, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        z = torch.randn(7, 3)
        output = vmap(lambda x: vmap(lambda y: vmap(lambda z: x * y * z)(z))(y))(x)
        self.assertEqual(output.shape, (2, 5, 7, 3))
        self.assertEqual(output, x.view(2, 1, 1, 3) * y.view(5, 1, 3) * z)

if __name__ == '__main__':
    run_tests()
