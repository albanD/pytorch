import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._pytree import tree_map
from torch._C import _disabled_torch_function_impl

# General wrapper
class GeneralWrapper(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    # _skip_autograd = True
    __torch_function__ = _disabled_torch_function_impl

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # Make a subclass that doesn't have the autograd key and should not
        # be involved with autograd
        r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
        r.elem = elem
        return cls._post_new(r, args, kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.elem})"

    @staticmethod
    def _post_new(obj, args, kwargs):
        return r

    # Basic no-op implementation if needed
    @classmethod
    def _delegate_call(cls, func, types, args, kwargs, extra_args=tuple(), extra_kwargs=None):
        print(f"Entering call for {cls.__name__} to {func.__module__}.{func.__name__}")
        if extra_kwargs is None:
            extra_kwargs = {}
        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            return cls(e, *extra_args, **extra_kwargs) if isinstance(e, torch.Tensor) else e

        raw_args = tree_map(unwrap, args)
        raw_kwargs = tree_map(unwrap, kwargs)
        print(f"Delegating call for {cls.__name__} to {func.__module__}.{func.__name__}")
        print(args, kwargs)
        print(tuple(unpack_batched(a, 0) if is_batched(a) else a for a in raw_args ))
        print(raw_kwargs)
        raw_res = func(*raw_args, **raw_kwargs)
        print(f"Result for call for {cls.__name__} to {func.__module__}.{func.__name__}")
        # print(tuple(is_batched(a) for a in raw_res ))
        # print(tuple(unpack_batched(a, 0) if is_batched(a) else a for a in raw_res ))
        # print(raw_res)
        return tree_map(wrap, raw_res)

# Some printing tool
class PrintingTensor(GeneralWrapper):
    @staticmethod
    def _post_new(obj, args, kwargs):
        assert len(args) == 1
        assert len(kwargs) == 0
        assert isinstance(args[0], list)
        obj.printer = args[0]
        return obj

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        assert isinstance(args[0], PrintingTensor)
        printer = args[0].printer
        printer.append(f"{func.__module__}.{func.__name__}")

        return cls._delegate_call(func, types, args, kwargs, (printer,))

# Top Tensor has no autograd/batched. Wrapped Tensor does
class BackwardADTensor(GeneralWrapper):
    @staticmethod
    def _post_new(obj, args, kwargs):
        assert len(args) == 0
        assert len(kwargs) == 0 or (len(kwargs) == 1 and "requires_grad" in kwargs)
        if kwargs.get("requires_grad", False):
            assert obj.elem.requires_grad is False
            assert obj.elem.grad_fn is None
            obj.elem.requires_grad = True
        return obj

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        res = cls._delegate_call(func, types, args, kwargs)
        return res

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

class BatchedTensor(GeneralWrapper):
    _dim = -1

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # Make a subclass that doesn't have the autograd key and should not
        # be involved with autograd
        print("in")
        safe_print(elem)
        if is_batched(elem):
            base = torch._remove_batch_dim(elem, cls._dim, -1, cls._dim)
        else:
            base = elem
        base = base.select(cls._dim, 0)
        r = torch.Tensor._make_subclass(cls, base.to('meta'), elem.requires_grad)
        r.elem = elem
        return cls._post_new(r, args, kwargs)

    @staticmethod
    def _post_new(obj, args, kwargs):
        assert len(args) == 0
        assert len(kwargs) == 0
        assert obj._dim != -1
        if not is_batched(obj.elem):
            obj.elem = torch._add_batch_dim(obj.elem, obj._dim, 0)
        return obj

    def __repr__(self):
        return f"{self.__class__.__name__}(is batched? {is_batched(self)}/{is_batched(self.elem)}, {self.size(), self.elem.size()}, {unpack_batched(self.elem, self._dim).size()})"

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
        foo = BWLevel0(torch.rand(2), requires_grad=True)
        bar = BWLevel0(torch.rand(2), requires_grad=True)

        print("prod")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)

        bar = torch.rand(2, requires_grad=True)
        print("prod")
        print(foo)
        print(bar)
        res = foo * bar
        print(res)

        # BWLevel1 = type("BWLevel1", (BWLevel0,), {})
        # bar = BWLevel1(torch.rand(2), requires_grad=True)

        # print("prod")
        # print(foo)
        # print(bar)
        # res = foo * bar
        # print(res)

    def test_batched(self):
        Vmap0 = type("Vmap0", (BatchedTensor,), {"_dim": 0})
        # foo = Vmap0(torch.rand(2, 1))
        # bar = Vmap0(torch.rand(2, 1))

        # print(foo)
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


        Vmap1 = type("Vmap1", (Vmap0,), {"_dim": 0})
        foo = Vmap0(torch.rand(2, 3, 1))
        print(foo)

        bar = Vmap1(foo)
        print("baz", bar)
        safe_print(bar.elem)

        res = bar.exp()

        print("Exp", res)



if __name__ == '__main__':
    # run_tests()

    Vmap0 = type("Vmap0", (BatchedTensor,), {"_dim": 0})
    Vmap1 = type("Vmap1", (Vmap0,), {"_dim": 0})
    foo = Vmap0(torch.rand(2, 3, 1))
    print(foo)
    bool(torch.rand(10))

    bar = Vmap1(foo)
    print("baz", bar)
    safe_print(bar.elem)

    res = bar.exp()

    print("Exp", res)
    safe_print(res.elem)
    v = unpack_batched(res.elem, 0)
    print(v)
    ve = v.elem
    print("ah?")
    print(unpack_batched(ve, 0))
    def fn(ve):
        return ve != torch.ceil(ve)
    import dis
    print(dis.dis(fn))
    # fn(ve)
    # print(ve)


    print(bool(ve.exp()))

    print("ok!")
    input("goo")



