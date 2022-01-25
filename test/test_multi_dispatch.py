import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._pytree import tree_map, tree_unflatten, tree_flatten

import contextlib
from typing import Iterator, List

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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

# General wrapper
class GeneralWrapper(WrapperTensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    def __init__(self, elem, *args, **kwargs):
        self.elem = elem

    @classmethod
    def get_wrapper_properties(cls, elem, *args, **kwargs):
        return elem, {}

    __torch_function__ = torch._C._disabled_torch_function_impl

    # Basic no-op implementation if needed
    @classmethod
    def _delegate_call(cls, func, types, args, kwargs, extra_args=tuple(), extra_kwargs=None):
        print(f"Delegating call for {cls.__name__} to {func.__module__}.{func.__name__}")
        if extra_kwargs is None:
            extra_kwargs = {}
        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            return cls(e, *extra_args, **extra_kwargs) if isinstance(e, torch.Tensor) else e

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))

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

# Backward AD

# Simplify (a lot) by assuming:
# - each Tensor is used once (Meaning that as soon as a gradient is computed, the next
#   node is ready to consume it.)
# - all node have a single output
# - no broadcasting

class Node():
    __slots__ = ['parents']
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def capture(self, inputs, outputs):
        pass

class LeafNode(Node):
    def __init__(self, var):
        self.var = var
        self.parents = None

    def apply(self, grad_outputs):
        assert len(grad_outputs) == 1
        grad = grad_outputs[0]
        curr_grad = getattr(self.var, "grad_v2", None)
        self.var.grad_v2 = grad if curr_grad is None else grad + curr_grad

class MulNode(Node):
    def capture(self, inputs, outputs):
        assert len(inputs) == 2
        self.inputs = inputs
    
    def apply(self, grad_outputs):
        assert len(grad_outputs) == 1
        grad = grad_outputs[0]
        return grad * self.inputs[1], grad * self.inputs[0]

class AddNode(Node):    
    def apply(self, grad_outputs):
        assert len(grad_outputs) == 1
        grad = grad_outputs[0]
        return grad, grad

class SumNode(Node):
    def capture(self, inputs, outputs):
        assert len(inputs) == 1
        self.size = inputs[0].size()
    
    def apply(self, grad_outputs):
        assert len(grad_outputs) == 1
        grad = grad_outputs[0]
        assert grad.ndimension() == 0
        res = grad.expand(self.size)
        return res

op_id_to_node = {
    "torch._ops.aten.mul": MulNode,
    "torch._ops.aten.add": AddNode,
    "torch._ops.aten.sum": SumNode,
}

def AD_run_backward(inp):
    assert inp.ndimension() == 0
    assert isinstance(inp, BackwardADTensor)
    root = inp.node

    queue = [(root, torch.tensor(1.))]
    while queue:
        node, grad_outputs = queue.pop()

        grad_inputs = node.apply((grad_outputs,))
        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)

        if node.parents is None:
            continue
        for next_node, grad in zip(node.parents, grad_inputs):
            queue.append((next_node, grad))

class BackwardADTensor(GeneralWrapper):
    def __init__(self, elem, *args, **kwargs):
        super().__init__(elem, *args, **kwargs)
        assert len(args) == 0
        assert len(kwargs) == 0 or (len(kwargs) == 1 and "requires_grad" in kwargs)
        if kwargs.get("requires_grad", False):
            self.node = LeafNode(self)
        else:
            self.node = None

    @staticmethod
    def get_node_from_func(func):
        # Some black magic here
        op_id = f"{func.__module__}.{func.__name__}"
        return op_id_to_node[op_id]()

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        parents = tuple(arg.node for arg in args if (isinstance(arg, BackwardADTensor) and arg.node is not None))
        requires_grad = len(parents) > 0

        res = cls._delegate_call(func, types, args, kwargs)

        is_tuple = True
        if not isinstance(res, tuple):
            res = (res,)
            is_tuple = False

        if requires_grad:
            assert len(res) == 1

            node = cls.get_node_from_func(func)
            node.parents = parents
            node.capture(args, res)

            for i, r in enumerate(res):
                if isinstance(r, BackwardADTensor):
                    r.node = node
        return res if is_tuple else res[0]


class TestMultiDispatch(TestCase):
    def test_base_print(self):
        print_list = []
        foo = PrintingTensor(torch.rand(2), print_list)

        b = foo * foo
        b += 2
        b.sum()

        expected_print = ['torch._ops.aten.mul', 'torch._ops.aten.add_', 'torch._ops.aten.sum']
        self.assertEqual(print_list, expected_print)


    def test_base_backward(self):
        foo = BackwardADTensor(torch.rand(2), requires_grad=True)
        foo2 = BackwardADTensor(torch.rand(2), requires_grad=True)

        bar = foo * 2
        bar = bar + foo2
        loss = bar.sum()

        AD_run_backward(loss)

        self.assertEqual(foo.grad_v2, torch.tensor([2., 2.]))
        self.assertEqual(foo2.grad_v2, torch.tensor([1., 1.]))

    def test_mixing(self):
        print_list = []
        foo = BackwardADTensor(torch.rand(2), requires_grad=True)
        bar = PrintingTensor(torch.rand(2), print_list)

        res = torch.mul(foo, bar)


if __name__ == '__main__':
    run_tests()
