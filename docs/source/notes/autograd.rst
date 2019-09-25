Autograd mechanics
==================

This note will present an overview of how autograd works and records the
operations. It's not strictly necessary to understand all this, but we recommend
getting familiar with it, as it will help you write more efficient, cleaner
programs, and can aid you in debugging.


Autograd engine philosophy
^^^^^^^^^^^^^^^^^^^^^^^^^^

The main goal of the autograd in PyTorch is to provide gradients for Neural Networks.
To this end, it implements reverse-mode automatic differentiation also known as backpropagation.
The idea underlying backpropagation is that the derivative of a function composed of a set of elementary
subfunctions for which the derivatives are known can be computed using the chain rule.
For a function :math:`f(x) = g(h(x))`, :math:`df(x) / dx = df(x) / dh(x) * dh(x) / dx = dg(y) / dy * dh(x) / dx`.
As can be seen, only the derivatives of the elementary functions are required to compute the full derivative.
More informations can be found in the wikipedia article: https://en.wikipedia.org/wiki/Automatic_differentiation

To illustrate how the engine work, we will consider the graph for wich the nodes are these elementary functions
and the edges link the output of an elementary function to the input of another.
This graph is usually called the computational graph as it represents how computations are carried out as specified
in your code.
Because your program is finishing, this has to be a Directed Acyclic Graph.
To get the gradients of the overall function corresponding to this graph, we can traverse it backward, applying the chain
rule to each elementary function into it.

.. warning::
    The chain-rule is only mathematically correct when we work at points where **every** elementary function is differentiable.
    If this is not the case, there is no guarantee of the value returned by the engine.



How is the graph created
^^^^^^^^^^^^^^^^^^^^^^^^^^

We will first show a very simple example of the one to one correspondance between code and computational graph.
Consider the following piece of code:
.. code ::
    # a is a Tensor for which your want gradients
    b = 2 * a
    c = b + a

The computational graph that is created when running this code can be see in the graph below:
[some image]

More complex functions in your code can correspond to multiple elementary operations in the graph.
Note that arbitrary python function/class structure can be used here:

.. code ::
    def foo(t):
        b = 2 * a
        c = b + a
        return c

    # a is a Tensor for which your want gradients
    c = foo(a)
    # This will create the exact same graph as the previous example.
    # The autograd engine won't see the difference between the two.

Since the graph is created while your code is being run, you can use arbitrary python control flow and looping.
All operations performed on Tensor for which you require gradients will be recorded in the graph.

How to control how the graph is created
^^^^^^^^^^^^^^^^^^^^^^^^^^

requires_grad
"""""""""""""

Every Tensor has a flag: :attr:`requires_grad` that allows to specify whether gradients should be computed
for this Tensor or not.

If there's a single input to an operation that requires gradient, its output
will also require gradient. Conversely, only if all inputs don't require
gradient, the output also won't require it. Backward computation is never
performed in the subgraphs where all Tensors didn't require gradients.

.. code::

    >>> x = torch.randn(5, 5)  # requires_grad=False by default
    >>> y = torch.randn(5, 5)  # requires_grad=False by default
    >>> z = torch.randn((5, 5), requires_grad=True)
    >>> a = x + y
    >>> a.requires_grad
    False
    >>> b = a + z
    >>> b.requires_grad
    True

This is especially useful when you want to freeze part of your model, or you
know in advance that you're not going to use gradients w.r.t. some parameters.
For example if you want to finetune a pretrained CNN, it's enough to switch the
:attr:`requires_grad` flags in the frozen base, and no intermediate buffers will
be saved, until the computation gets to the last layer, where the affine
transform will use weights that require gradient, and the output of the network
will also require them.

.. code::

    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(512, 100)

    # Optimize only the classifier
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)



torch.no_grad
"""""""""""""

If you need to work with Tensors that have `requires_grad=True` but do not want the operations to be part of the graph,
you can use the torch.no_grad context manager.
All operations performed inside this block will not be recorded into the computational graph.

This is especially useful when you want to initialize the parameters of your model or perform weight updates with an optimizer.
Indeed such operations change the values of the weights but should be ignored when computing gradients.
The `torch.nn.init` package for example use this to perform initialization of the different neural network weights.

Such block completely disable the autograd engine and so should also be used when running inference to reduce memory consumption
and reduce any overhead associated with the autograd.

detach()
""""""""

The last one is `bar = foo.detach()` that can be called on a Tensor.
This can be seen as creating a special node in the graph such that `foo` and `bar` have the same values but the gradient that will flow
from `bar` to `foo` will always be `0`.
In practice this means that `bar` is detached from the computational graph that generated `foo` and so is completely independant of it.

This should be used to get a Tensor that does not require gradients from a Tensor that requires gradients.
For example for logging purposes, a common use is to accumulate the loss function over an epoch:

.. code ::
    epoch_loss = 0.
    for batch in dataloader:
        # Compute the loss
        # Perform weight update
        epoch_loss += loss.detach() # Accumulate loss over the whole epoch




How to control which gradients are computed
^^^^^^^^^^^^^^

Two methods are available to perform the backward pass.

1. `.backward()` will populate the `.grad` attribute for every leaf Tensor in the computational graph. Leaf Tensors are the ones
that require gradients but don't have a parent in the graph. You can check this with the `.is_leaf` attribute of Tensors. Note that the
`is_leaf` is not a writeable property. To make sure a Tensor is not a leaf anymore, you should set `requires_grad=False`.

2. `autograd.grad(inputs, outputs)` will compute the gradients of the outputs wrt the specified inputs.


In-place operations with autograd
^^^^^^^^^^^^^^^

Supporting in-place operations in autograd is a hard matter, and we discourage
their use in most cases. Autograd's aggressive buffer freeing and reuse makes
it very efficient and there are very few occasions when in-place operations
actually lower memory usage by any significant amount. Unless you're operating
under heavy memory pressure, you might never need to use them.

The idea behind inplace operations is that they should behave exactly the same way as
out-of-place operations that work by creating a new entry in the computational graph for each operation
that is performed. In particular, the trick we use is to change the node that is registered as having created the
Tensor that was modified inplace. This is the only place in the code base where we actual change
the computational graph. This should not be visible and this should behave exactly as if the new node
modified a copy of the original Tensor. We accept only one difference here: the in-place version can raise
"A tensor required for gradient computation was modified inplace" when the out-of-place version would work.

In-place operations can potentially overwrite values required to compute
gradients. The engine performs correctness checks and will always raise an
error if this happens. Every tensor keeps a version counter, that is incremented every time it is
marked dirty (meaning modified in-place) in any operation. When a Function saves any tensors for backward,
a version counter of their containing Tensor is saved as well. Once you access
``self.saved_tensors`` it is checked, and if it is greater than the saved value
an error is raised. This ensures that if you're using in-place
functions and not seeing any errors, you can be sure that the computed
gradients are correct.


