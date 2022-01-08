from typing import Callable, List, Optional, Tuple, Union
from fedjax.core.typing import Params
from fedjax.core.optimizers import Optimizer, ScalarOrSchedule, create_optimizer_from_optax
import optax

Grads = Params

def adagrad(learning_rate: ScalarOrSchedule,
            initial_accumulator_value: float = 0.1,
            eps: float = 1e-6) -> Optimizer:
    """The Adagrad optimizer.

    Adagrad is an algorithm for gradient based optimisation that anneals the
    learning rate for each parameter during the course of training.

    WARNING: Adagrad's main limit is the monotonic accumulation of squared
    gradients in the denominator: since all terms are >0, the sum keeps growing
    during training and the learning rate eventually becomes vanishingly small.

    References:
    [Duchi et al, 2011](https://jmlr.org/papers/v12/duchi11a.html)

    Args:
    learning_rate: This is a fixed global scaling factor.
    initial_accumulator_value: Initialisation for the accumulator.
    eps: A small constant applied to denominator inside of the square root (as
      in RMSProp) to avoid dividing by zero when rescaling.

    Returns:
    The corresponding `Optimizer`.
    """
    return create_optimizer_from_optax(
        optax.inject_hyperparams(optax.adagrad)(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        )
    )


def adam(learning_rate: ScalarOrSchedule,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-8,
         eps_root: float = 0.0) -> Optimizer:
    """The classic Adam optimiser.

    Adam is an SGD variant with learning rate adaptation. The `learning_rate`
    used for each weight is computed from estimates of first- and second-order
    moments of the gradients (using suitable exponential moving averages).

    References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

    Args:
    learning_rate: This is a fixed global scaling factor.
    b1: The exponential decay rate to track the first moment of past gradients.
    b2: The exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Adam.

    Returns:
    The corresponding `Optimizer`.
    """
    return create_optimizer_from_optax(
        optax.inject_hyperparams(optax.adam)(
            learning_rate=learning_rate, 
            b1=b1, 
            b2=b2, 
            eps=eps,
            eps_root=eps_root,
        )
    )

def rmsprop(learning_rate: ScalarOrSchedule,
            decay: float = 0.9,
            eps: float = 1e-8,
            initial_scale: float = 0.,
            centered: bool = False,
            momentum: Optional[float] = None,
            nesterov: bool = False) -> Optimizer:
    """A flexible RMSProp optimiser.

    RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
    used for each weight is scaled by a suitable estimate of the magnitude of the
    gradients on previous steps. Several variants of RMSProp can be found
    in the literature. This alias provides an easy to configure RMSProp
    optimiser that can be used to switch between several of these variants.

    References:
    [Tieleman and Hinton, 2012](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    [Graves, 2013](https://arxiv.org/abs/1308.0850)

    Args:
    learning_rate: This is a fixed global scaling factor.
    decay: The decay used to track the magnitude of previous gradients.
    eps: A small numerical constant to avoid dividing by zero when rescaling.
    initial_scale: Initialisation of accumulators tracking the magnitude of
      previous updates. PyTorch uses `0`, TF1 uses `1`. When reproducing results
      from a paper, verify the value used by the authors.
    centered: Whether the second moment or the variance of the past gradients is
      used to rescale the latest gradients.
    momentum: The `decay` rate used by the momentum term, when it is set to
      `None`, then momentum is not used at all.
    nesterov: Whether nesterov momentum is used.

    Returns:
    The corresponding `Optimizer`.
    """
    return create_optimizer_from_optax(
        optax.inject_hyperparams(optax.rmsprop)(
            learning_rate=learning_rate,
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
            centered=centered,
            momentum=momentum,
            nesterov=nesterov,
        )
    )

def sgd(learning_rate: ScalarOrSchedule,
        momentum: Optional[float] = None,
        nesterov: bool = False) -> Optimizer:
    """A canonical Stochastic Gradient Descent optimiser.

    This implements stochastic gradient descent. It also includes support for
    momentum, and nesterov acceleration, as these are standard practice when
    using stochastic gradient descent to train deep neural networks.

    References:
    [Sutskever et al, 2013](http://proceedings.mlr.press/v28/sutskever13.pdf)

    Args:
    learning_rate: This is a fixed global scaling factor.
    momentum: The `decay` rate used by the momentum term, when it is set to
      `None`, then momentum is not used at all.
    nesterov: Whether nesterov momentum is used.

    Returns:
    The corresponding `Optimizer`.
    """
    return create_optimizer_from_optax(
        optax.inject_hyperparams(optax.sgd)(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
        )
    )