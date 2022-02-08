from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from fedjax.core import dataclasses
from fedjax.core.metrics import Metric, MeanStat
from fedjax.core.typing import SingleExample
from fedjax.core.typing import SinglePrediction


def mean_squared_error_loss(orig: jnp.ndarray, output: jnp.ndarray) -> jnp.ndarray:
	return jnp.mean(jnp.square(jnp.ravel(orig) - jnp.ravel(output)))

@dataclasses.dataclass
class MeanSquaredErrorLoss(Metric):
  """Metric for MSE loss.

  Example::

    example = {'x': jnp.array([1.0, 1.0])}
    prediction = jnp.array([1.2, 0.4])
    metric = MeanSquaredErrorLoss()
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=0.2, weight=1) => 0.2

  Attributes:
    target_key: Key name in ``example`` for target.
    pred_key: Key name in ``prediction`` for unnormalized model output pred.
  """
  target_key: str = 'x'
  pred_key: Optional[str] = None

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes mean squared error loss for a single example.

    Args:
      example: One example with target in any shape.
      prediction: Reconstructed version in any shape with the same number of elements as target.

    Returns:
      MeanStat for loss for a single example.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    loss = mean_squared_error_loss(target, pred)
    return MeanStat.new(loss, 1.)
