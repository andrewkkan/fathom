# Copyright 2022 FATHOM Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extension to fedjax.core.tree_util

Utilities for working with tree-like container data structures.

In JAX, the term pytree refers to a tree-like structure built out of
container-like Python objects.
For more details, see https://jax.readthedocs.io/en/latest/pytrees.html.
"""

from typing import Iterable, Tuple

from fedjax.core.typing import PyTree
from fedjax.core.tree_util import tree_weight

import jax
import jax.numpy as jnp


@jax.jit
def tree_dot(left: PyTree, right: PyTree) -> float:
  """Returns squared l2 norm of tree."""
  return sum(jnp.vdot(l, r) for l, r in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)))


@jax.jit
def tree_inverse_weight(pytree: PyTree, weight: float) -> PyTree:
  """Weights tree leaves by ``1 / weight``."""
  inverse_weight = jnp.where(weight > 0., (1. / weight), 0.)
  return tree_weight(pytree, inverse_weight)