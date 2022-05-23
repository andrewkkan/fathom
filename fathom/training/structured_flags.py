# Copyright 2021 Google LLC
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
"""Structured flags commonly used in experiment binaries.

Structured flags are often used to construct complex structures via multiple
simple flags (e.g. an optimizer can be created by controlling learning rate and
other hyper parameters).
"""

import sys
from typing import Optional, Sequence, Tuple

from absl import flags
import jax
import jax.numpy as jnp
import fedjax
from fedjax.core import optimizers
from fedjax.core import client_datasets
from fedjax.training.structured_flags import NamedFlags
import fathom
from fathom.algorithms.fathom_fedavg import HyperParams

FLAGS = flags.FLAGS


class OptimizerFlags(fedjax.training.structured_flags.OptimizerFlags):
    """Constructs a fathom.core.Optimizer from flags."""
    """Note: OptimizerFlags is being re-defined because the one from FedJax uses fedjax.core.optimizers,"""
    """but we need to instantiate with fathom.core.optimizers."""

    def get(self) -> optimizers.Optimizer:
        """Gets the specified optimizer."""
        optimizer_name = self._get_flag('optimizer')
        learning_rate = self._get_flag('learning_rate')
        if optimizer_name == 'sgd':
            return fathom.core.optimizers.sgd(learning_rate)
        elif optimizer_name == 'momentum':
            return fathom.core.optimizers.sgd(learning_rate, self._get_flag('momentum'))
        elif optimizer_name == 'adam':
            return fathom.core.optimizers.adam(learning_rate, self._get_flag('adam_beta1'),
                self._get_flag('adam_beta2'),
                self._get_flag('adam_epsilon'),
            )
        elif optimizer_name == 'rmsprop':
            return fathom.core.optimizers.rmsprop(learning_rate, 
                self._get_flag('rmsprop_decay'),
                self._get_flag('rmsprop_epsilon'),
            )
        elif optimizer_name == 'adagrad':
            return fathom.core.optimizers.adagrad(learning_rate, eps=self._get_flag('adagrad_epsilon'))
        elif optimizer_name == 'yogi':
            return fathom.core.optimizers.yogi(learning_rate, 
                self._get_flag('yogi_beta1'),
                self._get_flag('yogi_beta2'),
                self._get_flag('yogi_epsilon')
            )
        else:
            raise ValueError(f'Unsupported optimizer {optimizer_name!r} from '
                f'--{self._prefix}optimizer.')

class ShuffleRepeatBatchHParamsFlags(NamedFlags):
    """Constructs ShuffleRepeatBatchHParams from flags."""

    def __init__(self, name: Optional[str] = None, default_batch_size: int = 128, default_batch_seed: int = 123):
        super().__init__(name)
        defaults = client_datasets.ShuffleRepeatBatchHParams(batch_size=-1)
        # TODO(wuke): Support other fields.
        self._integer('batch_size', default_batch_size, 'Batch size')
        self._integer('batch_seed', default_batch_seed, 'Batch seed')

    def get(self):
        return client_datasets.ShuffleRepeatBatchHParams(
            batch_size=self._get_flag('batch_size'),
            seed=jax.random.PRNGKey(self._get_flag('batch_seed'))
        )


class FathomFlags(NamedFlags):
    """Constructs HyperParams and a fathom.optimizer from flags."""

    def __init__(self, name: Optional[str] = None, 
        default_learning_rate: float = 0.1, default_epochs: float = 1.0, default_batch_size: float = 16.0,
        default_alpha: float = 0.5, default_eta_h: float = 1.0,
        default_eta_h012: Tuple[float] = (0.01, 0.01, 0.1), default_ub: Tuple[float] = (10.0, 0.5, 5096),
    ):
        super().__init__(name)
        self._float('initial_learning_rate', default_learning_rate, 'Initial learning rate')
        self._float('initial_epochs', default_epochs, 'Initial epochs')
        self._float('initial_batch_size', default_batch_size, 'Initial batch size')
        self._float('alpha', default_alpha, 'Fathom alpha')
        self._float('eta_h', default_alpha, 'Fathom eta_h')
        self._float('eta_h0', default_alpha, 'Fathom eta_h0')
        self._float('eta_h1', default_alpha, 'Fathom eta_h1')
        self._float('eta_h2', default_alpha, 'Fathom eta_h2')
        self._float('Ep_ub', default_alpha, 'Fathom E_ub')
        self._float('eta_c_ub', default_alpha, 'Fathom eta_c_ub')
        self._float('bs_ub', default_alpha, 'Fathom bs_ub')

    def get(self) -> Tuple[optimizers.Optimizer, HyperParams, client_datasets.ShuffleRepeatBatchHParams]:
        fathom_opt = fathom.core.optimizers.sgd(learning_rate = self._get_flag('eta_h'))
        fathom_hparams = HyperParams(
            eta_c = float(self._get_flag('initial_learning_rate')),
            Ep = float(self._get_flag('initial_epochs')), # Initialize with 1 epoch's worth of data
            bs = float(self._get_flag('initial_batch_size')),
            alpha = float(self._get_flag('alpha')),
            eta_h = jnp.array([
                self._get_flag('eta_h0'), 
                self._get_flag('eta_h1'), 
                self._get_flag('eta_h2')
            ]),
            hparam_ub = jnp.array([
                self._get_flag('Ep_ub'),
                self._get_flag('eta_c_ub'),
                self._get_flag('bs_ub'),
            ]),
        )
        SRBatchHParams = client_datasets.ShuffleRepeatBatchHParams(
            batch_size = round(self._get_flag('initial_batch_size')),
            seed = jax.random.PRNGKey(FLAGS.client_batch_seed),
        )
        return fathom_opt, fathom_hparams, SRBatchHParams

