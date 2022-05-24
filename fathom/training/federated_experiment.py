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

from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple

import jax.numpy as jnp
from fedjax.training.federated_experiment import EvaluationFn
from fedjax.core import federated_algorithm
from fathom.algorithms.fathom_fedavg import ServerState, HyperState, HyperParams

class FathomHyperParamsStatusFn(EvaluationFn):
	"""Simple class that checks on HyperParams online adaptation status 
	"""
	def __call__(self, 
		state: federated_algorithm.ServerState, round_num: int,
	) -> Mapping[str, jnp.ndarray]:
		serverstate: ServerState = state
		return {
			'Client learning rate': serverstate.hyper_state.hyperparams.eta_c,
			'Epochs': serverstate.hyper_state.hyperparams.Ep,
			'Batch size': serverstate.hyper_state.hyperparams.bs,
			'Hypergrad global normalized (H_bar)': serverstate.hyper_state.hypergrad_glob,
			'Hypergrad local normalized (G_bar)': serverstate.hyper_state.hypergrad_local,
		}