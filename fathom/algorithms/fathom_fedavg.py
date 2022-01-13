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
"""Federated averaging implementation using fedjax.core.

This is the more performant implementation that matches what would be used in
the :py:mod:`fedjax.algorithms.fed_avg` . The key difference between this and
the basic version is the use of :py:mod:`fedjax.core.for_each_client`

Communication-Efficient Learning of Deep Networks from Decentralized Data
        H. Brendan McMahan, Eider Moore, Daniel Ramage,
        Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
        https://arxiv.org/abs/1602.05629

Adaptive Federated Optimization
        Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush,
        Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan. ICLR 2021.
        https://arxiv.org/abs/2003.00295
"""

from typing import Any, Callable, Mapping, Sequence, Tuple

from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import optimizers
from fedjax.core import tree_util
from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

import jax
import jax.numpy as jnp

Grads = Params


def create_train_for_each_client(grad_fn, client_optimizer):
    """Builds client_init, client_step, client_final for for_each_client."""

    def client_init(shared_input, client_rng):
        opt_state = client_optimizer.init(shared_input['params'])
        opt_state.hyperparams['learning_rate'] = shared_input['eta_c'] # Already sigmoided
        client_step_state = {
            'params': shared_input['params'],
            'opt_state': opt_state,
            'rng': client_rng,
        }
        return client_step_state

    def client_step(client_step_state, batch):
        rng, use_rng = jax.random.split(client_step_state['rng'])
        grads = grad_fn(client_step_state['params'], batch, use_rng)
        opt_state, params = client_optimizer.apply(
            grads,
            client_step_state['opt_state'],
            client_step_state['params'],
        )
        next_client_step_state = {
                'params': params,
                'opt_state': opt_state,
                'rng': rng,
        }
        return next_client_step_state

    def client_final(shared_input, client_step_state):
        delta_params = jax.tree_util.tree_multimap(lambda a, b: a - b,
                                                   shared_input['params'],
                                                   client_step_state['params'])
        return delta_params

    return for_each_client.for_each_client(client_init, client_step, client_final)


@dataclasses.dataclass
class Hyperparams:
    eta_c: float    # Local learning rate or step size
    tau: float      # Number of local steps
    alpha: float    # Momentum for global gradient estimate
    bs: float       # Local batch size


@dataclasses.dataclass
class MetaState:
    grad_glob: Params 
    hyperparams: Hyperparams 
    opt_state: optimizers.OptState


@dataclasses.dataclass
class ServerState:
    """State of server passed between rounds.

    Attributes:
        params: A pytree representing the server model parameters.
        opt_state: A pytree representing the server optimizer state.
    """
    params: Params
    opt_state: optimizers.OptState
    round_index: int
    meta_state: MetaState


def federated_averaging(
        grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
        client_optimizer: optimizers.Optimizer,
        server_optimizer: optimizers.Optimizer,
        hyper_optimizer: optimizers.Optimizer,
        client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
        # eta_hyper: Hyperparams,  # Learning rates.  For statically set hyperparams, set learning rates to 0.
        server_init_hparams: Hyperparams,
) -> federated_algorithm.FederatedAlgorithm:
    """Builds federated averaging.

    Args:
        grad_fn: A function from (params, batch_example, rng) to gradients.
            This can be created with :func:`fedjax.core.model.model_grad`.
        client_optimizer: Optimizer for local client training.
        server_optimizer: Optimizer for server update.
        client_batch_hparams: Hyperparameters for batching client dataset for train.

    Returns:
        FederatedAlgorithm
    """
    train_for_each_client = create_train_for_each_client(grad_fn, client_optimizer)

    def init(params: Params) -> ServerState:
        opt_state_server = server_optimizer.init(params)
        opt_state_client = client_optimizer.init(params) # Just to access hyperparams for eta_c
        assert(server_init_hparams.eta_c == opt_state_client.hyperparams['learning_rate'])
        assert(server_init_hparams.tau == float(client_batch_hparams.num_steps))
        assert(server_init_hparams.bs == float(client_batch_hparams.batch_size))
        hyperparams = Hyperparams(
            eta_c = -jnp.log(1. / server_init_hparams.eta_c - 1.),
            tau = float(server_init_hparams.tau) - 1.,
            alpha = server_init_hparams.alpha,
            bs = float(server_init_hparams.bs) - 1.,
        )
        opt_state_hyper = hyper_optimizer.init(hyperparams)
        meta_state = MetaState(
            grad_glob = tree_util.tree_zeros_like(params), # grad_glob 
            hyperparams = hyperparams,
            opt_state = opt_state_hyper,
        )
        # Need to initialize round_index to 1 for bias comp
        return ServerState(
            params = params, 
            opt_state = opt_state_server, 
            round_index = 1, 
            meta_state = meta_state,
        )

    def apply(
        server_state: ServerState,
        clients: Sequence[Tuple[
            federated_data.ClientId, 
            client_datasets.ClientDataset, 
            PRNGKey
        ]],
    ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
        client_num_examples = {cid: len(cds) for cid, cds, _ in clients}
        current_client_batch_hparams = client_datasets.ShuffleRepeatBatchHParams(
            batch_size = int(jax.nn.relu(server_state.meta_state.hyperparams.bs)+1.5), # In practice this is pushed to clients
            num_steps = int(jax.nn.relu(server_state.meta_state.hyperparams.tau)+1.5), # In practice this is pushed to clients 
            num_epochs = client_batch_hparams.num_epochs,
            drop_remainder = client_batch_hparams.drop_remainder,
            seed = client_batch_hparams.seed,
            skip_shuffle = client_batch_hparams.skip_shuffle,
        )
        batch_clients = [(cid, cds.shuffle_repeat_batch(current_client_batch_hparams), crng)
                                         for cid, cds, crng in clients]
        shared_input = {'params': server_state.params, 'eta_c': jax.nn.sigmoid(server_state.meta_state.hyperparams.eta_c)}
        client_diagnostics = {}
        # Running weighted mean of client updates. We do this iteratively to avoid
        # loading all the client outputs into memory since they can be prohibitively
        # large depending on the model parameters size.
        delta_params_sum = tree_util.tree_zeros_like(server_state.params)
        num_examples_sum = 0.
        for client_id, delta_params in train_for_each_client(shared_input, batch_clients):
            num_examples = client_num_examples[client_id]
            delta_params_sum = tree_util.tree_add(
                    delta_params_sum, tree_util.tree_weight(delta_params, num_examples))
            num_examples_sum += num_examples
            # We record the l2 norm of client updates as an example, but it is not
            # required for the algorithm.
            client_diagnostics[client_id] = {
                    'delta_l2_norm': tree_util.tree_l2_norm(delta_params)
            }
        mean_delta_params = tree_util.tree_inverse_weight(delta_params_sum, num_examples_sum)
        server_state = server_update(server_state, mean_delta_params)
        return server_state, client_diagnostics

    def server_update(server_state, mean_delta_params):
        opt_state, params = server_optimizer.apply(
            mean_delta_params, 
            server_state.opt_state, 
            server_state.params,
        )
        gradsum_inst = tree_util.tree_inverse_weight(mean_delta_params, jax.nn.sigmoid(server_state.meta_state.hyperparams.eta_c))
        alpha = jax.nn.relu(server_state.meta_state.hyperparams.alpha)
        grad_glob = jax.tree_util.tree_multimap(lambda a, b: (
                (a * alpha + b * (1. - alpha)) #/ # grad_glob with momentum noise filtering
                #(1. - alpha ** server_state.round_index) # bias comp
            ),
            server_state.meta_state.grad_glob,
            gradsum_inst,
        )
        hyper_optate, hyperparams = hyper_optimizer.apply(
            hyper_step(server_state, gradsum_inst, grad_glob), 
            server_state.meta_state.opt_state, 
            server_state.meta_state.hyperparams,
        )
        meta_state = MetaState(
            grad_glob = grad_glob, 
            hyperparams = hyperparams,
            opt_state = hyper_optate,
        )
        return ServerState(params, opt_state, server_state.round_index + 1, meta_state)

    def hyper_step(
        server_state: ServerState,
        gradsum_inst: Params,
        grad_glob: Params,
    ) -> Hyperparams:

        sigmoid_prime = jax.grad(jax.nn.sigmoid)(server_state.meta_state.hyperparams.eta_c)
        cossim = jax.tree_util.tree_multimap(lambda a, b: a * -b, grad_glob, gradsum_inst)
        cossim_flatten = jax.flatten_util.ravel_pytree(cossim)
        cossim_agg = jnp.sum(cossim_flatten[0])
        eta_c_step = sigmoid_prime * cossim_agg 

        relu_prime = jax.grad(jax.nn.relu)(server_state.meta_state.hyperparams.tau)
        tau_step = relu_prime * 0.0 

        hyper_step = Hyperparams(eta_c = eta_c_step, tau = tau_step, alpha = 0.0, bs = 0.0)
        return hyper_step

    return federated_algorithm.FederatedAlgorithm(init, apply)
