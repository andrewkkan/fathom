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


from typing import Any, Callable, Mapping, Sequence, Tuple, Union

from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import optimizers
from fedjax.core import tree_util
from fedjax.core import models
from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params, PyTree
from fedjax.core.typing import PRNGKey
from fedjax.core.typing import OptState, BatchExample

import jax
import jax.numpy as jnp

import fathom

Grads = Params


def create_train_for_each_client(grad_fn: Params, client_optimizer: optimizers.Optimizer, model: models.Model):
    """Builds client_init, client_step, client_final for for_each_client."""

    def client_init(shared_input, client_rng):
        opt_state = client_optimizer.init(shared_input['params'])
        opt_state.hyperparams['learning_rate'] = shared_input['eta_c'] # Already sigmoided
        client_step_state = {
            'params': shared_input['params'],
            'params0': shared_input['params'],
            'opt_state': opt_state,
            'rng': client_rng,
            'step_idx': 0,
            'min_hypergrad': tree_util.tree_l2_norm(shared_input['params']), # Large enough value
            }
        return client_step_state

    def client_step(client_step_state, batch):
        rng, use_rng, metrics_rng = jax.random.split(client_step_state['rng'], num=3)
        grad_opt = grad_fn(client_step_state['params'], batch, use_rng)
        opt_state, params = client_optimizer.apply(grad_opt, client_step_state['opt_state'], client_step_state['params'])
        delta_params = jax.tree_util.tree_multimap(jnp.subtract, client_step_state['params0'], params)
        hypergrad = fathom.core.tree_util.tree_dot(grad_opt, delta_params) 
        min_hypergrad = jnp.where(hypergrad > client_step_state['min_hypergrad'],
            client_step_state['min_hypergrad'], hypergrad
        )
        next_client_step_state = {
            'params': params,
            'params0': client_step_state['params0'],
            'opt_state': opt_state,
            'rng': rng,
            'step_idx': client_step_state['step_idx'] + 1,
            'min_hypergrad': min_hypergrad,
        }
        return next_client_step_state

    def client_final(shared_input, client_step_state) -> Tuple[Params, jnp.ndarray]:
        delta_params = jax.tree_util.tree_multimap(jnp.subtract, shared_input['params'], client_step_state['params'])
        return delta_params, client_step_state['min_hypergrad'], client_step_state['step_idx']

    return for_each_client.for_each_client(client_init, client_step, client_final)


def autoLip(
    data_dim: Mapping[str, Tuple[int]],
    params: Params, 
    model: models.Model,
) -> Union[jnp.ndarray, None]:
    '''
    Based on AutoLip (Algo 2) from "Lipschitz regularity of deep neural networks", arxiv:1805.10965.
    Adapted to estimate Lipschitz of the gradient of the loss function vice the loss function.
    The resulting L is the smoothness factor for convergence.
    '''
    vl = {k: jnp.zeros(shape=v) for k, v in data_dim.items()}
    vl['x'] = None # Will regenerate
    def loss(params: Params, v: BatchExample): 
        preds = model.apply_for_eval(params, v)
        example_loss = model.train_loss(v, preds)
        return jnp.mean(example_loss)
        # return - tree_util.tree_l2_norm(preds) 
    grad_fn = jax.jit(jax.grad(loss))
    def grad_loss_l2_norm(v: BatchExample):
        grad_loss = grad_fn(params, v)
        return - tree_util.tree_l2_norm(grad_loss)
    grad_grad_fn = jax.jit(jax.grad(grad_loss_l2_norm))
    lip_list = []
    for run in range(10):
        # This outter loop should be parallelized using vmap or something, and jitted.
        vl['x'] = jax.random.normal(key=jax.random.PRNGKey(17), shape=data_dim['x']) * 0.15 + 0.5
        for idx in range(50): # Rough L estimation
            vl = grad_grad_fn(vl)
            if float(tree_util.tree_l2_norm(vl['x'])) == 0.0:  # Numerical instability where vl lies outside of where gradients are available
                break
            else:
                vl['x'] = tree_util.tree_inverse_weight(vl['x'], tree_util.tree_l2_norm(vl['x']))
        if float(tree_util.tree_l2_norm(vl['x'])) > 0.0:
            lip_list.append(tree_util.tree_l2_norm(grad_fn(params, vl)))
    if lip_list:
        lip_sum = jnp.array(0.)
        for v in lip_list:
            lip_sum = lip_sum + v
        return lip_sum / len(lip_list)
    else:
        return None


@dataclasses.dataclass
class Hyperparams:
    eta_c: float    # Local learning rate or step size
    tau: float      # Number of local steps = ceil(local_samples / batch_size) * tau, where tau ~ num epochs worth of data.
    bs: float       # Local batch size
    alpha: float    # Momentum for glob grad estimation


@dataclasses.dataclass
class MetaState:
    hyperparams: Hyperparams 
    opt_state: optimizers.OptState
    opt_param: jnp.ndarray
    phase: int # TBD
    hypergrad_glob: float
    hypergrad_local: float


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
    lipschitz_ub: float
    grad_glob: Params
    mean_victor: float


@jax.jit
def estimate_grad_glob(server_state: ServerState, mean_delta_params: Params) -> Params:
    grad_glob = tree_util.tree_weight(server_state.grad_glob, server_state.meta_state.hyperparams.alpha)
    delta_params = tree_util.tree_weight(mean_delta_params, 1. - server_state.meta_state.hyperparams.alpha)
    grad_glob = tree_util.tree_add(grad_glob, delta_params)
    grad_glob = fathom.core.tree_util.tree_inverse_weight(
        grad_glob, 
        (1. - server_state.meta_state.hyperparams.alpha ** server_state.round_index)
    )
    return grad_glob


def federated_averaging(
        grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
        client_optimizer: optimizers.Optimizer,
        server_optimizer: optimizers.Optimizer,
        hyper_optimizer: optimizers.Optimizer,
        client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
        server_init_hparams: Hyperparams,
        model: models.Model,
        data_dim: Mapping[str, Tuple[int]],
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
    train_for_each_client = create_train_for_each_client(grad_fn, client_optimizer, model)

    def init(params: Params) -> ServerState:
        opt_state_server = server_optimizer.init(params)
        opt_state_client = client_optimizer.init(params) # Just to access hyperparams for eta_c
        opt_state_hyper = hyper_optimizer.init(server_init_hparams.tau)
        meta_state = MetaState(
            hyperparams = server_init_hparams,
            opt_state = opt_state_hyper,
            opt_param = jnp.array(server_init_hparams.tau), # Phase 1 starts with tau optim
            phase = 1,
            hypergrad_glob = 0.,
            hypergrad_local = 0.,
        )
        # Need to initialize round_index to 1 for bias comp
        return ServerState(
            params = params, 
            opt_state = opt_state_server, 
            round_index = 1, 
            lipschitz_ub = 0.0,
            grad_glob = tree_util.tree_zeros_like(params),
            meta_state = meta_state,
            mean_victor = 0.0,
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
        tau: float = server_state.meta_state.hyperparams.tau
        bs: int = int(server_state.meta_state.hyperparams.bs)
        batch_clients = [(cid, cds.shuffle_repeat_batch(
            client_datasets.ShuffleRepeatBatchHParams(
                batch_size = bs, 
                num_steps = int(max(1.0, jnp.ceil(tau * client_num_examples[cid] / bs))), 
                num_epochs = None, # This is required.  See ShuffleRepeatBatchView implementation in fedjax.core.client_datasets.py.
                drop_remainder = client_batch_hparams.drop_remainder,
                seed = client_batch_hparams.seed,
                skip_shuffle = client_batch_hparams.skip_shuffle,
            )
        ), crng) for cid, cds, crng in clients]
        shared_input = {'params': server_state.params, 'eta_c': server_state.meta_state.hyperparams.eta_c, 'tau': tau, 'bs': bs}
        client_diagnostics = {}
        # Running weighted mean of client updates. We do this iteratively to avoid
        # loading all the client outputs into memory since they can be prohibitively
        # large depending on the model parameters size.
        delta_params_sum = tree_util.tree_zeros_like(server_state.params)
        delta_params_seq: PyTree = []
        hypergrad_local_sum = jnp.array(0.0)
        num_examples_sum = 0.
        for client_id, (delta_params, min_hypergrad, num_steps) in train_for_each_client(shared_input, batch_clients):
            # Server collecting stats before sending them to server_update for updating params and metrics.
            num_examples = client_num_examples[client_id]
            weighted_delta_params =  tree_util.tree_weight(delta_params, num_examples)
            delta_params_sum = tree_util.tree_add(delta_params_sum, weighted_delta_params)
            delta_params_seq.append(weighted_delta_params) 
            hypergrad_local_sum = hypergrad_local_sum + min_hypergrad * num_examples
            num_examples_sum += num_examples
            # We record the l2 norm of client updates as an example, but it is not
            # required for the algorithm.
            client_diagnostics[client_id] = {
                    'delta_l2_norm': tree_util.tree_l2_norm(delta_params),
            }
            print(f'client_id = {client_id}, num_steps taken = {num_steps}')
        mean_delta_params = fathom.core.tree_util.tree_inverse_weight(delta_params_sum, num_examples_sum)
        mean_hypergrad_local = hypergrad_local_sum / num_examples_sum
        delta_params_seq = [fathom.core.tree_util.tree_inverse_weight(dp, num_examples_sum) for dp in delta_params_seq]
        # Expected value of Vk, as defined in "Local SGD: Unified Theory..." by Gorbunov (arxiv:2011.02828)
        # See section 2 and Lemma G.4 (81) for details.
        victor_sum = jnp.array(0.)
        for dp in delta_params_seq:
            victor_sum = victor_sum + jnp.square(tree_util.tree_l2_norm(
                jax.tree_util.tree_multimap(jnp.subtract, dp, mean_delta_params), 
            )) * num_examples 
        mean_victor = victor_sum / jnp.array(num_examples_sum) 
        server_state = server_update(server_state, mean_delta_params, mean_hypergrad_local, mean_victor)
        return server_state, client_diagnostics

    def server_update(
        server_state: ServerState, 
        mean_delta_params: Params, 
        mean_hypergrad_local: jnp.ndarray, 
        mean_victor: jnp.ndarray,
    ) -> ServerState:
        opt_state_server, params = server_optimizer.apply(
            mean_delta_params, 
            server_state.opt_state, 
            server_state.params,
        )
        lipschitz_ub: Optional[jnp.ndarray] = autoLip(data_dim, params, model)
        print(f"LIP = {lipschitz_ub}")
        if lipschitz_ub is None:
            lipschitz_ub = max(server_state.lipschitz_ub, 2.0)
        grad_glob: Params = estimate_grad_glob(server_state, mean_delta_params)
        meta_state: MetaState = hyper_update(
            server_state = server_state, 
            params = params, 
            lipschitz_ub = lipschitz_ub, 
            delta_params = mean_delta_params,
            hypergrad_local = mean_hypergrad_local,
            mean_victor = mean_victor,
        )
        return ServerState(
            params = params,
            opt_state = opt_state_server,
            round_index = server_state.round_index + 1,
            meta_state = meta_state,
            lipschitz_ub = lipschitz_ub,
            grad_glob = grad_glob,
            mean_victor = mean_victor,
        )

    def hyper_update(
        server_state: ServerState,
        params: Params,
        lipschitz_ub: jnp.ndarray,
        delta_params: Params,
        hypergrad_local: jnp.ndarray,
        mean_victor: jnp.ndarray,
    ) -> MetaState:

        grad_glob = server_state.grad_glob # Do not use the most current grad_glob as the result will bias positive
        hypergrad_glob: float = fathom.core.tree_util.tree_dot(grad_glob, delta_params)
        hypergrad = -hypergrad_local
        opt_state, opt_param = hyper_optimizer.apply(hypergrad, server_state.meta_state.opt_state, server_state.meta_state.opt_param)

        phase = server_state.meta_state.phase
        if phase == 1:
            eta_c = float(0.25 / lipschitz_ub)
            tau = float(jax.nn.relu(opt_param))
            # Need condtion for phase transition
        elif phase == 2:
            eta_c = jax.nn.sigmoid(opt_param)
            tau = server_state.meta_state.hyperparams.tau
            # Need condtion for phase transition
        elif phase == 3:
            eta_c = jax.nn.sigmoid(opt_param)
            tau = 0.0

        hyperparams = Hyperparams(
            eta_c = eta_c,
            tau = tau,
            bs = server_state.meta_state.hyperparams.bs,
            alpha = server_state.meta_state.hyperparams.alpha, 
        )
        meta_state = MetaState(
            hyperparams = hyperparams,
            opt_state = opt_state,
            opt_param = opt_param,
            phase = phase,
            hypergrad_glob = hypergrad_glob,
            hypergrad_local = hypergrad_local,
        )        
        return meta_state

    return federated_algorithm.FederatedAlgorithm(init, apply)
