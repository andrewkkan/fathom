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


from typing import Any, Callable, Mapping, Sequence, Tuple, Union, Optional

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




@dataclasses.dataclass
class FathomParams:
    # update_type can be one of the following:
    # HPL = Hypergradient Linear
    # HPM = Hypergradient Multiplicative 
    # EGU = Exponentiated Gradient Unnormalized
    # EGN = Exponentiated Gradient Normalized
    update_type: str            # HPL, HPM, EGU, or EGN

def create_train_for_each_client(grad_fn: Params, client_optimizer: optimizers.Optimizer, model: models.Model, fathom_params: FathomParams):
    """Builds client_init, client_step, client_final for for_each_client."""
    normalize_hypergrad = 0 if fathom_params is None or 'HPL' in fathom_params.update_type or \
        'EGU' in fathom_params.update_type else 1

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
        grad_opt_norm = tree_util.tree_l2_norm(grad_opt)
        delta_params_norm = tree_util.tree_l2_norm(delta_params)
        hypergrad = jnp.where(grad_opt_norm == 0., jnp.array(0),
            jnp.where(delta_params_norm == 0, jnp.array(0),
                jnp.where(normalize_hypergrad == 0 ,
                    fathom.core.tree_util.tree_dot(grad_opt, delta_params),
                    fathom.core.tree_util.tree_dot(grad_opt, delta_params) / grad_opt_norm / delta_params_norm)
            )
        )
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


@dataclasses.dataclass
class AutoLipParams:
    use: bool
    lamb: float

def autoLip(
    params: Params, 
    model: models.Model,
    lamb: float,
    img_dim: Optional[Mapping[str, Tuple[int]]] = None,
    vocab_embed_size: Optional[Mapping[str, int]] = None,
) -> Union[jnp.ndarray, None]:
    '''
    Based on AutoLip (Algo 2) from "Lipschitz regularity of deep neural networks", arxiv:1805.10965.
    Adapted to estimate Lipschitz of the gradient of the loss function vice the loss function.
    The resulting L is the smoothness factor for convergence.
    '''
    if vocab_embed_size:
        if vocab_embed_size['vocab_size'] == 10000: # stackoverflow
            _model: models.Model = fathom.models.stackoverflow.create_lstm_model_without_embedding(
                vocab_size = vocab_embed_size['vocab_size'], 
                embed_size = vocab_embed_size['embed_size'], 
            )
        elif vocab_embed_size['vocab_size'] == 86: # shakespeare
            _model: models.Model = fathom.models.shakespeare.create_lstm_model_without_embedding(
                vocab_size = vocab_embed_size['vocab_size'], 
                embed_size = vocab_embed_size['embed_size'], 
            )
    elif img_dim:
        _model: models.Model = model
    def loss(params: Params, v: BatchExample): 
        preds = _model.apply_for_eval(params, v)
        example_loss = _model.train_loss(v, preds)
        return jnp.mean(example_loss)
    grad_fn = jax.jit(jax.grad(loss))
    def grad_loss_l2_norm(v_in: jnp.ndarray, v_out: jnp.ndarray):
        v = {'x': v_in, 'y': v_out}
        v0 = {'x': jnp.zeros_like(v_in), 'y': v_out}
        grad_loss = grad_fn(params, v)
        grad_loss0 = grad_fn(params, v0)
        return - tree_util.tree_l2_norm(jax.tree_util.tree_multimap(jnp.subtract, grad_loss, grad_loss0)) + lamb * jnp.sum(jnp.square(v_in))
    grad_grad_fn = jax.jit(jax.grad(grad_loss_l2_norm))
    lip_list = []
    lip_key = jax.random.PRNGKey(17)
    for run in range(10): # Number of Monte Carlo trials
        # This outter loop should be parallelized using vmap or something, and jitted.
        rng_key, lip_key = jax.random.split(lip_key)
        if img_dim:
            vx = jax.random.normal(key = rng_key, shape = img_dim['x']) * 0.15 + 0.5
            vy = jnp.zeros(shape = img_dim['y'])
        elif vocab_embed_size:
            batch_size = 16
            vx = jax.random.normal(key = rng_key, shape = (vocab_embed_size['max_length'], batch_size, vocab_embed_size['embed_size'])) * 0.5
            vy = jax.random.choice(key = rng_key, a = vocab_embed_size['vocab_size']+4, shape = (batch_size, vocab_embed_size['max_length']))
        for idx in range(50): # Rough L estimation
            vx = grad_grad_fn(vx, vy)
            if float(jnp.square(vx).sum()) == 0.0:  # Numerical instability where vl lies outside of where gradients are available
                break
            else:
                vx = tree_util.tree_inverse_weight(vx, tree_util.tree_l2_norm(vx))
        if float(jnp.square(vx).sum()) > 0.0:
            v = {'x': vx, 'y': vy}
            v0 = {'x': jnp.zeros_like(vx), 'y': vy}
            grad_loss = grad_fn(params, v)
            grad_loss0 = grad_fn(params, v0)
            lip_list.append(tree_util.tree_l2_norm(jax.tree_util.tree_multimap(jnp.subtract, grad_loss, grad_loss0)))
    if lip_list:
        lip_sum = jnp.array(0.)
        for v in lip_list:
            lip_sum = lip_sum + v
        return lip_sum / len(lip_list)
    else:
        return None


@dataclasses.dataclass
class HyperParams:
    eta_c: float                # Local learning rate or step size
    tau: float                  # Number of local steps = ceil(local_samples / batch_size) * tau, where tau ~ num epochs worth of data.
    bs: float                   # Local batch size
    alpha: float                # Momentum for glob grad estimation
    eta_h: jnp.ndarray          # Hyper optimizer learning rates for tau, eta_c, and bs, respectively
    hparam_ub: jnp.ndarray      # Upperbound vals for tau, eta_c, and bs, respectively.


@dataclasses.dataclass
class HyperState:
    hyperparams: HyperParams 
    init_hparams: HyperParams
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
    params_bak: Params
    opt_state: optimizers.OptState
    round_index: int
    hyper_state: HyperState
    grad_glob: Params


@jax.jit
def estimate_grad_glob(server_state: ServerState, mean_delta_params: Params) -> Params:
    grad_glob = tree_util.tree_weight(server_state.grad_glob, server_state.hyper_state.hyperparams.alpha)
    delta_params = tree_util.tree_weight(mean_delta_params, 1. - server_state.hyper_state.hyperparams.alpha)
    grad_glob = tree_util.tree_add(grad_glob, delta_params)
    grad_glob = fathom.core.tree_util.tree_inverse_weight(
        grad_glob, 
        (1. - server_state.hyper_state.hyperparams.alpha ** server_state.round_index)
    )
    return grad_glob


def federated_averaging(
        grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
        client_optimizer: optimizers.Optimizer,
        server_optimizer: optimizers.Optimizer,
        hyper_optimizer: optimizers.Optimizer,
        client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
        server_init_hparams: HyperParams,
        model: models.Model,
        img_dim: Optional[Mapping[str, Tuple[int]]] = None,
        vocab_embed_size: Optional[Mapping[str, int]] = None,
        autolip_params: Optional[AutoLipParams] = None,
        fathom_params: Optional[FathomParams] = None,
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
    train_for_each_client = create_train_for_each_client(grad_fn, client_optimizer, model, fathom_params)

    def server_reset(params: Params, init_hparams: HyperParams) -> ServerState:
        opt_state_server = server_optimizer.init(params)
        if fathom_params is None or 'HPL' in fathom_params.update_type or 'HPM' in fathom_params.update_type:
            opt_param_hyper = -jnp.log(init_hparams.hparam_ub / jnp.array([
                init_hparams.tau, init_hparams.eta_c, init_hparams.bs]) - 1) * init_hparams.hparam_ub
            opt_state_hyper = hyper_optimizer.init(opt_param_hyper)
        elif 'EGU' in fathom_params.update_type or 'EGN' in fathom_params.update_type:
            opt_param_hyper = jnp.log(jnp.array([init_hparams.tau, init_hparams.eta_c, init_hparams.bs]))
            opt_state_hyper = hyper_optimizer.init(opt_param_hyper)
        hyper_state = HyperState(
            hyperparams = init_hparams,
            init_hparams = init_hparams,
            opt_state = opt_state_hyper,
            opt_param = opt_param_hyper,
            phase = 1,
            hypergrad_glob = 0.,
            hypergrad_local = 0.,
        )
        # Need to initialize round_index to 1 for bias comp
        return ServerState(
            params = params, 
            params_bak = params,
            opt_state = opt_state_server, 
            round_index = 1, 
            grad_glob = tree_util.tree_zeros_like(params),
            hyper_state = hyper_state,
        )        

    def init(params: Params) -> ServerState:
        return server_reset(params, server_init_hparams)

    def apply(
        server_state: ServerState,
        clients: Sequence[Tuple[
            federated_data.ClientId, 
            client_datasets.ClientDataset, 
            PRNGKey
        ]],
    ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
        client_num_examples = {cid: len(cds) for cid, cds, _ in clients}
        tau: float = server_state.hyper_state.hyperparams.tau
        bs: int = int(server_state.hyper_state.hyperparams.bs + 0.5)
        eta_c: float = server_state.hyper_state.hyperparams.eta_c
        batch_clients = [(cid, cds.shuffle_repeat_batch(
            client_datasets.ShuffleRepeatBatchHParams(
                batch_size = bs if server_state.hyper_state.phase == 1 else client_num_examples[cid], 
                num_steps = int(jnp.ceil(tau * client_num_examples[cid] / bs)) if server_state.hyper_state.phase == 1 else 1, 
                num_epochs = None, # This is required.  See ShuffleRepeatBatchView implementation in fedjax.core.client_datasets.py.
                drop_remainder = client_batch_hparams.drop_remainder,
                seed = client_batch_hparams.seed,
                skip_shuffle = client_batch_hparams.skip_shuffle,
            )
        ), crng) for cid, cds, crng in clients]
        shared_input = {'params': server_state.params, 'eta_c': eta_c, 'tau': tau, 'bs': bs}
        client_diagnostics = {}
        # Running weighted mean of client updates. We do this iteratively to avoid
        # loading all the client outputs into memory since they can be prohibitively
        # large depending on the model parameters size.
        delta_params_sum = tree_util.tree_zeros_like(server_state.params)
        hypergrad_local_sum = jnp.array(0.0)
        num_examples_sum = 0.
        for client_id, (delta_params, min_hypergrad, num_steps) in train_for_each_client(shared_input, batch_clients):
            # Server collecting stats before sending them to server_update for updating params and metrics.
            num_examples = client_num_examples[client_id]
            weighted_delta_params =  tree_util.tree_weight(delta_params, num_examples)
            delta_params_sum = tree_util.tree_add(delta_params_sum, weighted_delta_params)
            hypergrad_local_sum = hypergrad_local_sum + min_hypergrad * num_examples
            num_examples_sum += num_examples
            # We record the l2 norm of client updates as an example, but it is not
            # required for the algorithm.
            client_diagnostics[client_id] = {
                'delta_l2_norm': tree_util.tree_l2_norm(delta_params),
            }
        mean_delta_params = fathom.core.tree_util.tree_inverse_weight(delta_params_sum, num_examples_sum)
        mean_hypergrad_local = hypergrad_local_sum / num_examples_sum
        server_state = server_update(
            server_state = server_state, 
            mean_delta_params = mean_delta_params, 
            hypergrad_local = mean_hypergrad_local, 
        )
        return server_state, client_diagnostics

    def server_update(
        server_state: ServerState, 
        mean_delta_params: Params, 
        hypergrad_local: jnp.ndarray, 
    ) -> ServerState:
        opt_state_server, params = server_optimizer.apply(
            mean_delta_params, 
            server_state.opt_state, 
            server_state.params,
        )
        if autolip_params is not None and autolip_params.use:
            autolip_out: Union[jnp.ndarray, None] = autoLip(
                params = params, 
                model = model, 
                lamb = autolip_params.lamb, 
                img_dim = img_dim, 
                vocab_embed_size = vocab_embed_size
            )
            print(f"LIP = {autolip_out}")
            if autolip_out is None:
                hyperparams = HyperParams(
                    eta_c = server_init_hparams.eta_c,
                    tau = server_init_hparams.tau,
                    bs = server_state.hyper_state.init_hparams.bs * 2.,
                    alpha = server_init_hparams.alpha,
                    eta_h = server_init_hparams.eta_h,
                    hparam_ub = server_init_hparams.hparam_ub,
                )
                return server_reset(server_state.params_bak, hyperparams)
        grad_glob: Params = estimate_grad_glob(server_state, mean_delta_params)
        hyper_state: HyperState = hyper_update(
            server_state = server_state, 
            params = params, 
            delta_params = mean_delta_params,
            hypergrad_local = hypergrad_local,
        )
        return ServerState(
            params = params,
            params_bak = server_state.params_bak, # Keep initial params rather than update
            opt_state = opt_state_server,
            round_index = server_state.round_index + 1,
            hyper_state = hyper_state,
            grad_glob = grad_glob,
        )

    def hyper_update(
        server_state: ServerState,
        params: Params,
        delta_params: Params,
        hypergrad_local: jnp.ndarray,
    ) -> HyperState:

        opt_param, opt_state = server_state.hyper_state.opt_param, server_state.hyper_state.opt_state
        # Do not use the most current grad_glob as the result will bias positive
        hypergrad_glob: float = fathom.core.tree_util.tree_dot(server_state.grad_glob, delta_params)
        if 'HPM' in fathom_params.update_type or 'EGN' in fathom_params.update_type:
            # Normalizing hypergrad_global here
            # hypergrad_local already normalized from local calculations
            grad_glob_norm = tree_util.tree_l2_norm(server_state.grad_glob)
            delta_params_norm = tree_util.tree_l2_norm(delta_params)
            hypergrad_glob = jnp.where((grad_glob_norm > 0. and delta_params_norm > 0.),
                hypergrad_glob / grad_glob_norm / delta_params_norm,
                0
            )
        hypergrad = - jnp.array([
            hypergrad_glob + hypergrad_local,   # tau
            hypergrad_glob,                     # eta_c
            -hypergrad_local,                   # bs
        ]) 
        if fathom_params is None or 'HPL' in fathom_params.update_type or 'HPM' in fathom_params.update_type:
            hypergrad = hypergrad * jax.nn.sigmoid(opt_param / server_state.hyper_state.hyperparams.hparam_ub) \
                * (1. - jax.nn.sigmoid(opt_param / server_state.hyper_state.hyperparams.hparam_ub))    # grad(sigmoid)
        # This is where individual learning rates are applied, assuming opt is SGD.
        # With any other opt, individual learning rates need to be set at opt instantiation.
        hypergrad = hypergrad * jnp.where(server_state.hyper_state.phase == 1, 
            server_state.hyper_state.hyperparams.eta_h, # This is an array of 3 eta_h's
            jnp.zeros_like(server_state.hyper_state.hyperparams.eta_h)
        )
        opt_state, opt_param = hyper_optimizer.apply(hypergrad, opt_state, opt_param)

        if fathom_params is None or 'HPL' in fathom_params.update_type:
            opt_state, opt_param = hyper_optimizer.apply(hypergrad, opt_state, opt_param)
            hparams_vals = jax.nn.sigmoid(opt_param / server_state.hyper_state.hyperparams.hparam_ub) \
                * server_state.hyper_state.hyperparams.hparam_ub
        elif 'HPM' in fathom_params.update_type:
            hparams_vals = jax.nn.sigmoid(opt_param / server_state.hyper_state.hyperparams.hparam_ub) \
                * server_state.hyper_state.hyperparams.hparam_ub
            hypergrad = hypergrad * hparams_vals
            opt_state, opt_param = hyper_optimizer.apply(hypergrad, opt_state, opt_param)
            hparams_vals = jax.nn.sigmoid(opt_param / server_state.hyper_state.hyperparams.hparam_ub) \
                * server_state.hyper_state.hyperparams.hparam_ub
        elif 'EGU' in fathom_params.update_type or 'EGN' in fathom_params.update_type:
            # EGN gradients are already normalized
            opt_state, opt_param = hyper_optimizer.apply(hypergrad, opt_state, opt_param)
            hparams_vals = jnp.exp(opt_param) # Convert back to linear from log scale
            hparams_vals = jnp.clip(hparams_vals, a_max = server_state.hyper_state.hyperparams.hparam_ub)

        phase = jnp.where(server_state.hyper_state.phase == 1,
            jnp.where(hparams_vals[0] > 0.5, 1, 2),
            2 
        )
        # Beware that hypergrads become really noisy or nonexistent during phase 2, so we may want to reduce reliance on them.
        tau = jnp.where(phase == 1, # if
            hparams_vals[0],
            0.0 # Else
        )
        eta_c = jnp.where(phase == 1, # if
            hparams_vals[1],
            jnp.where(server_state.hyper_state.phase == 1, # else-if
                server_state.hyper_state.hyperparams.eta_c * 0.5 / 0.25, # Transition value for eta_c
                server_state.hyper_state.hyperparams.eta_c * server_state.round_index / (server_state.round_index + 1)          
            )
        )
        bs = jnp.where(phase == 1,
            hparams_vals[2],
            server_state.hyper_state.hyperparams.bs * (server_state.round_index + 1) / server_state.round_index 
        )
        hyperparams = HyperParams(
            tau = tau,
            eta_c = eta_c,
            bs = bs,
            eta_h = server_state.hyper_state.hyperparams.eta_h,
            alpha = server_state.hyper_state.hyperparams.alpha, 
            hparam_ub = server_state.hyper_state.hyperparams.hparam_ub, 
        )
        hyper_state = HyperState(
            hyperparams = hyperparams,
            init_hparams = server_state.hyper_state.init_hparams,
            opt_state = opt_state,
            opt_param = opt_param,
            phase = phase,
            hypergrad_glob = hypergrad_glob,
            hypergrad_local = hypergrad_local,
        )        
        return hyper_state

    return federated_algorithm.FederatedAlgorithm(init, apply)
