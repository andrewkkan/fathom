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
"""Training federated EMNIST via federated averaging.

- The model is a CNN with dropout
- The client optimizer is SGD

"""

# Remove the following 2 lines after fathom becomes an installable package
import sys
sys.path.append('./')


from absl import app
from absl import flags

import fedjax
from fedjax.core import models

import fathom
from fathom.algorithms import fathom_fedavg
from fathom.algorithms.fathom_fedavg import HyperParams, AutoLipParams, FathomParams

import jax
import jax.numpy as jnp

from typing import Optional
from jax.config import config


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'sim_rounds', 4001, 'Total number of rounds for sim run.')
flags.DEFINE_integer(
    'random_seed', 17, 'Random seed.')

flags.DEFINE_float(
    'alpha', 0.5, 'Momentum for global grad estimation.')

flags.DEFINE_float(
    'eta_h', 0.01, 'Init Hyper Learning Rate for all')
flags.DEFINE_float(
    'eta_h0', 1.0, 'Init Hyper Learning Rate for tau')
flags.DEFINE_float(
    'eta_h1', 1.0, 'Init Hyper Learning Rate for eta_c')
flags.DEFINE_float(
    'eta_h2', 0.0, 'Init Hyper Learning Rate for bs')

flags.DEFINE_float(
    'tau_ub', 10.0, 'Upperbound value for tau')
flags.DEFINE_float(
    'eta_c_ub', 0.1, 'Upperbound value for eta_c')
flags.DEFINE_float(
    'bs_ub', 20.0, 'Upperbound value for bs')

flags.DEFINE_float(
    'tau', 1.0, 'Init Num Epochs')
flags.DEFINE_float(
    'eta_c', 10**(-1.5), 'Init Client Learning Rate')
flags.DEFINE_integer(
    'batch_size', 5, 'Init Local Batch Size')

flags.DEFINE_integer(
    'clients_per_round', 10, 'Number of clients participating in federated learning in each round.')

flags.DEFINE_integer(
    'use_autolip', 0, 'Use AutoLip to detect fault in global model.  False means no fault detection.')
flags.DEFINE_float(
    'autolip_lambda', 0.0, 'Regularization factor for AutoLip.')

# hyper_update can be one of the following:
# HPL = Hypergradient Linear
# HPM = Hypergradient Multiplicative 
# EGU = Exponentiated Gradient Unnormalized
# EGN = Exponentiated Gradient Normalized
flags.DEFINE_string(
    'hyper_update', 'EGN', 'HPL, HPM, EGU, or EGN')
flags.register_validator('hyper_update',
                         lambda value: 'HPL' in value or 'HPM' in value or 'EGU' in value or 'EGN' in value,
                         message='--hyper_update must be HPL, HPM, EGU, or EGN')

def main(_):
    print(f"FLAGS values:")
    print(f"    --alpha {FLAGS.alpha}")
    print(f"    --eta_h {FLAGS.eta_h}")
    print(f"    --eta_h0 {FLAGS.eta_h0}")
    print(f"    --eta_h1 {FLAGS.eta_h1}")
    print(f"    --eta_h2 {FLAGS.eta_h2}")
    print(f"    --tau_ub {FLAGS.tau_ub}")
    print(f"    --eta_c_ub {FLAGS.eta_c_ub}")
    print(f"    --bs_ub {FLAGS.bs_ub}")
    print(f"    --tau {FLAGS.tau}")
    print(f"    --eta_c {FLAGS.eta_c}")
    print(f"    --batch_size {FLAGS.batch_size}")
    print(f"    --clients_per_round {FLAGS.clients_per_round}")
    print(f"    --use_autolip {FLAGS.use_autolip}")
    print(f"    --autolip_lambda {FLAGS.autolip_lambda}")
    print(f"    --hyper_update {FLAGS.hyper_update}")
    print(f"    --random_seed {FLAGS.random_seed}")
    # We only use TensorFlow for datasets, so we restrict it to CPU only to avoid
    # issues with certain ops not being available on GPU/TPU.
    # It does not affect operations other than datasets.
    fedjax.training.set_tf_cpu_only()

    # Load train and test federated data for EMNIST.
    train_fd, test_fd = fathom.datasets.cifar100.load_data() # train_fd.num_clients() returns 500, test_fc.num_clients() returns 100
    model: models.Model = fathom.models.cifar100.create_resnet18_model()

    # Scalar loss function with model parameters, batch of examples, and seed
    # PRNGKey as input.
    def loss(params, batch, rng):
        # `rng` used with `apply_for_train` to apply dropout during training.
        preds = model.apply_for_train(params, batch, rng)
        # Per example loss of shape [batch_size].
        example_loss = model.train_loss(batch, preds)
        # reg_loss = jnp.square(fedjax.tree_util.tree_l2_norm(params))
        return jnp.mean(example_loss) # + 0.05 * reg_loss

    # Gradient function of `loss` w.r.t. to model `params` (jitted for speed).
    grad_fn = jax.jit(jax.grad(loss))

    # Create federated averaging algorithm.
    client_optimizer = fathom.optimizers.sgd(learning_rate = FLAGS.eta_c)
    # server_optimizer = fedjax.optimizers.adam(
    #          learning_rate=10**(-1.5), b1=0.9, b2=0.999, eps=10**(-4))
    server_optimizer = fedjax.optimizers.sgd(learning_rate = 1.0) # Fed Avg
    hyper_optimizer = fedjax.optimizers.sgd(learning_rate = FLAGS.eta_h) # Individual learning rates are set separately.  SGD is required.
    # Hyperparameters for client local traing dataset preparation.
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(
        batch_size = FLAGS.batch_size, # Ideally this is adaptive and not necessary but batch_size is required.
        seed = jax.random.PRNGKey(FLAGS.random_seed),
    )
    server_init_hparams: HyperParams = HyperParams(
        eta_c = float(FLAGS.eta_c),
        tau = FLAGS.tau, # Initialize with 1 epoch's worth of data
        bs = float(FLAGS.batch_size),
        alpha = float(FLAGS.alpha),
        eta_h = jnp.array([FLAGS.eta_h0, FLAGS.eta_h1, FLAGS.eta_h2]),
        hparam_ub = jnp.array([FLAGS.tau_ub, FLAGS.eta_c_ub, FLAGS.bs_ub]),
    )
    autolip_params: AutoLipParams = AutoLipParams(
        use = True if FLAGS.use_autolip == 1 else False,
        lamb = FLAGS.autolip_lambda,
    )
    fathom_params: FathomParams = FathomParams(
        update_type = FLAGS.hyper_update,
    )
    data_dim = jax.tree_util.tree_map(lambda a: a[0:1].shape, test_fd.get_client(next(test_fd.client_ids())).all_examples())
    algorithm = fathom_fedavg.federated_averaging(
        grad_fn = grad_fn, 
        client_optimizer = client_optimizer,
        server_optimizer = server_optimizer,
        hyper_optimizer = hyper_optimizer,
        client_batch_hparams = client_batch_hparams,
        server_init_hparams = server_init_hparams,
        model = model,
        img_dim = data_dim,
        autolip_params = autolip_params,
        fathom_params = fathom_params,
    )

    # Initialize model parameters and algorithm server state.
    init_params = model.init(jax.random.PRNGKey(FLAGS.random_seed))
    server_state = algorithm.init(init_params)

    # Train and eval loop.
    train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
            fd=train_fd, num_clients=FLAGS.clients_per_round, seed=FLAGS.random_seed+1)
    test_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
            fd=test_fd, num_clients=FLAGS.clients_per_round, seed=FLAGS.random_seed+2)
    for round_num in range(1, FLAGS.sim_rounds):
        # Sample 10 clients per round without replacement for training.
        train_clients = train_client_sampler.sample()
        # Run one round of training on sampled clients.
        server_state, client_diagnostics = algorithm.apply(server_state, train_clients)
        print(f'[round {round_num}]')
        # Optionally print client diagnostics if curious about each client's model
        # update's l2 norm.
        # print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

        if round_num % 1 == 0:
            # Periodically evaluate the trained server model parameters.
            # Read and combine clients' train and test datasets for evaluation.
            train_client_ids = [cid for cid, _, _ in train_clients]
            train_eval_datasets = [cds for _, cds in train_fd.get_clients(train_client_ids)]
            test_clients = test_client_sampler.sample()
            test_client_ids = [cid for cid, _, _ in test_clients]
            test_eval_datasets = [cds for _, cds in test_fd.get_clients(test_client_ids)]
            train_eval_batches = fedjax.padded_batch_client_datasets(
                train_eval_datasets, 
                batch_size=256,
            )
            test_eval_batches = fedjax.padded_batch_client_datasets(
                test_eval_datasets, 
                batch_size=256,
            )

            # Run evaluation metrics defined in `model.eval_metrics`.
            train_metrics = fedjax.evaluate_model(
                model, 
                server_state.params,
                train_eval_batches
            )
            test_metrics = fedjax.evaluate_model(
                model, 
                server_state.params,
                test_eval_batches
            )
            print(f'[round {round_num}] train_metrics={train_metrics}')
            print(f'[round {round_num}] eta_c={server_state.hyper_state.hyperparams.eta_c}, tau={server_state.hyper_state.hyperparams.tau}, bs={server_state.hyper_state.hyperparams.bs}')
            print(f'[round {round_num}] hypergrad_glob={server_state.hyper_state.hypergrad_glob}, hypergrad_local={server_state.hyper_state.hypergrad_local}')
            print(f'[round {round_num}] test_metrics={test_metrics}')

    # Save final trained model parameters to file.
    fedjax.serialization.save_state(server_state.params, '/tmp/params')

if __name__ == '__main__':
    config.config_with_absl()
    app.run(main)
