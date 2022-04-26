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
"""Training federated Stack Overflow via federated averaging.

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
from fedjax.algorithms import fed_avg

import jax
import jax.numpy as jnp

from typing import Optional
from jax.config import config


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'sim_rounds', 1501, 'Total number of rounds for sim run.')
flags.DEFINE_integer(
    'random_seed', 17, 'Random seed.')

flags.DEFINE_float(
    'eta_c', 10**(-0.5), 'Init Client Learning Rate')
flags.DEFINE_integer(
    'batch_size', 16, 'Init Local Batch Size')
flags.DEFINE_float(
    'tau', 1.0, 'Init Num Epochs')

flags.DEFINE_integer(
    'clients_per_round', 50, 'Number of clients participating in federated learning in each round.')

def main(_):
    print(f"FLAGS values:")
    print(f"    --tau {FLAGS.tau}")
    print(f"    --eta_c {FLAGS.eta_c}")
    print(f"    --batch_size {FLAGS.batch_size}")
    print(f"    --clients_per_round {FLAGS.clients_per_round}")
    print(f"    --random_seed {FLAGS.random_seed}")
    # We only use TensorFlow for datasets, so we restrict it to CPU only to avoid
    # issues with certain ops not being available on GPU/TPU.
    # It does not affect operations other than datasets.
    fedjax.training.set_tf_cpu_only()

    # Load partially preprocessed splits.
    train_fd, _, test_fd = fedjax.datasets.stackoverflow.load_data() # *_fd.num_clients() returns 342477

    # Apply tokenizer during batching.
    tokenizer = fedjax.datasets.stackoverflow.StackoverflowTokenizer()
    # Same max sequence length as https://arxiv.org/pdf/2003.00295.pdf.
    vocab_size, embed_size, max_length = 10000, 96, 20
    train_fd = train_fd.preprocess_batch(tokenizer.as_preprocess_batch(max_length))
    test_fd = test_fd.preprocess_batch(tokenizer.as_preprocess_batch(max_length))
    model: models.Model = fedjax.models.stackoverflow.create_lstm_model(vocab_size = vocab_size, embed_size = embed_size, expected_length = 13.3)

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
    client_optimizer = fedjax.optimizers.sgd(learning_rate = FLAGS.eta_c)
    # server_optimizer = fedjax.optimizers.adam(
    #         learning_rate=10**(-2.5), b1=0.9, b2=0.999, eps=10**(-4))
    server_optimizer = fedjax.optimizers.sgd(learning_rate = 1.0) # Fed Avg
    # Hyperparameters for client local traing dataset preparation.
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(
        batch_size = FLAGS.batch_size, # Ideally this is adaptive and not necessary but batch_size is required.
        seed = jax.random.PRNGKey(FLAGS.random_seed),
    )
    algorithm = fed_avg.federated_averaging(grad_fn, client_optimizer,
                              server_optimizer,
                              client_batch_hparams)

    # Initialize model parameters and algorithm server state.
    init_params = model.init(jax.random.PRNGKey(FLAGS.random_seed))
    server_state = algorithm.init(init_params)

    # Train and eval loop.
    train_client_sampler, test_client_sampler = (
        fedjax.client_samplers.UniformGetClientSampler(fd = train_fd, num_clients = FLAGS.clients_per_round, seed = FLAGS.random_seed+1), 
        fedjax.client_samplers.UniformGetClientSampler(fd = test_fd, num_clients = FLAGS.clients_per_round, seed = FLAGS.random_seed+3)  
    )
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
            train_client_ids = [cid for cid, _, _ in clients]
            test_clients = test_client_sampler.sample()
            test_client_ids = [cid for cid, _, _ in test_clients]

            train_eval_datasets = [cds for _, cds in test_fd.get_clients(train_client_ids)]
            train_eval_batches = fedjax.padded_batch_client_datasets(
                train_eval_datasets, 
                batch_size = 256,
            )
            test_eval_datasets = [cds for _, cds in test_fd.get_clients(test_client_ids)]
            test_eval_batches = fedjax.padded_batch_client_datasets(
                test_eval_datasets, 
                batch_size = 256,
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
            print(f'[round {round_num}] test_metrics={test_metrics}')

    # Save final trained model parameters to file.
    fedjax.serialization.save_state(server_state.params, '/tmp/params')

if __name__ == '__main__':
    config.config_with_absl()
    app.run(main)
