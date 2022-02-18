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
"""EMNIST models."""

from fedjax.core import metrics
from fedjax.core import models
import haiku as hk
import jax
import numpy as np

import fathom

# Defines the expected structure of input batches to the model. This is used to
# determine the model parameter shapes.
_HAIKU_SAMPLE_BATCH = {
    'x': np.zeros((1, 28, 28, 1), dtype=np.float32),
    'y': np.zeros(1, dtype=np.float32)
}
_TRAIN_LOSS_CLASSIFIER = lambda b, p: metrics.unreduced_cross_entropy_loss(b['y'], p)
_EVAL_METRICS_CLASSIFIER = {
    'loss': metrics.CrossEntropyLoss(),
    'accuracy': metrics.Accuracy()
}
_TRAIN_LOSS_AUTOENCODER = lambda b, p: fathom.core.metrics.mean_squared_error_loss(b['x'], p)
_EVAL_METRICS_AUTOENCODER = {
    'loss': fathom.core.metrics.MeanSquaredErrorLoss(),
}

def create_mlp_model(only_digits: bool = False) -> models.Model:
    """ MLP model used in https://arxiv.org/abs/2008.03606.
            2 hidden layer (300u-100) MLP  
    """ 
    num_classes = 10 if only_digits else 62

    def forward_pass(batch):
        network = hk.Sequential([
            hk.Flatten(),
            hk.Linear(300),
            jax.nn.relu,
            hk.Linear(100),
            jax.nn.relu,
            hk.Linear(num_classes),
        ])
        return network(batch['x'])

    transformed_forward_pass = hk.transform(forward_pass)
    return models.create_model_from_haiku(
        transformed_forward_pass=transformed_forward_pass,
        sample_batch=_HAIKU_SAMPLE_BATCH,
        train_loss=_TRAIN_LOSS_CLASSIFIER,
        eval_metrics=_EVAL_METRICS_CLASSIFIER)

def create_autoencoder_model() -> models.Model:
    """ AE model used in https://arxiv.org/abs/2003.00295 """ 
    def forward_pass(batch):
        network = hk.Sequential([
            hk.Flatten(),
            hk.Linear(1000),
            jax.nn.sigmoid,
            hk.Linear(500),
            jax.nn.sigmoid,
            hk.Linear(250),
            jax.nn.sigmoid,
            hk.Linear(30),
            jax.nn.sigmoid,
            hk.Linear(250),
            jax.nn.sigmoid,
            hk.Linear(500),
            jax.nn.sigmoid,
            hk.Linear(1000),
            jax.nn.sigmoid,
            hk.Linear(784),
        ])
        return network(batch['x'])

    transformed_forward_pass = hk.transform(forward_pass)
    return models.create_model_from_haiku(
        transformed_forward_pass=transformed_forward_pass,
        sample_batch=_HAIKU_SAMPLE_BATCH,
        train_loss=_TRAIN_LOSS_AUTOENCODER,
        eval_metrics=_EVAL_METRICS_AUTOENCODER)
