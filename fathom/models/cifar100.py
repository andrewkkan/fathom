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
from fathom.models import resnets_gn # ResNets with GroupNorm vice BatchNorm

# Defines the expected structure of input batches to the model. This is used to
# determine the model parameter shapes.
_HAIKU_SAMPLE_BATCH = {
    'x': np.zeros((1, 24, 24, 3), dtype=np.float32),
    'y': np.zeros(1, dtype=np.int32)
}
_TRAIN_LOSS_CLASSIFIER = lambda b, p: metrics.unreduced_cross_entropy_loss(b['y'], p)
_EVAL_METRICS_CLASSIFIER = {
    'loss': metrics.CrossEntropyLoss(),
    'accuracy': metrics.Accuracy()
}

def create_resnet18_model() -> models.Model:
    num_classes = 100
    def forward_pass(batch):
        network = resnets_gn.ResNet18(num_classes = num_classes)
        return network(inputs = batch['x']) 
    transformed_forward_pass = hk.transform(forward_pass)
    return models.create_model_from_haiku(
        transformed_forward_pass = transformed_forward_pass,
        sample_batch = _HAIKU_SAMPLE_BATCH,
        train_loss = _TRAIN_LOSS_CLASSIFIER,
        eval_metrics = _EVAL_METRICS_CLASSIFIER,
    )
