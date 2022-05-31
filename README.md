# FATHOM: Federated AuTomatic Hyperparameter OptiMization

[**Documentation**](./documentation/fathom/index.html) |
[**Full Paper**](./conf-paper/fathom.pdf) | [**Main Paper**](./conf-paper/fathom-paper.pdf) | 
[**Supplementary To Paper**](./conf-paper/fathom-supp.pdf) 


## What is FATHOM?

FATHOM is Federated AuTomatic Hyperparameter OptiMization, which is an online algorithm that operates as a one-shot procedure for [Federated Learning].  Its focus is specifically on hyperparameter optimization of: 1) client learning rate, 2) number of local steps, as well as 3) batch size, for [Federated Averaging].  


## Installation

You will need a moderately recent version of Python. Please check
[the PyPI page](https://pypi.org/project/fedjax/) for the up to date version
requirement.

First, install [JAX]. We highly recommend the GPU version.

Then copy this folder to your local working directory, and follow the quickstart guide.

## Quickstart

For EMNIST Character Recognition task:
```
python3 experiments/fathom/run_fathom.py \
  -flagfile=experiments/fathom/fathom.EMNIST_CONV.flags \
  -root_dir=./tmp/fathom
```

For Stack Overflow Next-Word Prediction task:
```
python3 experiments/fathom/run_fathom.py \
  -flagfile=experiments/fathom/fathom.STACKOVERFLOW_WORD.flags \
  -root_dir=./tmp/fathom
```

For replicating simulation results from the paper:
```
source experiments/fathom/sim_launch.sh
```

## Contributing

If you'd like to contribute, or have any suggestions, you can contact us at FATHOM-AUTHORS@FATHOM-AUTHORS.org or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the Apache License, Version 2.0. See [LICENSE] for the full license text.


[JAX]: https://github.com/google/jax
[Federated Learning]: https://ai.googleblog.com/2017/04/federated-learning-collaborative.html
[Federated Averaging]: https://arxiv.org/abs/1602.05629
[LICENSE]: LICENSE

