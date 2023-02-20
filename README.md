# FATHOM: Federated AuTomatic Hyperparameter OptiMization

[**Documentation**](./documentation/fathom/index.html) |
[**Link to Paper**](https://arxiv.org/abs/2211.02106) 


## What is FATHOM?

FATHOM is Federated AuTomatic Hyperparameter OptiMization, which is an online algorithm that operates as a one-shot procedure for [Federated Learning].  Its focus is specifically on hyperparameter optimization of: 1) client learning rate, 2) number of local steps, as well as 3) batch size, for [Federated Averaging].  


## Installation

You will need a moderately recent version of Python.  We recommend 3.7.

You will need to install [JAX]. We highly recommend the GPU version.  

Next, after copying this folder to your local working directory, install [FedJAX]:
```
pip install -r requirements.txt

```

Then follow the quickstart guide.

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

## Citation
```
@misc{kan2022federated,
    title={Federated Hypergradient Descent},
    author={Andrew K Kan},
    year={2022},
    eprint={2211.02106},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


## Contributing

If you'd like to contribute, or have any suggestions, you can contact us at andrew.k.kan@gmail.com or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the Apache License, Version 2.0. See [LICENSE] for the full license text.


[JAX]: https://github.com/google/jax
[FedJAX]: https://github.com/google/fedjax
[Federated Learning]: https://ai.googleblog.com/2017/04/federated-learning-collaborative.html
[Federated Averaging]: https://arxiv.org/abs/1602.05629
[LICENSE]: LICENSE

