"""Tiered Language Models (TLMs).

A framework for training language models with a public configuration and one
or more keyed configurations over a single set of weights. A compact secret
key specifies a permutation over a small parameter subset, inducing an
alternative computation graph.

Modules:
    model: Decoder-only transformer with tiered configuration support
    permutation: Key management and weight permutation utilities
    train: Pretraining and finetuning utilities
"""
