import itertools
import sys
import collections
import os
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
from mxnet import nd, autograd, gluon, init, context, image
from mxnet.gluon import nn, rnn
import random
import re
import time
import tarfile
import zipfile

import mxnet as mx
import gluonnlp as nlp

import d2l



def load_data_imdb(batch_size, num_steps=500):
    d2l.download_imdb()
    train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = nlp.Vocab(nlp.data.count_tokens(itertools.chain.from_iterable(train_tokens)), min_freq=5)
    train_features = mx.nd.array([d2l.trim_pad(vocab[line], num_steps, vocab[vocab.unknown_token])
                               for line in train_tokens])
    test_features = mx.nd.array([d2l.trim_pad(vocab[line], num_steps, vocab[vocab.unknown_token])
                               for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab


# from d2l import train_ch12 as train
def train_batch_ch12(net, features, labels, loss, trainer, ctx_list):
    Xs, ys = d2l.split_batch(features, labels, ctx_list)
    with autograd.record():
        pys = [net(X) for X in Xs]
        ls = [loss(py, y) for py, y in zip(pys, ys)]
    for l in ls:
        l.backward()
    trainer.step(feat