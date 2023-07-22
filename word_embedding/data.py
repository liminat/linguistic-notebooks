# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=
"""Word embedding training datasets."""

__all__ = [
    'WikiDumpStream', 'preprocess_dataset', 'wiki', 'transform_data_fasttext',
    'transform_data_word2vec', 'skipgram_lookup', 'cbow_lookup',
    'skipgram_fasttext_batch', 'cbow_fasttext_batch', 'skipgram_batch',
    'cbow_batch']

import functools
import io
import itertools
import json
import math
import os
import warnings

import mxnet as mx
import numpy as np

import gluonnlp as nlp
from gluonnlp import Vocab
from gluonnlp.base import numba_njit
from gluonnlp.data import CorpusDataset, SimpleDatasetStream
from utils import print_time


def preprocess_dataset(data, min_freq=5, max_vocab_size=None):
    """Dataset preprocessing helper.

    Parameters
    ----------
    data : mx.data.Dataset
        Input Dataset. For example gluonnlp.data.Text8 or gluonnlp.data.Fil9
    min_freq : int, default 5
        Minimum token frequency for a token to be included in the vocabulary
        and returned DataStream.
    max_vocab_size : int, optional
        Specifies a maximum size for the vocabulary.

    Returns
    -------
    gluonnlp.data.DataStream
        Each sample is a valid input to
        gluonnlp.data.EmbeddingCenterContextBatchify.
    gluonnlp.Vocab
        Vocabulary of all tokens in Text8 that occur at least min_freq times of
        maximum size max_vocab_size.
    idx_to_counts : list of int
        Mapping from token indices to their occurrence-counts in the Text8
        dataset.

    """
    with print_time('count and construct vocabulary'):
        counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
        vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                          bos_token=None, eos_token=None, min_freq=min_freq,
                          max_size=max_vocab_size)
        idx_to_counts = [counter[w] for w in vocab.idx_to_token]

    def code(sentence):
        return [vocab[token] for token in sentence if token in vocab]

    with print_time('code data'):
        data = data.transform(code, lazy=False)
    data = nlp.data.SimpleDataStream([data])
    return data, vocab, idx_to_counts


def wiki(wiki_root, wiki_date, wiki_language, max_vocab_size=None):
    """Wikipedia dump helper.

    Parameters
    ----------
    wiki_root : str
        Parameter for WikiDumpStream
    wiki_date : str
        Parameter for WikiDumpStream
    wiki_langua