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
    wiki_language : str
        Parameter for WikiDumpStream
    max_vocab_size : int, optional
        Specifies a maximum size for the vocabulary.

    Returns
    -------
    gluonnlp.data.DataStream
        Each sample is a valid input to
        gluonnlp.data.EmbeddingCenterContextBatchify.
    gluonnlp.Vocab
        Vocabulary of all tokens in the Wikipedia corpus as provided by
        WikiDumpStream but with maximum size max_vocab_size.
    idx_to_counts : list of int
        Mapping from token indices to their occurrence-counts in the Wikipedia
        corpus.

    """
    data = WikiDumpStream(
        root=os.path.expanduser(wiki_root), language=wiki_language,
        date=wiki_date)
    vocab = data.vocab
    if max_vocab_size:
        for token in vocab.idx_to_token[max_vocab_size:]:
            vocab.token_to_idx.pop(token)
        vocab.idx_to_token = vocab.idx_to_token[:max_vocab_size]
    idx_to_counts = data.idx_to_counts

    def code(shard):
        return [[vocab[token] for token in sentence if token in vocab]
                for sentence in shard]

    data = data.transform(code)
    return data, vocab, idx_to_counts


def transform_data_fasttext(data, vocab, idx_to_counts, cbow, ngram_buckets,
                            ngrams, batch_size, window_size,
                            frequent_token_subsampling=1E-4, dtype='float32',
                            index_dtype='int64'):
    """Transform a DataStream of coded DataSets to a DataStream of batches.

    Parameters
    ----------
    data : gluonnlp.data.DataStream
        DataStream where each sample is a valid input to
        gluonnlp.data.EmbeddingCenterContextBatchify.
    vocab : gluonnlp.Vocab
        Vocabulary containing all tokens whose indices occur in data. For each
        token, it's associated subwords will be computed and used for
        constructing the batches. No subwords are used if ngram_buckets is 0.
    idx_to_counts : list of int
        List of integers such that idx_to_counts[idx] represents the count of
        vocab.idx_to_token[idx] in the underlying dataset. The count
        information is used to subsample frequent words in the dataset.
        Each token is independently dropped with probability 1 - sqrt(t /
        (count / sum_counts)) where t is the hyperparameter
        frequent_token_subsampling.
    cbow : boolean
        If True, batches for CBOW are returned.
    ngram_buckets : int
        Number of hash buckets to consider for the fastText
        nlp.vocab.NGramHashes subword function.
    ngrams : list of int
        For each integer n in the list, all ngrams of length n will be
        considered by the nlp.vocab.NGramHashes subword function.
    batch_size : int
        The returned data stream iterates over batches of batch_size.
    window_size : int
        The context window size for
        gluonnlp.data.EmbeddingCenterContextBatchify.
    frequent_token_subsampling : float
        Hyperparameter for subsampling. See idx_to_counts above for more
        information.
    dtype : str or np.dtype, default 'float32'
        Data type of data array.
    index_dtype : str or np.dtype, default 'int64'
        Data type of index arrays.

    Returns
    -------
    gluonnlp.data.DataStream
        Stream over batches. Each returned element is a list corresponding to
        the arguments for the forward pass of model.SG or model.CBOW
        respectively based on if cbow is False or True. If ngarm_buckets > 0,
        the returned sample will contain ngrams. Both model.SG or model.CBOW
        will handle them correctly as long as they are initialized with the
        subword_function returned as second argument by this function (see
        below).
    gluonnlp.vocab.NGramHashes
        The subword_function used for obtaining the subwords in the returned
        batches.

    """
    if ngram_buckets <= 0:
        raise ValueError('Invalid ngram_buckets. Use Word2Vec training '
                         'pipeline if not interested in ngrams.')

    sum_counts = float(sum(idx_to_counts))
    idx_to_pdiscard = [
        1 - math.sqrt(frequent_token_subsampling / (count / sum_counts))
        for count in idx_to_counts]

    def subsample(shard):
        return [[
            t for t, r in zip(sentence,
                              np.random.uniform(0, 1, size=len(sentence)))
            if r > idx_to_pdiscard[t]] for sentence in shard]

    data = data.transform(subsample)

    batchify = nlp.data.batchify.EmbeddingCen