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
"""Word embedding models."""

import mxnet as mx
import numpy as np

import gluonnlp as nlp


class Net(mx.gluon.HybridBlock):
    """Base class for word2vec and fastText SkipGram and CBOW networks.

    Parameters
    ----------
    token_to_idx : dict
        token_to_idx mapping of the vocabulary that this model is to be trained
        with. token_to_idx is used for __getitem__ and __contains__. For
        len(token_to_idx) is used during initialization to obtain the input_dim
        of the embedding matrix.
    output_dim : int
        Dimension of the dense embedding.
    batch_size : int
        Batchsize this model will be trained with. TODO temporary until
        random_like ops are supported
    negatives_weights : mxnet.nd.NDArray
        Weights for UnigramCandidateSampler for sampling negatives.
    smoothing : float, default 0.75
        Smoothing factor applied to negatives_weights. Final weights are
        mxnet.nd.power(negative_weights, smoothing).
    num_negatives : int, default 5
        Number of negatives to sample for each real sample.
    sparse_grad : bool, default True
        Specifies mxnet.gluon.nn.Embedding sparse_grad argument.
    dtype : str, default 'float32'
        dtype argument passed to gluon.nn.Embedding

    """

    # pylint: disable=abstract-method
    def __init__(self, token_to_idx, output_dim, batch_size, negatives_weights,
                 subword_function=None, num_negatives=5, smoothing=0.75,
                 sparse_grad=True, dtype='float32', **kwargs):
        super(Net, self).__init__(**kwargs)

        self._kwargs = dict(
            input_dim=len(token_to_idx), output_dim=output_dim, dtype=dtype,
            sparse_grad=sparse_grad, num_negatives=num_negatives)

        with self.name_scope():
            if subword_function is not None:
                self.embedding = nlp.model.train.FasttextEmbeddingModel(
                    token_to_idx=token_to_idx,
                    subword_function=subword_function,
                    output_dim=output_dim,
                    weight_initializer=mx.init.Uniform(scale=1 / output_dim),
                    sparse_grad=sparse_grad,
                )
            else:
                self.embedding = nlp.model.train.CSREmbeddingModel(
                    token_to_idx=token_to_idx,
                    output_dim=output_dim,
                    weight_initializer=mx.init.Uniform(scale=1 / output_dim),
                    sparse_grad=sparse_grad,
                )
            self.embedding_out = mx.gluon.nn.Embedding(
                len(token_to_idx), output_dim=output_dim,
                weight_initializer=mx.init.Zero(), sparse_grad=sparse_grad,
                dtype=dtype)

            self.negatives_sampler = nlp.data.UnigramCandidateSampler(
                weights=negatives_weights**smoothing, shape=(batch_size, ),
                dtype='int64')

    def __getitem__(self, tokens):
        return self.embedding[tokens]


class SG(Net):
    """SkipGram network"""

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, center, context, center_words):
        """SkipGram forward pass.

        Parameters
        ----------
        center : mxnet.nd.NDArray or mxnet.sym.Symbol
            Sparse CSR array of word / subword indices of shape (batch_size,
            len(token_to_idx) + num_subwords). Embedding for center words are
            computed via F.sparse.dot between the CSR center array and the
            weight matrix.
        context : mxnet.nd.NDArray or mxnet.sym.Symbol
            Dense array of context words of shape (batch_size, ). Also used for
            row-wise independently masking negatives equal to one of context.
        center_words : mxnet.nd.