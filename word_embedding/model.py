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
    def __init__(self, token_to_idx, output_dim, b