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
"""Data utilities for the named entity recognition task."""

import logging
from collections import namedtuple

import numpy as np
import mxnet as mx
import gluonnlp as nlp

TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])
PredictedToken = namedtuple('PredictedToken', ['text', 'true_tag', 'pred_tag'])

NULL_TAG = 'X'

def bio_bioes(tokens):
    """Convert a list of TaggedTokens in BIO(2) scheme to BIOES scheme.

    Parameters
    ----------
    tokens: List[TaggedToken]
        A list of tokens in BIO(2) scheme

    Returns
    -------
    List[TaggedToken]:
        A list of tokens in BIOES scheme
    """
    ret = []
    for index, token in enumerate(tokens):
        if token.tag == 'O':
            ret.append(token)
        elif token.tag.startswith('B'):
      