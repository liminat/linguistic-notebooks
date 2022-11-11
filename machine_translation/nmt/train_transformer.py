"""
Transformer
=================================

This example shows how to implement the Transformer model with Gluon NLP Toolkit.

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones,
          Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6000--6010},
  year={2017}
}
"""

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
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import time
import random
import os
import logging
import math
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp

from gluonnlp.loss import MaskedSoftmaxCELoss, LabelSmoothing
from gluonnlp.model.translation import NMTModel
from gluonnlp.model.transformer import get_transformer_encoder_decoder, ParallelTransformer
from gluonnlp.utils.parallel import Parallel
from translation import BeamSearchTranslator

from utils import logging_config
from bleu import _bpe_to_words, compute_bleu
import dataprocessor

np.random.seed(100)
random.seed(1