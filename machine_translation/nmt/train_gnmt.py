"""
Google Neural Machine Translation
=================================

This example shows how to implement the GNMT model with Gluon NLP Toolkit.

@article{wu2016google,
  title={Google's neural machine translation system:
   Bridging the gap between human and machine translation},
  author={Wu, Yonghui and Schuster, Mike and Chen, Zhifeng and Le, Quoc V and
   Norouzi, Mohammad and Macherey, Wolfgang and Krikun, Maxim and Cao, Yuan and Gao, Qin and
   Macherey, Klaus and others},
  journal={arXiv preprint arXiv:1609.08144},
  year={2016}
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
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp

from gluonnlp.model.translation import NMTModel
from gluonnlp.loss import MaskedSoftmaxCELoss
from gnmt import get_gnmt_encoder_decoder
from translation import BeamSearchTranslator
from utils import logging_config
from bleu import compute_bleu
import dataprocessor

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Google NMT model')
parser.add_argument('--dataset', type=str, default='IWSLT2015', help='Dataset to use.')
parser.add_argument('--src_lang', type=str, default='en', help='Source language')
parser.add_argument('--tgt_lang', type=str, default='vi', help='Target language')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('--num_hidden', type=int, default=128, help='Dimension of the embedding '
                                                                'vectors and states.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the encoder'
                                                              ' and decoder')
parser.add_argument('--num_bi_layers', type=int, default=1,
                    help='number of bidirectional layers in the encoder and decoder')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--beam_size', type=int, default=4, help='Beam size')
parser.add_argument('--lp_alpha', type=float, default=1.0,
                    help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')
parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size')
parser.add_argument('--num_buckets', type=int, default=5, help='Bucket number')
parser.add_argument('--bucket_scheme', type=str, default='constant',
                    help='Strategy for generating bucket keys. It supports: '
                         '"constant": all the buckets have the same width; '
                         '"linear": the width of bucket increases linearly; '
                         '"exp": the width of bucket increases exponentially')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the '
                                                                    'throughput of the bucketing')
parser.add_argument('--src_max_len', type=int, default=50, help='Maximum length of the source '
                                                                'sentence')
parser.add_argument('--tgt_max_len', type=int, default=50, help='Maximum length of the target '
                                                                'sentence')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=1E-3, help='Initial learning rate')
parser.add_argument('--lr_update_factor', type=float, default=0.5,
                    help='Learning rate decay factor')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='out_