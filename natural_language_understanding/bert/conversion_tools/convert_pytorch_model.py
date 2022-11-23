# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation
""" Script for converting PyTorch Model to Gluon. """

import argparse
import json
import logging
import os
import sys

import mxnet as mx
import gluonnlp as nlp
import torch
from gluonnlp.model import BERTEncoder, BERTModel
from gluonnlp.model.bert import bert_hparams

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
from utils import get_hash, load_text_vocab, tf_vocab_to_gluon_vocab

parser = argparse.ArgumentParser(description='Conversion script for PyTorch BERT model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='bert_12_768_12',
                    choices=['bert_12_768_12', 'bert_24_1024_16'], help='BERT model name')
parser.add_argument('--pytorch_checkpoint_dir', type=str,
                    help='Path to Tensorflow checkpoint folder.')
parser.add_argument('--vocab_file', type=str, help='Full path to the vocab.txt')
parser.add_argument('--gluon_pytorch_name_mapping', type=str,
                    default='gluon_to_pytorch_naming.json',
                    help='Output of infer_pytorch_gluon_parameter_name_mapping.py')
parser.add_argument('--out_dir', type=str, default=os.path.join('~', 'output'),
                    help='Path to output folder. The folder must exist.')
parser.add_argument('--debug', action='store_true', help='debugging mode')
args = parser.parse_args()
logging.getLogger().setLevel(