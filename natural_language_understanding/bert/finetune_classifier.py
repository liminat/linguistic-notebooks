
"""
Sentence Pair Classification with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
sentence pair classification, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
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

import io
import os
import time
import argparse
import random
import logging
import warnings
import multiprocessing
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from gluonnlp.model import get_model
from gluonnlp.data import BERTTokenizer

from model.classification import BERTClassifier, BERTRegression
from data.classification import MRPCTask, QQPTask, RTETask, STSBTask, SSTTask
from data.classification import QNLITask, CoLATask, MNLITask, WNLITask, XNLITask
from data.classification import LCQMCTask, ChnSentiCorpTask
from data.transform import BERTDatasetTransform

tasks = {
    'MRPC': MRPCTask(),
    'QQP': QQPTask(),
    'QNLI': QNLITask(),
    'RTE': RTETask(),
    'STS-B': STSBTask(),
    'CoLA': CoLATask(),
    'MNLI': MNLITask(),
    'WNLI': WNLITask(),
    'SST': SSTTask(),
    'XNLI': XNLITask(),
    'LCQMC': LCQMCTask(),
    'ChnSentiCorp': ChnSentiCorpTask()
}

parser = argparse.ArgumentParser(
    description='BERT fine-tune examples for GLUE tasks.')
parser.add_argument(
    '--epochs', type=int, default=3, help='number of epochs, default is 3')
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Batch size. Number of examples per gpu in a minibatch, default is 32')
parser.add_argument(
    '--dev_batch_size',
    type=int,
    default=8,
    help='Batch size for dev set and test set, default is 8')
parser.add_argument(
    '--optimizer',
    type=str,
    default='bertadam',
    help='Optimization algorithm, default is bertadam')
parser.add_argument(
    '--lr',
    type=float,
    default=5e-5,
    help='Initial learning rate, default is 5e-5')
parser.add_argument(
    '--epsilon',
    type=float,
    default=1e-06,
    help='Small value to avoid division by 0, default is 1e-06'
)
parser.add_argument(
    '--warmup_ratio',
    type=float,
    default=0.1,
    help='ratio of warmup steps used in NOAM\'s stepsize schedule, default is 0.1')
parser.add_argument(
    '--log_interval',
    type=int,
    default=10,
    help='report interval, default is 10')
parser.add_argument(
    '--max_len',
    type=int,
    default=128,
    help='Maximum length of the sentence pairs, default is 128')
parser.add_argument(
    '--pad',
    action='store_true',
    help='Whether to pad to maximum length when preparing data batches. Default is False.')
parser.add_argument(
    '--seed', type=int, default=2, help='Random seed, default is 2')
parser.add_argument(
    '--accumulate',
    type=int,
    default=None,
    help='The number of batches for gradients accumulation to simulate large batch size. '
         'Default is None')
parser.add_argument(
    '--gpu', type=int, default=None, help='Which gpu for finetuning. By default cpu is used.')
parser.add_argument(
    '--task_name',
    type=str,
    choices=tasks.keys(),
    help='The name of the task to fine-tune. Choices include MRPC, QQP, '
         'QNLI, RTE, STS-B, CoLA, MNLI, WNLI, SST.')
parser.add_argument(
    '--bert_model',
    type=str,
    default='bert_12_768_12',
    help='The name of pre-trained BERT model to fine-tune'
    '(bert_24_1024_16 and bert_12_768_12).')
parser.add_argument(
    '--bert_dataset',
    type=str,
    default='book_corpus_wiki_en_uncased',
    help='The dataset BERT pre-trained with.'
    'Options include \'book_corpus_wiki_en_cased\', \'book_corpus_wiki_en_uncased\''
    'for both bert_24_1024_16 and bert_12_768_12.'
    '\'wiki_cn_cased\', \'wiki_multilingual_uncased\' and \'wiki_multilingual_cased\''
    'for bert_12_768_12 only.')
parser.add_argument(
    '--pretrained_bert_parameters',
    type=str,
    default=None,
    help='Pre-trained bert model parameter file. default is None')
parser.add_argument(
    '--model_parameters',
    type=str,
    default=None,
    help='A parameter file for the model that is loaded into the model'
    ' before training/inference. It is different from the parameter'
    ' file written after the model is trained. default is None')
parser.add_argument(
    '--output_dir',
    type=str,
    default='./output_dir',
    help='The output directory where the model params will be written.'
    ' default is ./output_dir')
parser.add_argument(
    '--only_inference',
    action='store_true',
    help='If set, we skip training and only perform inference on dev and test data.')

args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)
logging.captureWarnings(True)
logging.info(args)

batch_size = args.batch_size
dev_batch_size = args.dev_batch_size
task_name = args.task_name
lr = args.lr
epsilon = args.epsilon
accumulate = args.accumulate
log_interval = args.log_interval * accumulate if accumulate else args.log_interval
if accumulate:
    logging.info('Using gradient accumulation. Effective batch size = %d',
                 accumulate * batch_size)

# random seed
np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)

ctx = mx.cpu() if args.gpu is None else mx.gpu(args.gpu)

task = tasks[task_name]

# model and loss
only_inference = args.only_inference
model_name = args.bert_model
dataset = args.bert_dataset
pretrained_bert_parameters = args.pretrained_bert_parameters
model_parameters = args.model_parameters
if only_inference and not model_parameters:
    warnings.warn('model_parameters is not set. '
                  'Randomly initialized model will be used for inference.')

get_pretrained = not (pretrained_bert_parameters is not None
                      or model_parameters is not None)
bert, vocabulary = get_model(
    name=model_name,
    dataset_name=dataset,
    pretrained=get_pretrained,
    ctx=ctx,
    use_pooler=True,
    use_decoder=False,
    use_classifier=False)

if not task.class_labels: