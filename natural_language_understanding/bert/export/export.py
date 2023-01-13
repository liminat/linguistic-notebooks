"""
Export the BERT Model for Deployment

====================================

This script exports the BERT model to a hybrid model serialized as a symbol.json file,
which is suitable for deployment, or use with MXNet Module API.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming- \
      Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

# coding=utf-8

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
import logging
import warnings
import os
import time

import mxnet as mx
import gluonnlp as nlp
from hybrid_bert import get_hybrid_model
from hybrid_bert import HybridBERTClassifier, HybridBERTRegression, HybridBERTForQA

parser = argparse.ArgumentParser(description='Export hybrid BERT base model.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='The model parameter file saved from training.')

parser.add_argument('--model_name',
                    type=str,
                    default='bert_12_768_12',
                    choices=['bert_12_768_12', 'bert_24_1024_16'],
                    help='BERT model name. Options are "bert_12_768_12" and "bert_24_1024_16"')

parser.add_argument('--task',
                    type=str,
                    choices=['classification', 'regression', 'question_answering'],
                    required=True,
                    help='Task to export. Options are "classification", "regression", '
                         '"question_answering"')

parser.add_argument('--dataset_name',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                             'wiki_multilingual_uncased', 'wiki_multilingual_cased',
                             'wiki_cn_cased'],
                    help='BERT dataset name. Options include '
                         '"book_corpus_wiki_en_uncased", "book_c