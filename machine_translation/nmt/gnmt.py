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
"""Encoder and decoder usded in sequence-to-sequence learning."""
__all__ = ['GNMTEncoder', 'GNMTDecoder', 'get_gnmt_encoder_decoder']

import mxnet as mx
from mxnet.base import _as_list
from mxnet.gluon import nn, rnn
from mxnet.gluon.block import HybridBlock
from gluonnlp.model.seq2seq_encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, \
     _get_attention_cell, _get_cell_type, _nested_sequence_last


class GNMTEncoder(Seq2SeqEncoder):
    r"""Structure of the RNN Encoder similar to that used in
     "[Arxiv2016] Google's Neural Machine Translation System:
                 Bridgeing the Gap between Human and Machine Translation"

    The encoder first stacks several bidirectional RNN layers and then stacks multiple
    uni-directional RNN layers with residual connections.

    Parameters
    ----------
    cell_type : str or function
        Can be "lstm", "gru" or constructor functions that can be directly called,
         like rnn.LSTMCell
    num_layers : int
        Total number of layers
    num_bi_layers : int
        Total number of bidirectional layers
    hidden_size : int
        Number of hidden units
    dropout : float
        The dropout rate
    use_residual : bool
        Whether to use residual connection. Residual connection will be added in the
        uni-directional RNN layers
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, cell_type='lstm', num_layers=2, num_bi_layers=1, hidden_size=128,
                 dropout=0.0, use_residual=True,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 prefix=None, params=None):
        super(GNMTEncoder, self).__init__(prefix=prefix, params=params)
        self._cell_type = _get_cell_type(cell_type)
        assert num_bi_layers <= num_layers,\
            'Number of bidirectional layers must be smaller than the total number of layers, ' \
            'num_bi_layers={}, num_layers={}'.format(num_bi_layers, num_layers)
        self._num_bi_layers = num_bi_layers
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._use_residual = use_residual
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.rnn_cells = nn.HybridSequential()
            for i in range(num_layers):
                if i < num_bi_layers:
                    self.rnn_cells.add(rnn.BidirectionalCell(
                        l_cell=self._cell_type(hidden_size=self._hidden_size,
                                               i2h_weight_initializer=i2h_weight_initializer,
                                               h2h_weight_initializer=h2h_weight_initializer,
                                               i2h_bias_initializer=i2h_bias_initializer,
                                               h2h_bias_initializer=h2h_bias_initializer,
                                               prefix='rnn%d_l_' % i),
                        r_cell=self._cell_type(hidden_size=self._hidden_size,
                                               i2h_weight_initializer=i2h_weight_initializer,
                                               h2h_weight_initializer=h2h_weight_initializer,
                                               i2h_bias_initializer=i2h_bias_initializer,
                                               h2h_bias_initializer=h2h_bias_initializer,
                                               prefix='rnn%d_r_' % i)))
                else:
                    self.rnn_cells.add(
                        self._cell_type(hidden_size=self._hidden_size,
                                        i2h_weight_initializer=i2h_weight_initializer,
                                        h2h_weight_initializer=h2h_weight_initializer,
                                        i2h_bias_initializer=i2h_bias_initializer,
                                        h2h_bias_initializer=h2h_bias_initializer,
                                        prefix='rnn%d_' % i))

    def __call__(self, inputs, states=None, valid_length=None):
        """Encoder the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of initial states
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the last RNN layer
            - new_states of all the RNN layers
        """
        return super(GNMTEncoder, self).__call__(inputs, states, valid_length)

    def forward(self, inputs, states=None, valid_length=None):  #pylint: disable=arguments-differ, missing-docstring
        # TODO(sxjscience) Accelerate the forward using HybridBlock
        _, length, _ = inputs.shape
        new_states = []
        outputs = inputs
        for i, cell in enumerate(self.rnn_cells):
            begin_state = None if states is None else states[i]
            outputs, layer_states = cell.unroll(
                length=length, inputs=inputs, begin_state=begin_state, merge_outputs=True,
                valid_length=valid_length, layout='NTC')
            if i < self._num_bi_layers:
                # For bidirectional RNN, we use the states of the backward RNN
                new_states.append(layer_states[len(self.rnn_cells[i].state_info()) // 2:])
            else:
                new_states.append(layer_states)
            # Apply Dropout
            outputs = self.dropout_layer(outputs)
            if self._use_residual:
                if i > self._num_bi_layers:
                    outputs = outputs + inputs
            inputs = outputs
        if valid_length is not None:
            outputs = mx.nd.SequenceMask(outputs, sequence_length=valid_length,
                                         use_sequence_length=True, axis=1)
        return [outputs, new_states], []


class GNMTDecoder(HybridBlock, Seq2SeqDecoder):
    """Structure of the RNN Encoder similar to that used in the
    Google Neural Machine Translation paper.

    We use gnmt_v2 strategy in tensorflow/nmt

    Parameters
    ----------
    cell_type : str or type
    attention_cell : AttentionCell or str
        Arguments of the attention cell.
        Can be 'scaled_luong', 'normed_mlp', 'dot'
    num_layers : int
    hidden_size : int
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, cell_type='lstm', attention_cell='scaled_luong',
                 num_layers=2, hidden_size=128,
                 dropout=0.0, use_residual=True, output_attention=False,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 prefix=None, params=None):
        super(GNMTDecoder, self).__init__(prefix=prefix, params=params)
        self._cell_type = _get_cell_type(cell_type)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            self.attention_cell = _get_attention_cell(attention_cell, units=hidden_size)
            self.dropout_layer = nn.Dropout(dropout)
            se