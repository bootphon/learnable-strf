#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Juan Manuel Coria

from typing import Optional
from typing import Text

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio import Spectrogram

from torch.nn.modules.conv import _ConvNd
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from typing import Optional

from .sincnet import SincNet
from .tdnn import XVectorNet

from .convolutional import Convolutional
from .recurrent import Recurrent
from .linear import Linear
from .pooling import Pooling
from .scaling import Scaling

from pyannote.audio.train.model import Model
from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import RESOLUTION_CHUNK
from pyannote.audio.train.model import RESOLUTION_FRAME
from pyannote.core import SlidingWindow
from torch.nn.utils.rnn import PackedSequence
from pyannote.audio.train.task import Task



def get_info(sequences):
    """Get info about batch of sequences

    Parameters
    ----------
    sequences : `torch.Tensor` or `PackedSequence`
        Batch of sequences given as a `torch.Tensor` of shape
        (batch_size, n_samples, n_features) if sequences all share the same
        length, or as a `PackedSequence` if they do not.

    Returns
    -------
    batch_size : `int`
        Number of sequences in batch.
    n_features : `int`
        Number of features.
    device : `torch.device`
        Device.
    """

    packed_sequences = isinstance(sequences, PackedSequence)

    if packed_sequences:
        _, n_features = sequences.data.size()
        batch_size = sequences.batch_sizes[0].item()
        device = sequences.data.device
    else:
        # check input feature dimension
        batch_size, _, n_features = sequences.size()
        device = sequences.device

    return batch_size, n_features, device


class AmplitudeToDB(torch.jit.ScriptModule):
    # type: (Tensor, float, float, float, Optional[float]) -> Tensor
    r"""Copy pasted pytorch/audio due to version compatibility
    Turn a tensor from the power/amplitude scale to the decibel scale.
    This output depends on the maximum value in the input tensor, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.
    Args:
        x (torch.Tensor): Input tensor before being converted to decibel scale
        multiplier (float): Use 10. for power and 20. for amplitude
        amin (float): Number to clamp ``x``
        db_multiplier (float): Log10(max(reference value and amin))
        top_db (Optional[float]): Minimum negative cut-off in decibels.
        A reasonable number
            is 80. (Default: ``None``)
    Returns:
        torch.Tensor: Output tensor in decibel scale
    """

    def __init__(self, stype='power', top_db=None):
        super(AmplitudeToDB, self).__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = torch.jit.Attribute(top_db, Optional[float])
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x):
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        if self.top_db is not None:
            x_db = x_db.clamp(min=x_db.max().item() - self.top_db)

        return x_db


class RNN(nn.Module):
    """Recurrent layers
    Parameters
    ----------
    n_features : `int`
        Input feature shape.
    unit : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    hidden_size : `int`, optional
        Number of features in the hidden state h. Defaults to 16.
    num_layers : `int`, optional
        Number of recurrent layers. Defaults to 1.
    bias : `boolean`, optional
        If False, then the layer does not use bias weights. Defaults to True.
    dropout : `float`, optional
        If non-zero, introduces a Dropout layer on the outputs of each layer
        except the last layer, with dropout probability equal to dropout.
        Defaults to 0.
    bidirectional : `boolean`, optional
        If True, becomes a bidirectional RNN. Defaults to False.
    concatenate : `boolean`, optional
        Concatenate output of each layer instead of using only the last one
        (which is the default behavior).
    pool : {'sum', 'max', 'last', 'x-vector'}, optional
        Temporal pooling strategy. Defaults to no pooling.
    """

    def __init__(
            self,
            n_features,
            unit="LSTM",
            hidden_size=16,
            num_layers=1,
            bias=True,
            dropout=0,
            bidirectional=False,
            concatenate=False,
            pool=None,
    ):
        super().__init__()

        self.n_features = n_features

        self.unit = unit
        Klass = getattr(nn, self.unit)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.concatenate = concatenate
        self.pool = pool
        self.pool_ = TemporalPooling.create(pool) if pool is not None else None

        if num_layers < 1:
            msg = '"bidirectional" must be set to False when num_layers < 1'
            if bidirectional:
                raise ValueError(msg)
            msg = '"concatenate" must be set to False when num_layers < 1'
            if concatenate:
                raise ValueError(msg)
            return

        if self.concatenate:

            self.rnn_ = nn.ModuleList([])
            for i in range(self.num_layers):

                if i > 0:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2
                else:
                    input_dim = self.n_features

                if i + 1 == self.num_layers:
                    dropout = 0
                else:
                    dropout = self.dropout

                rnn = Klass(
                    input_dim,
                    self.hidden_size,
                    num_layers=1,
                    bias=self.bias,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=self.bidirectional,
                )

                self.rnn_.append(rnn)

        else:
            self.rnn_ = Klass(
                self.n_features,
                self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )

    def forward(self, features, return_intermediate=False):
        """Apply recurrent layer (and optional temporal pooling)
        Parameters
        ----------
        features : `torch.Tensor`
            Features shaped as (batch_size, n_frames, n_features)
        return_intermediate : `boolean`, optional
            Return intermediate RNN hidden state.
        Returns
        -------
        output : `torch.Tensor`
            TODO. Shape depends on parameters...
        intermediate : `torch.Tensor`
            (num_layers, batch_size, hidden_size * num_directions)
        """

        if self.num_layers < 1:

            if return_intermediate:
                msg = ('"return_intermediate" must be set to False '
                       "when num_layers < 1")
                raise ValueError(msg)

            output = features

        else:

            if return_intermediate:
                num_directions = 2 if self.bidirectional else 1

            if self.concatenate:

                if return_intermediate:
                    msg = ('"return_intermediate" is not supported '
                           'when "concatenate" is True')
                    raise NotADirectoryError(msg)

                outputs = []

                hidden = None
                output = None
                # apply each layer separately...
                for i, rnn in enumerate(self.rnn_):
                    if i > 0:
                        output, hidden = rnn(output, hidden)
                    else:
                        output, hidden = rnn(features)
                    outputs.append(output)

                # ... and concatenate their output
                output = torch.cat(outputs, dim=2)

            else:
                output, hidden = self.rnn_(features)

                if return_intermediate:
                    if self.unit == "LSTM":
                        h = hidden[0]
                    elif self.unit == "GRU":
                        h = hidden

                    # to (num_layers, batch_size, num_directions * hidden_size)
                    h = h.view(self.num_layers, num_directions, -1,
                               self.hidden_size)
                    intermediate = (h.transpose(2, 1).contiguous().view(
                        self.num_layers, -1,
                        num_directions * self.hidden_size))

        if self.pool_ is not None:
            output = self.pool_(output)

        if return_intermediate:
            return output, intermediate

        return output

    def dimension():
        doc = "Output features dimension."

        def fget(self):
            if self.num_layers < 1:
                dimension = self.n_features
            else:
                dimension = self.hidden_size

            if self.bidirectional:
                dimension *= 2

            if self.concatenate:
                dimension *= self.num_layers

            if self.pool == "x-vector":
                dimension *= 2

            return dimension

        return locals()

    dimension = property(**dimension())

    def intermediate_dimension(self, layer):
        if self.num_layers < 1:
            dimension = self.n_features
        else:
            dimension = self.hidden_size

        if self.bidirectional:
            dimension *= 2

        return dimension


class FF(nn.Module):
    """Feedforward layers
    Parameters
    ----------
    n_features : `int`
        Input dimension.
    hidden_size : `list` of `int`, optional
        Linear layers hidden dimensions. Defaults to [16, ].
    """

    def __init__(self, n_features, hidden_size=[
            16,
    ]):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size

        self.linear_ = nn.ModuleList([])
        for hidden_size in self.hidden_size:
            linear = nn.Linear(n_features, hidden_size, bias=True)
            self.linear_.append(linear)
            n_features = hidden_size

    def forward(self, features):
        """
        Parameters
        ----------
        features : `torch.Tensor`
            (batch_size, n_samples, n_features) or (batch_size, n_features)
        Returns
        -------
        output : `torch.Tensor`
            (batch_size, n_samples, hidden_size[-1]) or (batch_size, hidden_size[-1])
        """

        output = features
        for linear in self.linear_:
            output = linear(output)
            output = torch.tanh(output)
        return output

    def dimension():
        doc = "Output dimension."

        def fget(self):
            if self.hidden_size:
                return self.hidden_size[-1]
            return self.n_features

        return locals()

    dimension = property(**dimension())


class Embedding(nn.Module):
    """Embedding
    Parameters
    ----------
    n_features : `int`
        Input dimension.
    batch_normalize : `boolean`, optional
        Apply batch normalization. This is more or less equivalent to
        embedding whitening.
    scale : {"fixed", "logistic"}, optional
        Scaling method. Defaults to no scaling.
    unit_normalize : deprecated in favor of 'scale'
    """

    def __init__(
            self,
            n_features: int,
            batch_normalize: bool = False,
            scale: Text = None,
            unit_normalize: bool = False,
    ):
        super().__init__()

        self.n_features = n_features

        self.batch_normalize = batch_normalize
        if self.batch_normalize:
            self.batch_normalize_ = nn.BatchNorm1d(
                n_features, eps=1e-5, momentum=0.1, affine=False)

        self.scale = scale
        self.scaling = Scaling(n_features, method=scale)

        if unit_normalize is True:
            msg = f"'unit_normalize' has been deprecated in favor of 'scale'."
            raise ValueError(msg)

    def forward(self, embedding):

        if self.batch_normalize:
            embedding = self.batch_normalize_(embedding)

        return self.scaling(embedding)

    @property
    def dimension(self):
        """Output dimension."""
        return self.n_features


class PyanNet(Model):
    """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output
    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters. Use {'skip': True} to use handcrafted features
        instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
        becomes [ features -> RNN -> ...].
    rnn : `dict`, optional
        Recurrent network parameters. Defaults to `RNN` default parameters.
    ff : `dict`, optional
        Feed-forward layers parameters. Defaults to `FF` default parameters.
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
    """

    @staticmethod
    def get_alignment(task: Task, sincnet=None, **kwargs):
        """Get frame alignment"""

        if sincnet is None:
            sincnet = dict()

        if sincnet.get("skip", False):
            return "center"

        return SincNet.get_alignment(task, **sincnet)

    @staticmethod
    def get_resolution(
        task: Task,
        sincnet: Optional[dict] = None,
        rnn: Optional[dict] = None,
        **kwargs,
    ) -> Resolution:
        """Get sliding window used for feature extraction
        Parameters
        ----------
        task : Task
        sincnet : dict, optional
        rnn : dict, optional
        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
            Returns RESOLUTION_CHUNK if model returns one vector per input
            chunk, RESOLUTION_FRAME if model returns one vector per input
            frame, and specific sliding window otherwise.
        """

        if rnn is None:
            rnn = {"pool": None}

        if rnn.get("pool", None) is not None:
            return RESOLUTION_CHUNK

        if sincnet is None:
            sincnet = {"skip": False}

        if sincnet.get("skip", False):
            return RESOLUTION_FRAME

        return SincNet.get_resolution(task, **sincnet)

    def init(
        self,
        sincnet: Optional[dict] = None,
        rnn: Optional[dict] = None,
        ff: Optional[dict] = None,
        embedding: Optional[dict] = None,
    ):
        """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output
        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters. Use {'skip': True} to use handcrafted features
            instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
            becomes [ features -> RNN -> ...].
        rnn : `dict`, optional
            Recurrent network parameters. Defaults to `RNN` default parameters.
        ff : `dict`, optional
            Feed-forward layers parameters. Defaults to `FF` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        n_features = self.n_features

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if not sincnet.get("skip", False):
            if n_features != 1:
                msg = (
                    f"SincNet only supports mono waveforms. "
                    f"Here, waveform has {n_features} channels."
                )
                raise ValueError(msg)
            self.sincnet_ = SincNet(**sincnet)
            n_features = self.sincnet_.dimension

        if rnn is None:
            rnn = dict()
        self.rnn = rnn
        self.rnn_ = RNN(n_features, **rnn)
        n_features = self.rnn_.dimension

        if ff is None:
            ff = dict()
        self.ff = ff
        self.ff_ = FF(n_features, **ff)
        n_features = self.ff_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
            return

        self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
        self.activation_ = self.task.default_activation

    def forward(self, waveforms, return_intermediate=None):
        """Forward pass
        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms. In case SincNet is skipped, a tensor with shape
            (batch_size, n_samples, n_features) is expected.
        return_intermediate : `int`, optional
            Index of RNN layer. Returns RNN intermediate hidden state.
            Defaults to only return the final output.
        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        intermediate : `torch.Tensor`
            Intermediate network output (only when `return_intermediate`
            is provided).
        """

        if self.sincnet.get("skip", False):
            output = waveforms
        else:
            output = self.sincnet_(waveforms)

        if return_intermediate is None:
            output = self.rnn_(output)
        else:
            if return_intermediate == 0:
                intermediate = output
                output = self.rnn_(output)
            else:
                return_intermediate -= 1
                # get RNN final AND intermediate outputs
                output, intermediate = self.rnn_(output,
                                                 return_intermediate=True)
                # only keep hidden state of requested layer
                intermediate = intermediate[return_intermediate]

        output = self.ff_(output)

        if self.task.is_representation_learning:
            return self.embedding_(output)

        output = self.linear_(output)
        output = self.activation_(output)

        if return_intermediate is None:
            return output
        return output, intermediate

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)

    def intermediate_dimension(self, layer):
        if layer == 0:
            return self.sincnet_.dimension
        return self.rnn_.intermediate_dimension(layer - 1)


class SincTDNN(Model):
    """waveform -> SincNet -> XVectorNet (TDNN -> FC) -> output
    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters.
    tdnn : `dict`, optional
        X-Vector Time-Delay neural network parameters.
        Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
    """

    @staticmethod
    def get_alignment(task: Task, sincnet=None, **kwargs):
        """Get frame alignment"""

        if sincnet is None:
            sincnet = dict()

        return SincNet.get_alignment(task, **sincnet)

    supports_packed = False

    @staticmethod
    def get_resolution(
        task: Task, sincnet: Optional[dict] = None, **kwargs
    ) -> Resolution:
        """Get sliding window used for feature extraction
        Parameters
        ----------
        task : Task
        sincnet : dict, optional
        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
        """

        # TODO add support for frame-wise and sequence labeling tasks
        # TODO https://github.com/pyannote/pyannote-audio/issues/290
        return RESOLUTION_CHUNK

    def init(
        self,
        sincnet: Optional[dict] = None,
        tdnn: Optional[dict] = None,
        embedding: Optional[dict] = None,
    ):
        """waveform -> SincNet -> XVectorNet (TDNN -> FC) -> output
        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters.
        tdnn : `dict`, optional
            X-Vector Time-Delay neural network parameters.
            Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        n_features = self.n_features

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if n_features != 1:
            raise ValueError(
                "SincNet only supports mono waveforms. "
                f"Here, waveform has {n_features} channels."
            )
        self.sincnet_ = SincNet(**sincnet)
        n_features = self.sincnet_.dimension

        if tdnn is None:
            tdnn = dict()
        self.tdnn = tdnn
        self.tdnn_ = XVectorNet(n_features, **tdnn)
        n_features = self.tdnn_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
        else:
            self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
            self.activation_ = self.task.default_activation

    def forward(self, waveforms: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass
        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms
        Returns
        -------
        output : `torch.Tensor`
            Final network output or intermediate network output
            (only when `return_intermediate` is provided).
        """

        output = self.sincnet_(waveforms)

        return_intermediate = (
            "segment6" if self.task.is_representation_learning else None
        )
        output = self.tdnn_(output, return_intermediate=return_intermediate)

        if self.task.is_representation_learning:
            return self.embedding_(output)

        return self.activation_(self.linear_(output))

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)


class STRFNet(nn.Module):
    """STRFNet (learnable) feature extraction
    Parameters
    ----------
    waveform_normalize : `bool`, optional
        Standardize waveforms (to zero mean and unit standard deviation) and
        apply (learnable) affine transform. Defaults to True.
    instance_normalize : `bool`, optional
        Standardize internal representation (to zero mean and unit standard
        deviation) and apply (learnable) affine transform. Defaults to True.
    """

    @staticmethod
    def get_alignment(task: Task, **kwargs):
        """Get frame alignment"""
        return "strict"

    @staticmethod
    def get_resolution(
        task: Task,
        **kwargs,
    ) -> SlidingWindow:
        """Get frame resolution
        Parameters
        ----------
        task : Task
        sample_rate : int, optional
        kerne_size : list of int, optional
        stride : list of int, optional
        max_pool : list of int, optional
        Returns
        -------
        resolution : SlidingWindow
            Frame resolution.
        """

        return RESOLUTION_CHUNK

    def __init__(
        self,
        waveform_normalize=True,
        sample_rate=16000,
        num_gabor_filters=64,
        kernel_size=(9, 111),
        stride=[1, 1],
        n_mels=64,
        n_fft=2048,
        window='hamming',
        duration=0.025,
        step=0.01,
        pre_lstm_layer_bool=False,
        window_pre_lstm=1,
        mel_trainable=False,
        stft_trainable=False,
        pre_lstm_compression_dim=64,
        dropout_pre_lstm=0.0,
        gabor_mode='concat',
        instance_normalize=False,
    ):
        super().__init__()

        # check parameters values
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.step = step
        self.duration = self.n_fft / self.sample_rate
        self.hop_length = int(self.step * self.sample_rate)
        self.num_gabor_filters = num_gabor_filters
        self.kernel_size = kernel_size
        self.waveform_normalize = waveform_normalize
        self.pre_lstm_layer_bool = pre_lstm_layer_bool
        self.window_pre_lstm = window_pre_lstm
        self.stft_trainable = stft_trainable
        self.n_mels = n_mels
        self.stride = stride
        self.instance_normalize = instance_normalize
        # Waveform normalization
        self.waveform_normalize = waveform_normalize
        self.pre_lstm_compression_dim = pre_lstm_compression_dim
        if self.waveform_normalize:
            self.waveform_normalize_ = torch.nn.InstanceNorm1d(1, affine=True)
        # self.n_features_ = n_features
        # self.n_classes_ = n_classes
        self.duration = duration
        self.gabor_mode = gabor_mode
        cuda = torch.device('cuda')
        self.mel_layer = Spectrogram.MelSpectrogram(
            trainable_mel=mel_trainable,
            trainable_STFT=stft_trainable,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            device=cuda)
        self.lay_ampl = AmplitudeToDB(top_db=80)
        self.gabor_layer = STRFConv2d(
            in_channels=1,
            out_channels=self.num_gabor_filters,
            kernel_size=self.kernel_size,
            padding=(int(self.kernel_size[0] // 2),
                     int(self.kernel_size[1] // 2)),
            stride=self.stride,
            n_features=self.n_mels,
            classic_freq_unit_init=self.classic_freq_unit_init)
        # self.gabor_activation = nn.ReLU(inplace=True)
        self.gabor_activation = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_pre_lstm_layer = nn.Dropout2d(p=self.dropout_pre_lstm)
        if self.gabor_mode == 'concat':
            input_dim_conv = 2 * self.n_mels * self.num_gabor_filters
        else:
            input_dim_conv = self.n_mels * self.num_gabor_filters
        if self.gabor_mode == 'concat':
            input_dim_conv = 2 * self.n_mels
        else:
            input_dim_conv = self.n_mels
        if self.pre_lstm_layer_bool:
            self.pre_lstm_layer = nn.Conv2d(
                self.num_gabor_filters,
                self.pre_lstm_compression_dim,
                (input_dim_conv, window_pre_lstm),
                stride=(window_pre_lstm, 1),
                padding=int(window_pre_lstm // 2))
        if self.instance_normalize:
            self.instance_norm_layer = nn.InstanceNorm1d(out_channels,
                                                         affine=True)

    def forward(self, waveforms):
        """Extract STRFNet features
        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1)
            Batch of waveforms
        Returns
        -------
        features : (batch_size, n_frames, out_channels[-1])
        """

        if isinstance(waveforms, PackedSequence):
            msg = (f'{self.__class__.__name__} does not support batches '
                   f'containing sequences of variable length.')
            raise ValueError(msg)

        batch_size, n_features, device = get_info(waveforms)
        output = waveforms
        if self.waveform_normalize:
            output = output.transpose(1, 2)
            output = F.instance_norm(output)
            # output = output.transpose(1, 2)
        if self.mfcc_concat:
            output_mfcc = self.mfcc_layer(output)
        output = self.mel_layer(output)
        output = self.lay_ampl(output)
        if self.gabor_mode == 'abs':
            output = torch.pow(self.gabor_layer(
                output, use_real=True), 2) + torch.pow(self.gabor_layer(
                    output, use_real=False), 2)
            output = torch.pow(output, 0.5)
        elif self.gabor_mode == 'real':
            output = self.gabor_layer(output, use_real=True)
        elif self.gabor_mode == 'imag':
            output = self.gabor_layer(output, use_real=False)
        elif self.gabor_mode == 'concat':
            output_real = self.gabor_layer(output, use_real=True)
            output_imag = self.gabor_layer(output, use_real=False)
            output = torch.cat((output_real, output_imag), 1)
        elif self.gabor_mode == 'pass':
            output = output
        output_shape = output.shape
        output = self.dropout_pre_lstm_layer(output)
        if self.gabor_mode == 'concat':
            output = output.reshape(
                output.size(0), self.num_gabor_filters,
                2 * int(self.n_mels / self.stride[0]), output.size(3))
        else:
            output = output.reshape(
                output.size(0), self.num_gabor_filters,
                int(self.n_mels / self.stride[0]), output.size(3))
        if self.pre_lstm_layer_bool:
            # Pre-lstm-layer
            output = self.pre_lstm_layer(output)
            # apply non-linear activation function
            output = F.relu(output, inplace=True)
            output = output.reshape(output.size(0),
                                    output.size(3),
                                    output.size(1))
        return output

    def dimension():
        doc = "Output features dimension."

        def fget(self):
            return self.pre_lstm_compression_dim

        return locals()

    dimension = property(**dimension())


class STRFTDNN(Model):
    """waveform -> Mel-FB -> STRF layer -> XVectorNet (TDNN -> FC) -> output
    Parameters
    ----------
    strfnet : `dict`, optional
        STRFNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters.
    tdnn : `dict`, optional
        X-Vector Time-Delay neural network parameters.
        Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
    """

    @staticmethod
    def get_alignment(task: Task, strfnet=None, **kwargs):
        """Get frame alignment"""

        if strfnet is None:
            strfnet = dict()

        return STRFNet.get_alignment(task, **strfnet)

    supports_packed = False

    @staticmethod
    def get_resolution(
        task: Task, strfnet: Optional[dict] = None, **kwargs
    ) -> Resolution:
        """Get sliding window used for feature extraction
        Parameters
        ----------
        task : Task
        strfnet : dict, optional
        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
        """

        # TODO add support for frame-wise and sequence labeling tasks
        # TODO https://github.com/pyannote/pyannote-audio/issues/290
        return RESOLUTION_CHUNK

    def init(
        self,
        strfnet: Optional[dict] = None,
        tdnn: Optional[dict] = None,
        embedding: Optional[dict] = None,
    ):
        """waveform -> SincNet -> XVectorNet (TDNN -> FC) -> output
        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters.
        tdnn : `dict`, optional
            X-Vector Time-Delay neural network parameters.
            Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        n_features = self.n_features

        if strfnet is None:
            strfnet = dict()
        self.strfnet = strfnet

        if n_features != 1:
            raise ValueError(
                "strfnet only supports mono waveforms. "
                f"Here, waveform has {n_features} channels."
            )
        self.strfnet_ = STRFNet(**strfnet)
        n_features = self.strfnet_.dimension

        if tdnn is None:
            tdnn = dict()
        self.tdnn = tdnn
        self.tdnn_ = XVectorNet(n_features, **tdnn)
        n_features = self.tdnn_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
        else:
            self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
            self.activation_ = self.task.default_activation

    def forward(self, waveforms: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass
        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms
        Returns
        -------
        output : `torch.Tensor`
            Final network output or intermediate network output
            (only when `return_intermediate` is provided).
        """

        output = self.strfnet_(waveforms)

        return_intermediate = (
            "segment6" if self.task.is_representation_learning else None
        )
        output = self.tdnn_(output, return_intermediate=return_intermediate)

        if self.task.is_representation_learning:
            return self.embedding_(output)

        return self.activation_(self.linear_(output))

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)


class GaborInitConv2d(_ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 device=None,
                 n_features=64):

        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(GaborInitConv2d,
              self).__init__(in_channels, out_channels,
                             kernel_size, stride, padding, dilation, False,
                             _pair(0), groups, bias, padding_mode)
        self.n_features = n_features
        self.freq = (np.pi / 2) * 1.41**(
            -np.random.randint(0, 5, (out_channels, in_channels)))
        self.freq = nn.Parameter(torch.Tensor(self.freq), requires_grad=False)
        self.theta = (np.pi / 8) * torch.randint(
            0, 8, (out_channels, in_channels), requires_grad=False)
        self.theta = nn.Parameter(
            self.theta.type(torch.Tensor), requires_grad=False)
        self.psi = nn.Parameter(
            np.pi * torch.rand(out_channels, in_channels), requires_grad=False)
        self.sigma_x = nn.Parameter(np.pi / self.freq, requires_grad=False)
        self.sigma_y = nn.Parameter(np.pi / self.freq, requires_grad=False)
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]
        grid = [
            torch.linspace(-self.x0 + 1, self.x0, self.kernel_size[0]),
            torch.linspace(-self.y0 + 1, self.y0, self.kernel_size[1])
        ]
        y, x = torch.meshgrid(grid)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma_x = self.sigma_x[i, j].expand_as(y)
                sigma_y = self.sigma_y[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(y)
                theta = self.theta[i, j].expand_as(y)
                psi = self.psi[i, j].expand_as(y)
                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)
                g = torch.zeros(y.shape)
                g = torch.exp(-0.5 * ((rotx**2) / (sigma_x + 1e-3)**2 +
                                      (roty**2) / (sigma_y + 1e-3)**2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * np.pi * sigma_x * sigma_y)
                self.weight.data[i, j] = g

    def forward(self, sequences, use_real=True):
        # batch_size, n_features, device = get_info(sequences)
        packed_sequences = isinstance(sequences, PackedSequence)
        if packed_sequences:
            device = sequences.data.device
        else:
            device = sequences.device
        sequences = sequences.reshape(
            sequences.size(0), 1, self.n_features, -1)
        weight = self.weight.to(device)
        return F.conv2d(sequences, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
 
class STRFConv2d(_ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 device=None,
                 n_features=64,
                 classic_freq_unit_init=True):

        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(STRFConv2d,
              self).__init__(in_channels, out_channels,
                             kernel_size, stride, padding, dilation, False,
                             _pair(0), groups, bias, padding_mode)
        self.n_features = n_features

        self.theta = np.random.vonmises(0, 0, (out_channels, in_channels))
        self.gamma = np.random.vonmises(0, 0, (out_channels, in_channels))
        self.psi = np.random.vonmises(0, 0, (out_channels, in_channels))
        self.gamma = nn.Parameter(torch.Tensor(self.gamma))
        self.psi = nn.Parameter(torch.Tensor(self.psi))
        if classic_freq_unit_init:
            self.freq = (np.pi / 2) * 1.41**(
                -np.random.uniform(0, 5, size=(out_channels, in_channels)))
        else:
            self.freq = np.random.rayleigh(1.1,
                                           size=(out_channels, in_channels))
        # betaprime.rvs(1, 5, size=(out_channels, in_channels))

        self.freq = nn.Parameter(torch.Tensor(self.freq))
        self.theta = nn.Parameter(torch.Tensor(self.theta))

        self.sigma_x = 2 * 1.41**(np.random.uniform(
            0, 6, (out_channels, in_channels)))
        self.sigma_x = nn.Parameter(torch.Tensor(self.sigma_x))
        self.sigma_y = 2 * 1.41**(np.random.uniform(
            0, 6, (out_channels, in_channels)))
        self.sigma_y = nn.Parameter(torch.Tensor(self.sigma_y))
        self.f0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.t0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]

    def forward(self, sequences, use_real=True):
        # batch_size, n_features, device = get_info(sequences)
        packed_sequences = isinstance(sequences, PackedSequence)
        if packed_sequences:
            device = sequences.data.device
        else:
            device = sequences.device
        sequences = sequences.reshape(
            sequences.size(0), 1, self.n_features, -1)
        grid = [
            torch.linspace(-self.f0 + 1, self.f0, self.kernel_size[0]),
            torch.linspace(-self.t0 + 1, self.t0, self.kernel_size[1])
        ]
        f, t = torch.meshgrid(grid)
        f = f.to(device)
        t = t.to(device)
        weight = torch.empty(self.weight.shape, requires_grad=False)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma_x = self.sigma_x[i, j].expand_as(t)
                sigma_y = self.sigma_y[i, j].expand_as(t)
                # omega_freq = self.omega_freq[i, j].expand_as(y)
                # omega_time = self.omega_time[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(t)
                theta = self.theta[i, j].expand_as(t)
                gamma = self.gamma[i, j].expand_as(t)
                psi = self.psi[i, j].expand_as(t)
                rotx = t * torch.cos(theta) + f * torch.sin(theta)
                roty = -t * torch.sin(theta) + f * torch.cos(theta)
                rot_gamma = t * torch.cos(gamma) + f * torch.sin(gamma)
                g = torch.zeros(t.shape)
                g = torch.exp(-0.5 * ((f**2) / (sigma_x + 1e-3)**2 +
                                      (t**2) / (sigma_y + 1e-3)**2))
                if use_real:
                    # g = g * torch.cos(2 * np.pi * (omega_freq * x +
                    # omega_time * y) + psi)
                    g = g * torch.cos(freq * rot_gamma)
                else:
                    g = g * torch.sin(freq * rot_gamma)
                g = g / (2 * np.pi * sigma_x * sigma_y)
                weight[i, j] = g
                self.weight.data[i, j] = g
        weight = weight.to(device)
        return F.conv2d(sequences, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class StackedGaborRNNRawWave(Model):
    """Stacked recurrent neural network with Gabor Learnable first Layer with
    Raw waveform computations

    Parameters
    ----------
    specifications : `dict`
        Provides model IO specifications using the following data structure:
            {'X': {'dimension': DIMENSION},
             'y': {'classes': CLASSES},
             'task': TASK_TYPE}
        where
            * DIMENSION is the input feature dimension
            * CLASSES is the list of (human-readable) output classes
            * TASK_TYPE is either TASK_MULTI_CLASS_CLASSIFICATION,
            TASK_REGRESSION, or TASK_MULTI_LABEL_CLASSIFICATION.
            Depending on which task is
                adressed, the final activation will vary. Classification relies
                on log-softmax, multi-label classificatition and regression use
                sigmoid.
    instance_normalize : boolean, optional
        Apply mean/variance normalization on input sequences.
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent : list, optional
        List of hidden dimensions of stacked recurrent layers. Defaults to
        [16, ], i.e. one recurrent layer with hidden dimension of 16.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    pooling : {None, 'sum', 'max'}
        Apply temporal pooling before linear layers. Defaults to no pooling.
        This is useful for tasks expecting just one label per sequence.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, ], i.e.
        one linear layer with hidden dimension of 16.
    """

    def init(self,
             instance_normalize=False,
             rnn='LSTM',
             recurrent=[
                 16,
             ],
             bidirectional=False,
             linear=[
                 16,
             ],
             pooling=None,
             num_gabor_filters=64,
             kernel_size=(25, 25),
             stride=[1, 1],
             n_mels=128,
             waveform_normalize=True,
             n_fft=2048,
             window='hamming',
             duration=0.025,
             step=0.001,
             task_duration=2.0,
             sample_rate=16000,
             pre_lstm_layer_bool=False,
             window_pre_lstm=1,
             mel_trainable=False,
             stft_trainable=False,
             pre_lstm_compression_dim=64,
             dropout_pre_lstm=0.0,
             gabor_mode='real',
             Spectrogram_input='MEL',
             direct_mel=False):
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.step = step
        self.duration = self.n_fft / self.sample_rate
        self.hop_length = int(self.step * self.sample_rate)
        self.num_gabor_filters = num_gabor_filters
        self.kernel_size = kernel_size
        self.waveform_normalize = waveform_normalize
        self.pre_lstm_layer_bool = pre_lstm_layer_bool
        self.window_pre_lstm = window_pre_lstm
        self.stft_trainable = stft_trainable
        self.n_mels = n_mels
        self.dropout_pre_lstm = dropout_pre_lstm
        self.Spectrogram_input = Spectrogram_input
        self.Spectrogram_input = Spectrogram_input
        if self.Spectrogram_input == 'STFT':
            self.n_mels = int(self.n_fft / 2) + 1

        if self.waveform_normalize:
            self.waveform_normalize_ = torch.nn.InstanceNorm1d(1, affine=True)

        n_features = self.specifications['X']['dimension']
        self.n_features_ = n_features

        n_classes = len(self.specifications['y']['classes'])
        self.n_classes_ = n_classes

        self.instance_normalize = instance_normalize

        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.pre_lstm_compression_dim = pre_lstm_compression_dim
        self.stride = stride
        self.pooling = pooling
        self.linear = linear
        self.duration = duration
        self.task_duration = task_duration
        cuda = torch.device('cuda')
        if self.Spectrogram_input == 'STFT':
            self.mel_layer = Spectrogram.STFT(
                trainable=self.stft_trainable,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                freq_scale='linear',
                center=False,
                device=cuda)
        elif self.Spectrogram_input == 'CQT':
            self.mel_layer = Spectrogram.CQT1992v2(
                sr=self.sample_rate,
                bins_per_octave=24,
                n_bins=self.n_mels,
                center=True,
                fmin=125,
                hop_length=self.hop_length,
                pad_mode='constant',
                device=cuda)
        elif self.Spectrogram_input == 'MEL':
            self.mel_layer = Spectrogram.MelSpectrogram(
                trainable_mel=mel_trainable,
                trainable_STFT=stft_trainable,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=False,
                device=cuda)
        if self.mfcc_concat:
            self.mfcc_layer = Spectrogram.MFCC(
                n_mfcc=20,
                norm='ortho',
                trainable_mel=mel_trainable,
                trainable_STFT=stft_trainable,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=False,
                device=cuda)
        self.lay_ampl = AmplitudeToDB(top_db=80)
        self.num_directions_ = 2 if self.bidirectional else 1
        # self.gabor_layer = GaborConv2d(
        #     in_channels=1,
        #     out_channels=self.num_gabor_filters,
        #     kernel_size=self.kernel_size,
        #     padding=(int(self.kernel_size[0] // 2),
        #              int(self.kernel_size[1] // 2)),
        #     stride=self.stride,
        #     n_features=self.n_mels)
        self.gabor_layer = STRFConv2d(
            in_channels=1,
            out_channels=self.num_gabor_filters,
            kernel_size=self.kernel_size,
            padding=(int(self.kernel_size[0] // 2),
                     int(self.kernel_size[1] // 2)),
            stride=self.stride,
            n_features=self.n_mels)
        self.gabor_activation = nn.ReLU(inplace=True)
        self.dropout_pre_lstm_layer = nn.Dropout2d(p=self.dropout_pre_lstm)
        if self.gabor_mode == 'concat':
            input_dim_conv = 2 * self.n_mels * self.num_gabor_filters
        else:
            input_dim_conv = self.n_mels * self.num_gabor_filters
        if self.pre_lstm_layer_bool:
            if self.variant_architecture:
                self.pre_lstm_layer = nn.Conv2d(
                    self.num_gabor_filters,
                    self.pre_lstm_compression_dim,
                    (input_dim_conv, window_pre_lstm),
                    stride=(window_pre_lstm, 1),
                    padding=int(window_pre_lstm // 2))
            else:
                self.pre_lstm_layer = nn.Conv2d(
                    1,
                    self.pre_lstm_compression_dim,
                    (input_dim_conv, window_pre_lstm),
                    stride=(window_pre_lstm, 1),
                    padding=int(window_pre_lstm // 2))
            input_dim = self.n_mels
        # create list of recurrent layers
        self.recurrent_layers_ = []
        # This must be ensured to well-declare the padding in
        assert kernel_size[0] % 2 == 1, \
            'Should be odd size for gabor kernels'
        # input_dim = int(self.n_mels * self.num_gabor_filters / stride[0])
        # if self.gabor_mode == 'concat':
        #     input_dim = self.pre_lstm_compression_dim
        # else:
        if self.gabor_mode == 'pass':
            input_dim = self.n_mels
        elif self.gabor_mode == 'concat_spectrogram':
            input_dim = self.n_mels + self.pre_lstm_compression_dim
        else:
            input_dim = self.pre_lstm_compression_dim
        if self.mfcc_concat:
            input_dim += 20
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    bidirectional=self.bidirectional,
                    batch_first=True)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(
                    input_dim,
                    hidden_dim,
                    bidirectional=self.bidirectional,
                    batch_first=True)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

        self.last_hidden_dim_ = input_dim

        self.final_layer_ = nn.Linear(self.last_hidden_dim_, self.n_classes_)

        # Define output activation
        self.activation_ = self.task.default_activation

    @property
    def task(self):
        """Type of task addressed by the model

        Shortcut for self.specifications['task']
        """
        return self.specifications['task']

    @property
    def classes(self):
        return self.specifications['y']['classes']

    @property
    def n_classes(self):
        return len(self.specifications['y']['classes'])

    @staticmethod
    def get_alignment(task: Task, **kwargs):
        return 'center'

    @staticmethod
    def get_resolution(task: Task,
                       sample_rate=16000,
                       duration=0.025,
                       step=0.005,
                       **kwargs) -> Resolution:
        """
        """
        return SlidingWindow(
            start=-.5 * duration, duration=duration, step=step)

    def forward(self, waveforms, return_intermediate=None):
        """

        Parameters
        ----------
        sequences : (batch_size, n_samples, n_features) `torch.tensor`
            Batch of sequences.

        Returns
        -------
        predictions : `torch.tensor`
            Shape is (batch_size, n_samples, n_classes) without pooling, and
            (batch_size, n_classes) with pooling.
        """
        if isinstance(waveforms, PackedSequence):
            msg = (f'{self.__class__.__name__} does not support batches '
                   f'containing sequences of variable length.')
            raise ValueError(msg)

        batch_size, n_features, device = get_info(waveforms)

        if n_features != self.n_features_:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features_))

        output = waveforms
        if self.waveform_normalize:
            output = output.transpose(1, 2)
            output = F.instance_norm(output)
            # output = output.transpose(1, 2)
        if self.mfcc_concat:
            output_mfcc = self.mfcc_layer(output)
        output = self.mel_layer(output)
        output = self.lay_ampl(output)
        length_slice = int(self.sample_rate * self.task_duration)

        n_frames = int(
            (length_slice + int(self.step * self.sample_rate) - self.n_fft) //
            float(int(self.step * self.sample_rate)))
        p1d = (1, np.max(
            int(self.task_duration / self.step - n_frames) - 1, 0))
        if self.Spectrogram_input != 'CQT':
            if self.mfcc_concat:
                output_mfcc = F.pad(output_mfcc, p1d, "constant", 0)
                output = F.pad(output, p1d, "constant", 0)
            else:
                output_mel = F.pad(output, p1d, "constant", 0)
                output = F.pad(output, p1d, "constant", 0)
        else:
            output = output[:, :, :-1]
        if self.gabor_mode == 'abs':
            output = torch.pow(self.gabor_layer(
                output, use_real=True), 2) + torch.pow(self.gabor_layer(
                    output, use_real=False), 2)
            output = torch.pow(output, 0.5)
        elif self.gabor_mode == 'real':
            output = self.gabor_layer(output, use_real=True)
        elif self.gabor_mode == 'imag':
            output = self.gabor_layer(output, use_real=False)
        elif self.gabor_mode == 'concat':
            output_real = self.gabor_layer(output, use_real=True)
            output_imag = self.gabor_layer(output, use_real=False)
            output = torch.cat((output_real, output_imag), 1)
        elif self.gabor_mode == 'pass':
            output = output
        output_shape = output.shape
        output = self.dropout_pre_lstm_layer(output)

        if self.gabor_mode == 'concat':
            output = output.reshape(
                output.size(0), 1, 2 * int(
                    self.n_mels * self.num_gabor_filters / self.stride[0]),
                output.size(3))
        elif self.gabor_mode == 'pass':
            output = output.reshape(
                output.size(0), int(self.n_mels), 1, output.size(2))
        else:
            output = output.reshape(
                output.size(0), 1,
                int(self.n_mels * self.num_gabor_filters / self.stride[0]),
                output.size(3))
        if self.gabor_mode != 'pass':
            output = self.gabor_activation(output)
        if self.pre_lstm_layer_bool:
            # Pre-lstm-layer

            output = self.pre_lstm_layer(output)
            # apply non-linear activation function
            output = F.relu(output, inplace=True)
            if self.gabor_mode == 'concat_spectrogram':
                output = torch.cat(
                    (output,
                     output_mel.reshape(
                         output_mel.size(0), output_mel.size(1), 1,
                         output_mel.size(2))), 1)
        output = output.reshape(output.size(0), output.size(3), output.size(1))

        # stack recurrent layers
        for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):

            if self.rnn == 'LSTM':
                # initial hidden and cell states
                h = torch.zeros(
                    self.num_directions_,
                    batch_size,
                    hidden_dim,
                    device=device,
                    requires_grad=False)
                c = torch.zeros(
                    self.num_directions_,
                    batch_size,
                    hidden_dim,
                    device=device,
                    requires_grad=False)
                hidden = (h, c)

            elif self.rnn == 'GRU':
                # initial hidden state
                hidden = torch.zeros(
                    self.num_directions_,
                    batch_size,
                    hidden_dim,
                    device=device,
                    requires_grad=False)

            # apply current recurrent layer and get output sequence
            output, _ = layer(output, hidden)

            # average both directions in case of bidirectional layers
            if self.bidirectional:
                output = .5 * (
                    output[:, :, :hidden_dim] + output[:, :, hidden_dim:])

        if self.pooling is not None:
            if self.pooling == 'sum':
                output = output.sum(dim=1)
            elif self.pooling == 'max':
                output, _ = output.max(dim=1)

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = torch.tanh(output)

        # apply final classification layer
        output = self.final_layer_(output)

        output = self.activation_(output)

        if return_intermediate is None:
            return output
        return output, intermediate


class StackedGaborInitRNNRawWave(Model):
    """Stacked recurrent neural network with Gabor Learnable first Layer with
    Raw waveform computations

    Parameters
    ----------
    specifications : `dict`
        Provides model IO specifications using the following data structure:
            {'X': {'dimension': DIMENSION},
             'y': {'classes': CLASSES},
             'task': TASK_TYPE}
        where
            * DIMENSION is the input feature dimension
            * CLASSES is the list of (human-readable) output classes
            * TASK_TYPE is either TASK_MULTI_CLASS_CLASSIFICATION,
            TASK_REGRESSION, or TASK_MULTI_LABEL_CLASSIFICATION.
            Depending on which task is
                adressed, the final activation will vary. Classification relies
                on log-softmax, multi-label classificatition and regression use
                sigmoid.
    instance_normalize : boolean, optional
        Apply mean/variance normalization on input sequences.
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent : list, optional
        List of hidden dimensions of stacked recurrent layers. Defaults to
        [16, ], i.e. one recurrent layer with hidden dimension of 16.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    pooling : {None, 'sum', 'max'}
        Apply temporal pooling before linear layers. Defaults to no pooling.
        This is useful for tasks expecting just one label per sequence.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, ], i.e.
        one linear layer with hidden dimension of 16.
    """

    def init(self,
             instance_normalize=False,
             rnn='LSTM',
             recurrent=[
                 16,
             ],
             bidirectional=False,
             linear=[
                 16,
             ],
             pooling=None,
             num_gabor_filters=64,
             kernel_size=(25, 25),
             stride=[1, 1],
             n_mels=128,
             waveform_normalize=True,
             n_fft=2048,
             window='hamming',
             duration=0.025,
             step=0.001,
             task_duration=2.0,
             sample_rate=16000,
             pre_lstm_layer_bool=False,
             window_pre_lstm=7,
             mel_trainable=False,
             stft_trainable=False,
             pre_lstm_compression_dim=64,
             dropout_pre_lstm=0.0,
             gabor_mode='real',
             Spectrogram_input='MEL'):
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.step = step
        self.duration = self.n_fft / self.sample_rate
        self.hop_length = int(self.step * self.sample_rate)
        self.num_gabor_filters = num_gabor_filters
        self.kernel_size = kernel_size
        self.waveform_normalize = waveform_normalize
        self.pre_lstm_layer_bool = pre_lstm_layer_bool
        self.window_pre_lstm = window_pre_lstm
        self.stft_trainable = stft_trainable
        self.n_mels = n_mels
        self.dropout_pre_lstm = dropout_pre_lstm
        self.Spectrogram_input = Spectrogram_input
        self.Spectrogram_input = Spectrogram_input
        if self.Spectrogram_input == 'STFT':
            self.n_mels = int(self.n_fft / 2) + 1

        if self.waveform_normalize:
            self.waveform_normalize_ = torch.nn.InstanceNorm1d(1, affine=True)

        n_features = self.specifications['X']['dimension']
        self.n_features_ = n_features

        n_classes = len(self.specifications['y']['classes'])
        self.n_classes_ = n_classes

        self.instance_normalize = instance_normalize

        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.pre_lstm_compression_dim = pre_lstm_compression_dim
        self.stride = stride
        self.pooling = pooling
        self.linear = linear
        self.duration = duration
        self.task_duration = task_duration
        self.gabor_mode = gabor_mode

        cuda = torch.device('cuda')
        if self.Spectrogram_input == 'STFT':
            self.mel_layer = Spectrogram.STFT(
                trainable=self.stft_trainable,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                freq_scale='linear',
                center=False,
                device=cuda)
        elif self.Spectrogram_input == 'CQT':
            self.mel_layer = Spectrogram.CQT1992v2(
                sr=self.sample_rate,
                bins_per_octave=24,
                n_bins=self.n_mels,
                center=True,
                fmin=125,
                hop_length=self.hop_length,
                pad_mode='constant',
                device=cuda)
        elif self.Spectrogram_input == 'MEL':
            self.mel_layer = Spectrogram.MelSpectrogram(
                trainable_mel=mel_trainable,
                trainable_STFT=stft_trainable,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=False,
                device=cuda)
        if self.mfcc_concat:
            self.mfcc_layer = Spectrogram.MFCC(
                n_mfcc=20,
                norm='ortho',
                trainable_mel=mel_trainable,
                trainable_STFT=stft_trainable,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=False,
                device=cuda)
        self.lay_ampl = AmplitudeToDB(top_db=80)
        self.num_directions_ = 2 if self.bidirectional else 1

        self.gabor_layer = GaborInitConv2d(
            in_channels=1,
            out_channels=self.num_gabor_filters,
            kernel_size=self.kernel_size,
            padding=(int(self.kernel_size[0] // 2),
                     int(self.kernel_size[1] // 2)),
            stride=self.stride,
            n_features=self.n_mels)
        self.gabor_activation = nn.ReLU(inplace=True)
        self.dropout_pre_lstm_layer = nn.Dropout2d(p=self.dropout_pre_lstm)
        if self.gabor_mode == 'concat':
            input_dim_conv = 2 * self.n_mels * self.num_gabor_filters
        else:
            input_dim_conv = self.n_mels * self.num_gabor_filters
        if self.variant_architecture:
            if self.gabor_mode == 'concat':
                input_dim_conv = 2 * self.n_mels
            else:
                input_dim_conv = self.n_mels
        if self.pre_lstm_layer_bool:
            if self.variant_architecture:
                self.pre_lstm_layer = nn.Conv2d(
                    self.num_gabor_filters,
                    self.pre_lstm_compression_dim,
                    (input_dim_conv, window_pre_lstm),
                    stride=(window_pre_lstm, 1),
                    padding=int(window_pre_lstm // 2))
            else:
                self.pre_lstm_layer = nn.Conv2d(
                    1,
                    self.pre_lstm_compression_dim,
                    (input_dim_conv, window_pre_lstm),
                    stride=(window_pre_lstm, 1),
                    padding=int(window_pre_lstm // 2))
            input_dim = self.n_mels
        # create list of recurrent layers
        self.recurrent_layers_ = []
        # This must be ensured to well-declare the padding in
        assert kernel_size[0] % 2 == 1, \
            'Should be odd size for gabor kernels'
        # input_dim = int(self.n_mels * self.num_gabor_filters / stride[0])
        # if self.gabor_mode == 'concat':
        #     input_dim = self.pre_lstm_compression_dim
        # else:
        if self.gabor_mode == 'pass':
            input_dim = self.n_mels
        elif self.gabor_mode == 'concat_spectrogram':
            input_dim = self.n_mels + self.pre_lstm_compression_dim
        else:
            input_dim = self.pre_lstm_compression_dim
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    bidirectional=self.bidirectional,
                    batch_first=True)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(
                    input_dim,
                    hidden_dim,
                    bidirectional=self.bidirectional,
                    batch_first=True)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

        self.last_hidden_dim_ = input_dim

        self.final_layer_ = nn.Linear(self.last_hidden_dim_, self.n_classes_)

        # Define output activation
        self.activation_ = self.task.default_activation

    @property
    def task(self):
        """Type of task addressed by the model

        Shortcut for self.specifications['task']
        """
        return self.specifications['task']

    @property
    def classes(self):
        return self.specifications['y']['classes']

    @property
    def n_classes(self):
        return len(self.specifications['y']['classes'])

    @staticmethod
    def get_alignment(task: Task, **kwargs):
        return 'center'

    @staticmethod
    def get_resolution(task: Task,
                       sample_rate=16000,
                       duration=0.025,
                       step=0.005,
                       **kwargs) -> Resolution:
        """
        """
        return SlidingWindow(
            start=-.5 * duration, duration=duration, step=step)

    def forward(self, waveforms, return_intermediate=None):
        """

        Parameters
        ----------
        sequences : (batch_size, n_samples, n_features) `torch.tensor`
            Batch of sequences.

        Returns
        -------
        predictions : `torch.tensor`
            Shape is (batch_size, n_samples, n_classes) without pooling, and
            (batch_size, n_classes) with pooling.
        """
        if isinstance(waveforms, PackedSequence):
            msg = (f'{self.__class__.__name__} does not support batches '
                   f'containing sequences of variable length.')
            raise ValueError(msg)

        batch_size, n_features, device = get_info(waveforms)

        if n_features != self.n_features_:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features_))

        output = waveforms
        if self.waveform_normalize:
            output = output.transpose(1, 2)
            output = F.instance_norm(output)
            # output = output.transpose(1, 2)
        output = self.mel_layer(output)
        output = self.lay_ampl(output)
        length_slice = int(self.sample_rate * self.task_duration)

        n_frames = int(
            (length_slice + int(self.step * self.sample_rate) - self.n_fft) //
            float(int(self.step * self.sample_rate)))
        p1d = (1, np.max(
            int(self.task_duration / self.step - n_frames) - 1, 0))
        if self.Spectrogram_input != 'CQT':
            output_mel = F.pad(output, p1d, "constant", 0)
            output = F.pad(output, p1d, "constant", 0)
        else:
            output = output[:, :, :-1]
        if self.gabor_mode == 'abs' or self.gabor_mode == 'concat_spectrogram':
            output = torch.pow(self.gabor_layer(
                output, use_real=True), 2) + torch.pow(self.gabor_layer(
                    output, use_real=False), 2)
            output = torch.pow(output, 0.5)
        elif self.gabor_mode == 'real':
            output = self.gabor_layer(output, use_real=True)
        elif self.gabor_mode == 'imag':
            output = self.gabor_layer(output, use_real=False)
        elif self.gabor_mode == 'concat':
            output_real = self.gabor_layer(output, use_real=True)
            output_imag = self.gabor_layer(output, use_real=False)
            output = torch.cat((output_real, output_imag), 1)
        elif self.gabor_mode == 'pass':
            output = output
        output_shape = output.shape
        output = self.dropout_pre_lstm_layer(output)
        if self.variant_architecture:
            if self.gabor_mode == 'concat':
                output = output.reshape(
                    output.size(0), self.num_gabor_filters,
                    2 * int(self.n_mels / self.stride[0]), output.size(3))
            else:
                output = output.reshape(
                    output.size(0), self.num_gabor_filters,
                    int(self.n_mels / self.stride[0]), output.size(3))
        else:
            if self.gabor_mode == 'concat':
                output = output.reshape(
                    output.size(0), 1, 2 * int(
                        self.n_mels * self.num_gabor_filters / self.stride[0]),
                    output.size(3))
            elif self.gabor_mode == 'max_pool_freq':
                output = output.reshape(
                    output.size(0), 1,
                    int(self.num_gabor_filters / self.stride[0]),
                    output.size(3))
            elif self.gabor_mode == 'pass':
                output = output.reshape(
                    output.size(0), int(self.n_mels), 1, output.size(2))
            else:
                output = output.reshape(
                    output.size(0), 1,
                    int(self.n_mels * self.num_gabor_filters / self.stride[0]),
                    output.size(3))
        if self.gabor_mode != 'pass':
            output = self.gabor_activation(output)
        if self.pre_lstm_layer_bool:
            # Pre-lstm-layer
            if self.tucker_decomposition:
                output, _ = partial_tucker(
                    output,
                    modes=[1, 2, 3],
                    ranks=(1, self.pre_lstm_compression_dim, output.shape[3]))
            else:
                output = self.pre_lstm_layer(output)
            # apply non-linear activation function
            output = F.relu(output, inplace=True)
            if self.gabor_mode == 'concat_spectrogram':
                output = torch.cat(
                    (output,
                     output_mel.reshape(
                         output_mel.size(0), output_mel.size(1), 1,
                         output_mel.size(2))), 1)
        output = output.reshape(output.size(0), output.size(3), output.size(1))

        # stack recurrent layers
        for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):

            if self.rnn == 'LSTM':
                # initial hidden and cell states
                h = torch.zeros(
                    self.num_directions_,
                    batch_size,
                    hidden_dim,
                    device=device,
                    requires_grad=False)
                c = torch.zeros(
                    self.num_directions_,
                    batch_size,
                    hidden_dim,
                    device=device,
                    requires_grad=False)
                hidden = (h, c)

            elif self.rnn == 'GRU':
                # initial hidden state
                hidden = torch.zeros(
                    self.num_directions_,
                    batch_size,
                    hidden_dim,
                    device=device,
                    requires_grad=False)

            # apply current recurrent layer and get output sequence
            output, _ = layer(output, hidden)

            # average both directions in case of bidirectional layers
            if self.bidirectional:
                output = .5 * (
                    output[:, :, :hidden_dim] + output[:, :, hidden_dim:])

        if self.pooling is not None:
            if self.pooling == 'sum':
                output = output.sum(dim=1)
            elif self.pooling == 'max':
                output, _ = output.max(dim=1)

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = torch.tanh(output)

        # apply final classification layer
        output = self.final_layer_(output)

        output = self.activation_(output)

        if return_intermediate is None:
            return output
        return output, intermediate


class StackedNoGaborRNNRawWave(Model):
    """Stacked recurrent neural network with Gabor Learnable first Layer with
    Raw waveform computations

    Parameters
    ----------
    specifications : `dict`
        Provides model IO specifications using the following data structure:
            {'X': {'dimension': DIMENSION},
             'y': {'classes': CLASSES},
             'task': TASK_TYPE}
        where
            * DIMENSION is the input feature dimension
            * CLASSES is the list of (human-readable) output classes
            * TASK_TYPE is either TASK_MULTI_CLASS_CLASSIFICATION,
            TASK_REGRESSION, or TASK_MULTI_LABEL_CLASSIFICATION.
            Depending on which task is
                adressed, the final activation will vary. Classification relies
                on log-softmax, multi-label classificatition and regression use
                sigmoid.
    instance_normalize : boolean, optional
        Apply mean/variance normalization on input sequences.
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent : list, optional
        List of hidden dimensions of stacked recurrent layers. Defaults to
        [16, ], i.e. one recurrent layer with hidden dimension of 16.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    pooling : {None, 'sum', 'max'}
        Apply temporal pooling before linear layers. Defaults to no pooling.
        This is useful for tasks expecting just one label per sequence.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, ], i.e.
        one linear layer with hidden dimension of 16.
    """

    def init(self,
             instance_normalize=False,
             rnn='LSTM',
             recurrent=[
                 16,
             ],
             bidirectional=False,
             linear=[
                 16,
             ],
             pooling=None,
             num_gabor_filters=64,
             kernel_size=(25, 25),
             stride=[1, 1],
             n_mels=128,
             waveform_normalize=True,
             n_fft=2048,
             window='hamming',
             duration=0.025,
             step=0.001,
             task_duration=2.0,
             sample_rate=16000,
             pre_lstm_layer_bool=False,
             window_pre_lstm=7,
             mel_trainable=False,
             stft_trainable=False,
             pre_lstm_compression_dim=64,
             dropout_pre_lstm=0.0,
             gabor_mode='real',
             Spectrogram_input='MEL',
             direct_mel=False):

        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.step = step
        self.duration = self.n_fft / self.sample_rate
        self.hop_length = int(self.step * self.sample_rate)
        self.num_gabor_filters = num_gabor_filters
        self.kernel_size = kernel_size
        self.waveform_normalize = waveform_normalize
        self.pre_lstm_layer_bool = pre_lstm_layer_bool
        self.window_pre_lstm = window_pre_lstm
        self.stft_trainable = stft_trainable
        self.n_mels = n_mels
        self.dropout_pre_lstm = dropout_pre_lstm
        self.Spectrogram_input = Spectrogram_input
        self.Spectrogram_input = Spectrogram_input
        if self.Spectrogram_input == 'STFT':
            self.n_mels = int(self.n_fft / 2) + 1

        if self.waveform_normalize:
            self.waveform_normalize_ = torch.nn.InstanceNorm1d(1, affine=True)

        n_features = self.specifications['X']['dimension']
        self.n_features_ = n_features

        n_classes = len(self.specifications['y']['classes'])
        self.n_classes_ = n_classes

        self.instance_normalize = instance_normalize

        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.pre_lstm_compression_dim = pre_lstm_compression_dim
        self.stride = stride
        self.pooling = pooling
        self.linear = linear
        self.duration = duration
        self.task_duration = task_duration
        self.gabor_mode = gabor_mode
        cuda = torch.device('cuda')
        if self.Spectrogram_input == 'STFT':
            self.mel_layer = Spectrogram.STFT(
                trainable=self.stft_trainable,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                freq_scale='linear',
                center=False,
                device=cuda)
        elif self.Spectrogram_input == 'CQT':
            self.mel_layer = Spectrogram.CQT1992v2(
                sr=self.sample_rate,
                bins_per_octave=24,
                n_bins=self.n_mels,
                center=True,
                fmin=125,
                hop_length=self.hop_length,
                pad_mode='constant',
                device=cuda)
        elif self.Spectrogram_input == 'MEL':
            self.mel_layer = Spectrogram.MelSpectrogram(
                trainable_mel=mel_trainable,
                trainable_STFT=stft_trainable,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=False,
                device=cuda)

        self.lay_ampl = AmplitudeToDB(top_db=80)
        self.num_directions_ = 2 if self.bidirectional else 1

        self.gabor_layer = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_gabor_filters,
            kernel_size=self.kernel_size,
            padding=(int(self.kernel_size[0] // 2),
                     int(self.kernel_size[1] // 2)),
            stride=self.stride)
        self.gabor_activation = nn.ReLU(inplace=True)
        self.dropout_pre_lstm_layer = nn.Dropout2d(p=self.dropout_pre_lstm)
        if self.gabor_mode == 'concat':
            input_dim_conv = 2 * self.n_mels * self.num_gabor_filters
        else:
            input_dim_conv = self.n_mels * self.num_gabor_filters
        if self.variant_architecture:
            if self.gabor_mode == 'concat':
                input_dim_conv = 2 * self.n_mels
            else:
                input_dim_conv = self.n_mels
        if self.pre_lstm_layer_bool:
            if self.variant_architecture:
                self.pre_lstm_layer = nn.Conv2d(
                    self.num_gabor_filters,
                    self.pre_lstm_compression_dim,
                    (input_dim_conv, window_pre_lstm),
                    stride=(window_pre_lstm, 1),
                    padding=int(window_pre_lstm // 2))
            else:
                self.pre_lstm_layer = nn.Conv2d(
                    1,
                    self.pre_lstm_compression_dim,
                    (input_dim_conv, window_pre_lstm),
                    stride=(window_pre_lstm, 1),
                    padding=int(window_pre_lstm // 2))
            input_dim = self.n_mels
        # create list of recurrent layers
        self.recurrent_layers_ = []
        # This must be ensured to well-declare the padding in
        assert kernel_size[0] % 2 == 1, \
            'Should be odd size for gabor kernels'
        # input_dim = int(self.n_mels * self.num_gabor_filters / stride[0])
        # if self.gabor_mode == 'concat':
        #     input_dim = self.pre_lstm_compression_dim
        # else:
        if self.gabor_mode == 'pass':
            input_dim = self.n_mels
        elif self.gabor_mode == 'concat_spectrogram':
            input_dim = self.n_mels + self.pre_lstm_compression_dim
        else:
            input_dim = self.pre_lstm_compression_dim
        if self.mfcc_concat:
            input_dim += 20
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    bidirectional=self.bidirectional,
                    batch_first=True)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(
                    input_dim,
                    hidden_dim,
                    bidirectional=self.bidirectional,
                    batch_first=True)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

        self.last_hidden_dim_ = input_dim

        self.final_layer_ = nn.Linear(self.last_hidden_dim_, self.n_classes_)

        # Define output activation
        self.activation_ = self.task.default_activation

    @property
    def task(self):
        """Type of task addressed by the model

        Shortcut for self.specifications['task']
        """
        return self.specifications['task']

    @property
    def classes(self):
        return self.specifications['y']['classes']

    @property
    def n_classes(self):
        return len(self.specifications['y']['classes'])

    @staticmethod
    def get_alignment(task: Task, **kwargs):
        return 'center'

    @staticmethod
    def get_resolution(task: Task,
                       sample_rate=16000,
                       duration=0.025,
                       step=0.005,
                       **kwargs) -> Resolution:
        """
        """
        return SlidingWindow(
            start=-.5 * duration, duration=duration, step=step)

    def forward(self, waveforms, return_intermediate=None):
        """

        Parameters
        ----------
        sequences : (batch_size, n_samples, n_features) `torch.tensor`
            Batch of sequences.

        Returns
        -------
        predictions : `torch.tensor`
            Shape is (batch_size, n_samples, n_classes) without pooling, and
            (batch_size, n_classes) with pooling.
        """
        if isinstance(waveforms, PackedSequence):
            msg = (f'{self.__class__.__name__} does not support batches '
                   f'containing sequences of variable length.')
            raise ValueError(msg)

        batch_size, n_features, device = get_info(waveforms)

        if n_features != self.n_features_:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features_))

        output = waveforms
        if self.waveform_normalize:
            output = output.transpose(1, 2)
            output = F.instance_norm(output)
            # output = output.transpose(1, 2)
        if self.mfcc_concat:
            output_mfcc = self.mfcc_layer(output)
        output = self.mel_layer(output)
        output = self.lay_ampl(output)
        length_slice = int(self.sample_rate * self.task_duration)

        n_frames = int(
            (length_slice + int(self.step * self.sample_rate) - self.n_fft) //
            float(int(self.step * self.sample_rate)))
        p1d = (1, np.max(
            int(self.task_duration / self.step - n_frames) - 1, 0))
        if self.Spectrogram_input != 'CQT':
            if self.mfcc_concat:
                output_mfcc = F.pad(output_mfcc, p1d, "constant", 0)
                output = F.pad(output, p1d, "constant", 0)
            else:
                output_mel = F.pad(output, p1d, "constant", 0)
                output = F.pad(output, p1d, "constant", 0)
        else:
            output = output[:, :, :-1]

        output = output.reshape(output.size(0), 1, self.n_mels, -1)
        output = self.gabor_layer(output)
        output_shape = output.shape
        output = self.dropout_pre_lstm_layer(output)
        if self.variant_architecture:
            if self.gabor_mode == 'concat':
                output = output.reshape(
                    output.size(0), self.num_gabor_filters,
                    2 * int(self.n_mels / self.stride[0]), output.size(3))
            else:
                output = output.reshape(
                    output.size(0), self.num_gabor_filters,
                    int(self.n_mels / self.stride[0]), output.size(3))
        else:
            if self.gabor_mode == 'concat':
                output = output.reshape(
                    output.size(0), 1, 2 * int(
                        self.n_mels * self.num_gabor_filters / self.stride[0]),
                    output.size(3))
            elif self.gabor_mode == 'pass':
                output = output.reshape(
                    output.size(0), int(self.n_mels), 1, output.size(2))
            else:
                output = output.reshape(
                    output.size(0), 1,
                    int(self.n_mels * self.num_gabor_filters / self.stride[0]),
                    output.size(3))
        if self.gabor_mode != 'pass':
            output = self.gabor_activation(output)
        if self.pre_lstm_layer_bool:
            # Pre-lstm-layer
            output = self.pre_lstm_layer(output)
            # apply non-linear activation function
            output = F.relu(output, inplace=True)
            if self.gabor_mode == 'concat_spectrogram':
                output = torch.cat(
                    (output,
                     output_mel.reshape(
                         output_mel.size(0), output_mel.size(1), 1,
                         output_mel.size(2))), 1)
        output = output.reshape(output.size(0), output.size(3), output.size(1))

        # stack recurrent layers
        for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):

            if self.rnn == 'LSTM':
                # initial hidden and cell states
                h = torch.zeros(
                    self.num_directions_,
                    batch_size,
                    hidden_dim,
                    device=device,
                    requires_grad=False)
                c = torch.zeros(
                    self.num_directions_,
                    batch_size,
                    hidden_dim,
                    device=device,
                    requires_grad=False)
                hidden = (h, c)

            elif self.rnn == 'GRU':
                # initial hidden state
                hidden = torch.zeros(
                    self.num_directions_,
                    batch_size,
                    hidden_dim,
                    device=device,
                    requires_grad=False)

            # apply current recurrent layer and get output sequence
            output, _ = layer(output, hidden)

            # average both directions in case of bidirectional layers
            if self.bidirectional:
                output = .5 * (
                    output[:, :, :hidden_dim] + output[:, :, hidden_dim:])

        if self.pooling is not None:
            if self.pooling == 'sum':
                output = output.sum(dim=1)
            elif self.pooling == 'max':
                output, _ = output.max(dim=1)

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = torch.tanh(output)

        # apply final classification layer
        output = self.final_layer_(output)

        output = self.activation_(output)

        if return_intermediate is None:
            return output
        return output, intermediate


