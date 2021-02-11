Learning spectro-temporal representations of complex sounds with parameterized neural networks
=============

![image](https://user-images.githubusercontent.com/8396578/117989727-da7d8180-b33c-11eb-9d0e-ad9b4f87d2d0.png)


**c** is an open-source package to replicate the experiments in [^1]. 
 
 
## Dependencies

The main dependencies of learnable-strf are :
* [NumPy](https://numpy.org/>) (>= 1.10)
* [pytorch](https://www.pytorch.org/>) (== 1.3.0)
* [nnaudio](https://github.com/KinWaiCheuk/nnAudio/>) (== 1.3.0) 
* [pyannote.core](http://pyannote.github.io/pyannote-core/>) (>= 4.1)
* [pyannote.audio](http://pyannote.github.io/pyannote-audio/>) (== 2.0a1+60.gc683897) (installed with the shell)
* [pyannote.database](http://pyannote.github.io/pyannote-database/>) (== 4.0.1+5.g8394991)
* [pyannote.pipeline](http://pyannote.github.io/pyannote-pipeline/>) (==  1.5)
* [pyannote.metrics](http://pyannote.github.io/pyannote-metrics/>) (==  2.3)


 ## Implementation of the Learnable STRF

The Learnable STRF can be easily implemented in pytorch and is inspired by the implementation from this [package](https://github.com/iKintosh/GaborNet)

We used the [nnAudio](https://github.com/KinWaiCheuk/nnAudio) package to obtain the log Mel Filterbanks.
```python
from typing import Optional
from typing import Text

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio import Spectrogram
from torch.nn.utils.rnn import PackedSequence

from torch.nn.modules.conv import _ConvNd
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

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
               n_features=64):

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
      self.freq = (np.pi / 2) * 1.41**(
          -np.random.uniform(0, 5, size=(out_channels, in_channels)))

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
                  g = g * torch.cos(freq * rot_gamma)
              else:
                  g = g * torch.sin(freq * rot_gamma)
              g = g / (2 * np.pi * sigma_x * sigma_y)
              weight[i, j] = g
              self.weight.data[i, j] = g
      weight = weight.to(device)
      return F.conv2d(sequences, weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

```

## Replication of the engineering experiments

### Speech Activity detection
All the experiments for Speech Activity Detection are run with the `pyannote` ecosystem.
#### Databases

The AMI corpus can be obtained freely from [https://groups.inf.ed.ac.uk/ami/corpus/](https://groups.inf.ed.ac.uk/ami/corpus/).

The CHIME5 corpus can be obtained freely from [url](http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME5/data.html)

The protocol databases for the train/dev/test are _AMI.SpeakerDiarization.MixHeadset_ and  _CHiME5.SpeakerDiarization.U01_ and can be obtained via [pyannote.database](https://github.com/pyannote/pyannote-database) and the `pip`commands for [AMI](https://github.com/pyannote/pyannote-db-odessa-ami) and for  [CHIME5] it is required the following lines to your `.pyannote/database.yml`.


```shell script
  CHiME5:
    SpeakerDiarization:
      U01:
        train:
          annotation: /export/fs01/jsalt19/databases/CHiME5/train/allU01_train.rttm
          annotated: /export/fs01/jsalt19/databases/CHiME5/train/allU01_train.uem
        development:
          annotation: /export/fs01/jsalt19/databases/CHiME5/dev/allU01_dev.rttm
          annotated: /export/fs01/jsalt19/databases/CHiME5/dev/allU01_dev.uem
        test:
          annotation: /export/fs01/jsalt19/databases/CHiME5/test/allU01_test.rttm
          annotated: /export/fs01/jsalt19/databases/CHiME5/test/allU01_test.uem
 ```
 
 ### Speaker Verification
 
 We followed the protocol from [JM Coria et al.](https://github.com/juanmc2005/SpeakerEmbeddingLossComparison), and injected the network _STRFTDNN_ instead of _SincTDNN_.
 
 ### Urban Sound Classification
 
 We followed the protocol from [Arnault et al.](https://github.com/multitel-ai/urban-sound-classification-and-comparison), and just modified the Pann architecture by injecting the STRFConv2D on top of the Mel Filterbanks. 
 
 #### Models
 
 The models to run the Speech Activity Detection and Speaker Identification are in the file `models.py`. This file replaces the models.py in the pyannote.audio package to use the `Learnable STRF`
  
  
  #### Acknowledgements
  We are very grateful to authors from Pyannote, nnAudio, urban sound sound package, Theunissen's group, Shamma's group for the open source packages and datasets which made possible this work. 
 

 
 [1] Riad R., Karadyi J., Bachoud-Lévi AC., Dupoux, E.
   *Learning spectro-temporal representations of complex sounds with parameterized neural networks.*
   The Journal of the Acoustical Society of America
