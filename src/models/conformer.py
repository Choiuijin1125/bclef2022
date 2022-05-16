import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import torchaudio.transforms as T

from torchlibrosa.augmentation import SpecAugmentation
from pytorch_lightning.core.lightning import LightningModule
from conformer import Conformer
from .model_utils import GeM, Mixup


class ConformerGPU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.logmel_extractor = nn.Sequential(
            T.MelSpectrogram(sample_rate=cfg.sr,                 
                             n_fft=cfg.window_size,
                             hop_length=cfg.hop_size,
                             power=2.0, 
                             n_mels=cfg.n_mels,
                             f_min=cfg.fmin,
                             f_max=cfg.fmax,
                             normalized=False),
            T.AmplitudeToDB(top_db=cfg.top_db)
        )
        
        self.spec_augmenter = SpecAugmentation(time_drop_width=64//2, time_stripes_num=2,
                                               freq_drop_width=8//2, freq_stripes_num=2)


        self.model = Conformer(**cfg.encoder)
        self.mixup = Mixup(mix_beta=cfg.mix_beta)
        

    def forward(self, x, y, weight, x_len):
        
        with torch.cuda.amp.autocast(False):
            
            x = self.logmel_extractor(x).unsqueeze(1).transpose(2, 3)
            
        if self.training:
            if np.random.random() <= self.cfg.mixup:
                
                x, y, weight = self.mixup(x, y, weight)
                
            if np.random.random() <= self.cfg.mixup2:
                
                x, y, weight = self.mixup(x, y, weight)
            
            if np.random.random() < self.cfg.spec_aug:
                x = self.spec_augmenter(x)
        
        x = torch.mean(x, dim=1)
        encoder_outputs, encoder_output_lengths = self.model.encoder(x, x_len)
        outputs = self.model.fc(encoder_outputs)
        #outputs, _ = torch.max(outputs, dim=1)
        outputs = torch.mean(outputs, dim=1)
        output_dict = {
            "framewise_output": outputs,
            "clipwise_output": torch.sigmoid(outputs),
            "logit": outputs,
            "framewise_logit": outputs,
            "targets": y,
            "weight": weight            
        }

        return output_dict    