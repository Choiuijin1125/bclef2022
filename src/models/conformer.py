import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import torchaudio.transforms as T

from torchlibrosa.augmentation import SpecAugmentation
from pytorch_lightning.core.lightning import LightningModule
from conformer import Conformer
from .model_utils import GeM, Mixup, Selective_Mixup


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
    
    
    
class SelectivePartConformerGPU(LightningModule):
    def __init__(self, cfg
                 ):
        super().__init__()
        # melextractor
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

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(cfg.n_mels)

        self.model = Conformer(**cfg.encoder)
        self.mixup = Mixup(mix_beta=cfg.mix_beta)
        self.selective_mixup = Selective_Mixup(mix_beta=cfg.mix_beta)
        self.factor = int(cfg.duration / 5.0)
        

    def forward(self, x, y, weight, selective_x=None, selective_y=None, selective_weight=None):
        
        bs, time = x.shape
        x = x.reshape(bs * self.factor, time // self.factor)
        
        if self.training:
            selective_x = selective_x.reshape(bs * self.factor, time // self.factor)
       
        with torch.cuda.amp.autocast(False):
            
            x = self.logmel_extractor(x).unsqueeze(1).transpose(2, 3)
            
            if self.training:
                selective_x = self.logmel_extractor(selective_x).unsqueeze(1).transpose(2, 3)
            

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            selective_x = selective_x.transpose(1, 3)
            selective_x = self.bn0(selective_x)
            selective_x = selective_x.transpose(1, 3)
            
        if self.training:
            b, c, t, f = x.shape
            
            x = x.permute(0, 2, 1, 3)           
            selective_x = selective_x.permute(0, 2, 1, 3)
            
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            selective_x = selective_x.reshape(b // self.factor, self.factor * t, c, f)
            
            if np.random.random() <= self.cfg.selective_mixup:
                
                x, y, weight = self.selective_mixup(x, y, weight, selective_x, selective_y, selective_weight)
                
            if np.random.random() <= self.cfg.mixup:
                
                x, y, weight = self.mixup(x, y, weight)
                
            if np.random.random() <= self.cfg.mixup2:
                
                x, y, weight = self.mixup(x, y, weight)
                            
            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 1, 3)
            
            if np.random.random() < self.cfg.spec_aug:
                
                x = self.spec_augmenter(x)
            
        #x = x.transpose(2, 3)
        x = x.squeeze(1)
        x_len = torch.LongTensor([x.shape[1]]*x.shape[0])
        x, _ = self.model.encoder(x, x_len)
        b, seq, emb = x.shape
        # (batch_size, channels, frames, freq)
        x = x.reshape(b//self.factor, self.factor*seq, emb)
        
        outputs = self.model.fc(x)
        #outputs, _ = torch.max(outputs, dim=1)
        outputs = torch.mean(outputs, dim=2)
        
        output_dict = {
            "framewise_output": outputs,
            "clipwise_output": torch.sigmoid(outputs),
            "logit": outputs,
            "framewise_logit": outputs,
            "targets": y,
            "weight": weight            
        }

        return output_dict            