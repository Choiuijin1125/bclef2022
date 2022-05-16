import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import random
import numpy as np
import torchaudio.transforms as T

from torchlibrosa.augmentation import SpecAugmentation
from pytorch_lightning.core.lightning import LightningModule

from .model_utils import GeM, Mixup, Selective_Mixup

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()
        
def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output
        
        
class AttBlockV2(LightningModule):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.cla(x)
        logit = torch.sum(norm_att * cla, dim=2)
        x = self.nonlinear_transform(logit)
        #         norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        #         print(norm_att.shape)
        #         cla = self.nonlinear_transform(self.cla(x))
        #         print(cla.shape)
        #         x = torch.sum(norm_att * cla, dim=2)
        #         print(x.shape)

        return x, logit, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        
class TimmSEDGPU(LightningModule):
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
        #self.global_pooling = GeM()
        base_model = timm.create_model(
            cfg.base_model.name,
            pretrained=cfg.base_model.pretrained,
            in_chans=cfg.base_model.in_channels,
        )
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, cfg.num_classes, activation="sigmoid"
        )

        self.mixup = Mixup(mix_beta=cfg.mix_beta)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def forward(self, x, y, weight):
        
        with torch.cuda.amp.autocast(False):
            
            x = self.logmel_extractor(x).unsqueeze(1).transpose(2, 3)

        frames_num = x.size(2)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
            
        if self.training:
            if np.random.random() <= self.cfg.mixup:
                
                x, y, weight = self.mixup(x, y, weight)
                
            if np.random.random() <= self.cfg.mixup2:
                
                x, y, weight = self.mixup(x, y, weight)
            
            if np.random.random() < 0.25:
                x = self.spec_augmenter(x)
            

        x = x.transpose(2, 3)
        # (batch_size, channels, frames, freq)
        x = self.encoder(x)
        # (batch_size, channels, frames)
        x = torch.mean(x, dim=3)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, logit, segmentwise_logit) = self.att_block(x)
        # logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = segmentwise_logit.transpose(1, 2)
        
        segmentwise_output = torch.sigmoid(segmentwise_logit)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
            "targets": y,
            "weight": weight
        }

        return output_dict
    
    
class SelectiveTimmSEDGPU(LightningModule):
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
        #self.global_pooling = GeM()
        base_model = timm.create_model(
            cfg.base_model.name,
            pretrained=cfg.base_model.pretrained,
            in_chans=cfg.base_model.in_channels,
        )
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, cfg.num_classes, activation="sigmoid"
        )

        self.mixup = Mixup(mix_beta=cfg.mix_beta)
        self.selective_mixup = Selective_Mixup(mix_beta=cfg.mix_beta)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def forward(self, x, y, weight, selective_x=None, selective_y=None, selective_weight=None):
        
        
        with torch.cuda.amp.autocast(False):
            
            x = self.logmel_extractor(x).unsqueeze(1).transpose(2, 3)
            if self.training:
                selective_x = self.logmel_extractor(selective_x).unsqueeze(1).transpose(2, 3)
            
        frames_num = x.size(2)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            selective_x = selective_x.transpose(1, 3)
            selective_x = self.bn0(selective_x)
            selective_x = selective_x.transpose(1, 3)
            
        if self.training:
            x = x.permute(0, 2, 1, 3)
            selective_x = selective_x.permute(0, 2, 1, 3)
            
            if np.random.random() <= self.cfg.selective_mixup:
                
                x, y, weight = self.selective_mixup(x, y, weight, selective_x, selective_y, selective_weight)
                
            if np.random.random() <= self.cfg.mixup:
                
                x, y, weight = self.mixup(x, y, weight)
                
            if np.random.random() <= self.cfg.mixup2:
                
                x, y, weight = self.mixup(x, y, weight)
            
            x = x.permute(0, 2, 1, 3)
            
            if np.random.random() < self.cfg.spec_aug:
                
                x = self.spec_augmenter(x)
            
            
            
        x = x.transpose(2, 3)
        # (batch_size, channels, frames, freq)
        x = self.encoder(x)
        # (batch_size, channels, frames)
        x = torch.mean(x, dim=3)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, logit, segmentwise_logit) = self.att_block(x)
        # logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = segmentwise_logit.transpose(1, 2)
        
        segmentwise_output = torch.sigmoid(segmentwise_logit)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
            "targets": y,
            "weight": weight
        }

        return output_dict
    
    
class SelectivePartTimmSEDGPU(LightningModule):
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
        #self.global_pooling = GeM()
        base_model = timm.create_model(
            cfg.base_model.name,
            pretrained=cfg.base_model.pretrained,
            in_chans=cfg.base_model.in_channels,
        )
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, cfg.num_classes, activation="sigmoid"
        )

        self.mixup = Mixup(mix_beta=cfg.mix_beta)
        self.selective_mixup = Selective_Mixup(mix_beta=cfg.mix_beta)
        self.factor = int(cfg.duration / 5.0)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)
        
        

    def forward(self, x, y, weight, selective_x=None, selective_y=None, selective_weight=None):
        
        bs, time = x.shape
        x = x.reshape(bs * self.factor, time // self.factor)
        
        if self.training:
            selective_x = selective_x.reshape(bs * self.factor, time // self.factor)
       
        with torch.cuda.amp.autocast(False):
            
            x = self.logmel_extractor(x).unsqueeze(1).transpose(2, 3)
            
            if self.training:
                selective_x = self.logmel_extractor(selective_x).unsqueeze(1).transpose(2, 3)
            
        frames_num = x.size(2)

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
            
            
        x = x.transpose(2, 3)
        # (batch_size, channels, frames, freq)
        x = self.encoder(x)
        # (batch_size, channels, frames)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b // self.factor, self.factor * t, c, f)
        x = x.permute(0, 2, 1, 3)   
        x = torch.mean(x, dim=3)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, logit, segmentwise_logit) = self.att_block(x)
        # logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = segmentwise_logit.transpose(1, 2)
        
        segmentwise_output = torch.sigmoid(segmentwise_logit)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
            "targets": y,
            "weight": weight
        }

        return output_dict        