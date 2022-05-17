import torch

from models.baseline_model import CNN
from models.conformer.conformer_encoder import ConformerEncoder
from models.transformer.encoder import Encoder as TransformerEncoder
from torch.distributions import Beta
from torch.nn.parameter import Parameter
from torchlibrosa.augmentation import SpecAugmentation

import torchaudio.transforms as T
import torch.nn as nn
import numpy as np

class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight
        
class Selective_Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Selective_Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight, selective_X, selective_Y, selective_weight):

        bs = X.shape[0]
        selective_bs = selective_X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(selective_bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * selective_X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * selective_X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * selective_X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * selective_Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * selective_weight[perm]
            return X, Y, weight                

class SEDModel(torch.nn.Module):
    def __init__(
        self,
        n_class,
        cfg,
        cnn_kwargs=None,
        encoder_kwargs=None,
        encoder_type="Conformer",
        pooling="token",
        layer_init="pytorch",
        
    ):
        super(SEDModel, self).__init__()
        self.cfg = cfg
        self.factor = int(cfg.duration / 5.0)
        self.mixup = Mixup(mix_beta=cfg.mix_beta)
        self.selective_mixup = Selective_Mixup(mix_beta=cfg.mix_beta)        
        self.cnn = CNN(n_in_channel=1, **cnn_kwargs)
        input_dim = self.cnn.nb_filters[-1]
        adim = encoder_kwargs["adim"]
        self.pooling = pooling

        if encoder_type == "Transformer":
            self.encoder = TransformerEncoder(input_dim, **encoder_kwargs)
        elif encoder_type == "Conformer":
            self.encoder = ConformerEncoder(input_dim, **encoder_kwargs)
        else:
            raise ValueError("Choose encoder_type in ['Transformer', 'Conformer']")

        self.classifier = torch.nn.Linear(adim, n_class)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, n_class)
            self.sigmoid = torch.sigmoid
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            self.linear_emb = torch.nn.Linear(1, input_dim)

        self.reset_parameters(layer_init)
        
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
        self.bn0 = nn.BatchNorm2d(cfg.n_mels)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)        

    def forward(self, x, y, weight, mask=None):
        
        bs, time = x.shape
        x = x.reshape(bs * self.factor, time // self.factor)

        with torch.cuda.amp.autocast(False):
            
            x = self.logmel_extractor(x).unsqueeze(1).transpose(2, 3)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)            

        if self.training:
            
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)           
#             selective_x = selective_x.permute(0, 2, 1, 3)
            
            x = x.reshape(b // self.factor, self.factor * t, c, f)
#             selective_x = selective_x.reshape(b // self.factor, self.factor * t, c, f)

            selective_x = x[b // (self.factor * 2):]
            selective_y = y[b // (self.factor * 2):]
            selective_weight = weight[b // (self.factor * 2):]

            x = x[:b // (self.factor * 2)]
            y = y[:b // (self.factor * 2)]
            weight = weight[:b // (self.factor * 2)]
            
            
            if np.random.random() <= self.cfg.selective_mixup:
                
                x, y, weight = self.selective_mixup(x, y, weight, selective_x, selective_y, selective_weight)
                
            if np.random.random() <= self.cfg.mixup:
                
                x, y, weight = self.mixup(x, y, weight)
                
            if np.random.random() <= self.cfg.mixup2:
                
                x, y, weight = self.mixup(x, y, weight)
                            
            x = x.reshape(b // 2, t, c, f)
            x = x.permute(0, 2, 1, 3)
            
            if np.random.random() < self.cfg.spec_aug:
                
                x = self.spec_augmenter(x)        
        
        # input
        
        x = self.cnn(x)
        x = torch.mean(x, dim=-1).permute(0, 2, 1)

        if self.pooling == "token":
            tag_token = self.linear_emb(torch.ones(x.size(0), 1, 1).to("cuda"))
            x = torch.cat([tag_token, x], dim=1)
            
        x, _ = self.encoder(x, mask)
        b, s, e = x.shape
        
        x = x.reshape(b//self.factor, s*self.factor, e)
        if self.pooling == "attention":
            strong = self.classifier(x)
            sof = self.dense(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (torch.sigmoid(strong) * sof).sum(1) / sof.sum(1)  # [bs, nclass]
            # Convert to logit to calculate loss with bcelosswithlogits
            weak = torch.log(weak / (1 - weak))
        elif self.pooling == "token":
            x = self.classifier(x)
            weak = x[:, 0, :]
            strong = x[:, 1:, :]

        elif self.pooling == "auto":
            strong = self.classifier(x)
            weak = self.autopool(strong)

        output_dict = {
            "logit": weak,
            "framewise_logit": strong,
            "clipwise_output": torch.sigmoid(weak),
            "targets": y,
            "weight": weight
        }            
            
        return output_dict

    def reset_parameters(self, initialization: str = "pytorch"):
        if initialization.lower() == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if initialization.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif initialization.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif initialization.lower() == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif initialization.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown initialization: {initialization}")
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()
