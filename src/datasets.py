from torch.utils.data import Dataset
import torch
import numpy as np
import librosa
import ast
import pandas as pd
import audiomentations as AA
from hydra.utils import instantiate

class WaveformDatasetRating(torch.utils.data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 cfg,
                 mode='train'):
        self.df = df.copy()
        self.mode = mode
        self.cfg = cfg
        self.df = self.setup_df()
        
        if cfg.augment is not None:
            self.train_aug = AA.Compose(
                [
                    instantiate(cfg.augment.backgroundnoise),
#                     instantiate(cfg.augment.backgroundnoise2),
                    
                ]
            )

            
    def setup_df(self):
        df = self.df.copy()
        
        if self.mode == "train":
            df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)
            
        else:
            df["weight"] = 1
            
        return df
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        
        wav_path = sample["file_path"]
        labels = sample["new_target"]
        weight = sample["weight"]
        offset = sample["offset"]
        
        duration = self.cfg.duration
        wav = self.load_one(wav_path, offset, duration)
        
        if wav.shape[0] < (self.cfg.duration * self.cfg.sr):
            pad = self.cfg.duration * self.cfg.sr - wav.shape[0]
            wav = np.pad(wav, (0, pad))        
        
        targets = np.zeros(len(self.cfg.target_columns), dtype=float)
        
        for ebird_code in labels.split():
            targets[self.cfg.target_columns.index(ebird_code)] = 1.0
            
        if self.cfg.label_smoothing > 0:
            targets = np.clip(targets + self.cfg.label_smoothing, 0, 1)
        
        if self.mode == "train":
            if self.train_aug:
                wav = self.train_aug(samples=wav, sample_rate=self.cfg.sr)
                
        targets = targets.astype(np.float32)
        weight = weight.astype(np.float32)
        #wav = wav
        
        return {
            "image": torch.tensor(wav),
            "targets": torch.tensor(targets),
            "weight": torch.tensor(weight)
        }
    
    def load_one(self, wav_path, offset, duration):
        try:
            wav, sr = librosa.load(wav_path, sr=None, offset=offset, duration=duration)
        except:
            print("FAIL READING rec", wav_path)

        return wav    

class WaveformDatasetRatingV2(torch.utils.data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 cfg,
                 mode='train'):
        self.df = df.copy()
        self.mode = mode
        self.cfg = cfg
        self.df = self.setup_df()
        
        if cfg.augment is not None:
            self.train_aug = AA.Compose(
                [
                    instantiate(cfg.augment.backgroundnoise),
                    instantiate(cfg.augment.backgroundnoise2),
                    instantiate(cfg.augment.backgroundnoise3),
                    
                ]
            )
            
    def setup_df(self):
        df = self.df.copy()
        
        if self.mode == "train":
            df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)
            
        else:
            df["weight"] = 1
            
        return df
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]

        
        wav_path = sample["file_path"]
        labels = sample["new_target"]
        weight = sample["weight"]       
        wav_len = sample["length"]
        wav_len_sec = wav_len / self.cfg.sr        
        duration = self.cfg.duration
        max_offset = wav_len_sec - duration
        max_offset = max(max_offset, 1)
        offset = np.random.randint(max_offset)
        
        wav = self.load_one(wav_path, offset, duration)
        
        if wav.shape[0] < (self.cfg.duration * self.cfg.sr):
            pad = self.cfg.duration * self.cfg.sr - wav.shape[0]
            wav = np.pad(wav, (0, pad))        
        
        targets = np.zeros(len(self.cfg.target_columns), dtype=float)
        
        for ebird_code in labels.split():
            targets[self.cfg.target_columns.index(ebird_code)] = 1.0
                    
        if self.mode == "train":
            if self.train_aug:
                wav = self.train_aug(samples=wav, sample_rate=self.cfg.sr)
                
            if self.cfg.label_smoothing > 0:
                targets = np.clip(targets + self.cfg.label_smoothing, 0, 1)                
                
        targets = targets.astype(np.float32)
        weight = weight.astype(np.float32)
        #wav = wav
        
        return {
            "image": torch.tensor(wav),
            "targets": torch.tensor(targets),
            "weight": torch.tensor(weight)
        }
    
    def load_one(self, wav_path, offset, duration):
        try:
            wav, sr = librosa.load(wav_path, sr=None, offset=offset, duration=duration)
        except:
            print("FAIL READING rec", wav_path)

        return wav    

        