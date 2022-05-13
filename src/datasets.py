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
                    instantiate(cfg.augment.backgroundnoise)
                ]
            )
#         if mode == 'train':
#             self.wave_transforms = Compose(
#                 [
#                     Normalize(mean, std),
#                 ]
#             )
#             self.df = self.df[self.df["rating"] >= self.cfg.min_rating]
#         else:
#             self.wave_transforms = Compose(
#                 [
#                     Normalize(mean, std),
#                 ]
#             )

            
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

    
def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


tr_collate_fn = None
val_collate_fn = None


class CustomDataset(Dataset):
    def __init__(self, df, cfg, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()

        self.bird2id = {bird: idx for idx, bird in enumerate(cfg.birds)}
        if self.mode == "train":
            self.data_folder = cfg.train_data_folder
            self.df = self.df[self.df["rating"] >= self.cfg.min_rating]
        elif self.mode == "val":
            self.data_folder = cfg.val_data_folder
        elif self.mode == "test":
            self.data_folder = cfg.test_data_folder

        self.fns = self.df["filename"].unique()

        self.df = self.setup_df()

        self.aug_audio = cfg.train_aug

    def setup_df(self):
        df = self.df.copy()
        
        if self.mode == "train":
            df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)
            
        else:
            df["weight"] = 1
            
        return df
    
    def __getitem__(self, idx):

        if self.mode == "train":
            row = self.df.iloc[idx]
            fn = row["primary_label"] + "/" + row["filename"]
            label = row[[f"t{i}" for i in range(self.cfg.n_classes)]].values
            weight = row["weight"]
            fold = row["fold"]

            wav_len = row["length"]
            parts = 1
        else:
            fn = self.fns[idx]
            row = self.df.get_group(fn)
            label = row[[f"t{i}" for i in range(self.cfg.n_classes)]].values
            wav_len = None
            parts = label.shape[0]
            fold = -1
            weight = 1

        if self.mode == "train":
            wav_len_sec = wav_len / self.cfg.sample_rate
            duration = self.cfg.duration
            max_offset = wav_len_sec - duration
            max_offset = max(max_offset, 1)
            offset = np.random.randint(max_offset)
        else:
            offset = 0.0
            duration = None

        wav = self.load_one(fn, offset, duration)

        if wav.shape[0] < (self.cfg.wav_crop_len * self.cfg.sample_rate):
            pad = self.cfg.wav_crop_len * self.cfg.sample_rate - wav.shape[0]
            wav = np.pad(wav, (0, pad))

        if self.mode == "train":
            if self.aug_audio:
                wav = self.aug_audio(samples=wav, sample_rate=self.cfg.sample_rate)
        else:
            if self.cfg.val_aug:
                wav = self.cfg.val_aug(samples=wav, sample_rate=self.cfg.sample_rate)

        wav_tensor = torch.tensor(wav)  # (n_samples)
        if parts > 1:
            n_samples = wav_tensor.shape[0]
            wav_tensor = wav_tensor[: n_samples // parts * parts].reshape(
                parts, n_samples // parts
            )

        feature_dict = {
            "input": wav_tensor,
            "target": torch.tensor(label.astype(np.float32)),
            "weight": torch.tensor(weight),
            "fold": torch.tensor(fold),
        }
        return feature_dict

    def __len__(self):
        return len(self.fns)

    def load_one(self, id_, offset, duration):
        fp = self.data_folder + id_
        try:
            wav, sr = librosa.load(fp, sr=None, offset=offset, duration=duration)
        except:
            print("FAIL READING rec", fp)

        return wav