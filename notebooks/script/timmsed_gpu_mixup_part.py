#!/usr/bin/env python
# coding: utf-8

import os
import sys
import ast
import glob
import json
import torch
import torchmetrics
import pandas as pd
import numpy as np
from tqdm import tqdm

from omegaconf import OmegaConf
from hydra.utils import instantiate

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

sys.path.insert(0, "../")

from src.datasets import WaveformDatasetRatingV2
from src.models.timmsed import SelectivePartTimmSEDGPU

from src.models.module import SelectiveTrainModule
# import wandb
# wandb.init(settings=wandb.Settings(start_method='fork'))

# all_path = glob.glob("../datasets/train_np/*/*/*.npy")
train = pd.read_csv("../datasets/train_folds_all.csv")
train["new_target"] = (
    train["primary_label"]
    + " "
    + train["secondary_labels"].map(lambda x: " ".join(ast.literal_eval(x)))
)
train["len_new_target"] = train["new_target"].map(lambda x: len(x.split()))

train["file_path"] = "../datasets/2022/train_audio/" + train["filename"]

scored_birds = "../datasets/2022/scored_birds.json"
with open(scored_birds, "r") as f:
    scored_birds_list = json.load(f)
    
train.loc[~train["primary_label"].isin(scored_birds_list), "kfold"] = 6

label_dict = dict()
labels = train.primary_label.unique().tolist()
# labels.append("rocpig1")
for i, label in enumerate(labels):
    label_dict[label] = i
    
train["scored"] = False

for i, row in tqdm(train.iterrows(), total=len(train)):
    for label in row["new_target"].split():
        if label in scored_birds_list:
            train.loc[i, "scored"] = True
    
# tmp_df = train.loc[train["kfold"] != 0].drop_duplicates("filename")
# train = pd.concat([tmp_df, train.loc[train["kfold"] == 0]]).reset_index(drop=True)

def main():
    cfg = "../src/configs/timmsed/timmSED_exp_009.yaml"
    cfg = OmegaConf.load(cfg)
    cfg.model.target_columns = train.primary_label.unique().tolist()
    cfg.model.target_columns.append("rocpig1")
    cfg.model.num_classes = len(cfg.model.target_columns)
    seed_everything(cfg.exp_manager.seed)
    fold = 0
    trn_df = train[train.kfold != fold].reset_index(drop=True)
    val_df = train[train.kfold == fold].reset_index(drop=True)
    mixup = trn_df.loc[trn_df["scored"] == True].reset_index(drop=True)

    train_dataset = WaveformDatasetRatingV2(trn_df, cfg.model, mode="train")
    mixup_dataset = WaveformDatasetRatingV2(mixup, cfg.model, mode="train")
    
#     train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=cfg.model.train_ds.batch_size,
#         num_workers=cfg.model.valid_ds.num_workers,
#         pin_memory=True,
#         shuffle=True,
#         persistent_workers=True,
#     )
    valid_dataset = WaveformDatasetRatingV2(val_df, cfg.model, mode="valid")
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.model.valid_ds.batch_size,
        num_workers=cfg.model.valid_ds.num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
    )

    sed_model = SelectivePartTimmSEDGPU(cfg.model)
    train_module = SelectiveTrainModule(sed_model, cfg.model, train_dataset, mixup_dataset)

    cfg.model.sched.T_max = round(len(train_dataset) // cfg.model.train_ds.batch_size * cfg.trainer.max_epochs // cfg.trainer.devices)
    
    #cfg.model.sched.num_warmup_steps = round(len(train_dataset) // cfg.model.train_ds.batch_size * cfg.trainer.max_epochs // cfg.trainer.devices)

    #wandb_logger = instantiate(cfg.exp_manager.logger)
    lr_callback = instantiate(cfg.exp_manager.callbacks.lr_callback)
    checkpoint_callback = instantiate(cfg.exp_manager.callbacks.checkpoint_callback)

    #trainer = Trainer(**cfg.trainer, logger=wandb_logger, callbacks=[lr_callback, checkpoint_callback])
    trainer = Trainer(**cfg.trainer, callbacks=[lr_callback, checkpoint_callback])

    trainer.fit(train_module, val_dataloaders=valid_dataloader)

if __name__ == "__main__":
    main()