# bclef2022

## Main Idea

There are two track I've been trying

1. Reproduction of [kaggle-birdclef2021-2nd-place](https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/tree/main/configs)
2. TimmSED with scored birds selective mixup
- While training I made two dataloader with same batch `train_dataloader`, `scored_brids_mixup_dataloader`.
- `scored_brids_mixup_dataloader` is the dataloader which only has scored_birds audio from train set
- In every batch do mixup with train_set + scored_birds_set(kind of overfitting)
3. give penalty 
- with selective mixup, model can be overfitted with scored birds, and especially high frequency scored birds(`skyalr`, `houfin`, etc..), so trained model show very high confidence score in high frequency scored birds
- so I have a penalty to that class depending on distribution of class `train.primary_label.value_counts()[scored_birds_list]` It seems like 
```
scored_value = train_df.primary_label.value_counts()[conf.model.scored_columns]
scored_factor = scored_value / sum(scored_value)

for i, scored_column in enumerate(conf.model.scored_columns):
    df_preds[scored_column] = df_preds[scored_column] - scored_factor.values[i] * 0.4
```
4. quantile threshold
- to make ensemble flexable, I used quantile threshold 0.85(kaggle-birdclef2021)

## Basic Consensus

### Datasets
- 30sec random crop(birdclef2021-2nd-place)
- Use the audio rating for weighting the recordings contribution to the loss(birdclef2021-2nd-place)
- Label smoothing +0.01 to everything (birdclef2021-2nd-place)

  ex [0.1, 1, 0.1, 0,1,...]

### Augmentation
- only use backgroundnoise(ff1010bird_nocall, train_soundscapes_2021, aicrowd2020_noise_30sec)

### Loss
- BCEFocal2WayLoss
- BCEFocalLoss(timm_007, timm_008)

### Scheduler
- torch.optim.lr_scheduler.CosineAnnealingLR

### Training
- F32 training(I don't know why, if I use mixed precision training with pytorch ligthing loss will be explored)

### Model
1. TimmSED
- Baseline model
2. TimmSEDPartGPU
- TimmSED with 6x 5sec parts training(Idea from birdclef2021-2nd-place)
3. [Conformer](https://github.com/m-koichi/ConformerSED) : didn't work
4. [FDY](https://github.com/frednam93/FDY-SED) : didn't work
5. [HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer) : didn't work

## Main Idea
1. Public : 0.79
- TimmSEDGPUpart with scored birds selective mixup
- backbond : tf_efficientnet_b0_ns
- 5 folds ensemble
- give penalty 
- threshold : np.quantile(scored_birds.flatten(), 0.85)
- Inference Time : 3hours


3. Public : 0.75
- Reproduction of birdclef2021-2nd-place(binary classification + cfg_ps_12_v21.py 4folds)
- I train this model without validation(4fold ensemble)
- Inference Time : 3hours

## Next try
- other backbone with best score model
- train with 2021 datasets + selective mixup
- selective mixup with birdclef2021-2nd-place mode(public 0.75)
