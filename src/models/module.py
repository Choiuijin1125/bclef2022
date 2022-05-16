import torchmetrics
import torch
import numpy as np
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule
from .. import train_utils
from torch.cuda.amp import GradScaler, autocast


class TrainModule(LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.train_f1_score_03 = torchmetrics.F1(
            num_classes=self.cfg.num_classes,
            threshold=0.3,
            average="micro",
        )
        self.train_f1_score_05 = torchmetrics.F1(
            num_classes=self.cfg.num_classes,
            threshold=0.5,
            average="micro",
        )
        self.valid_f1_score_03 = torchmetrics.F1(
            num_classes=self.cfg.num_classes,
            threshold=0.3,
            average="micro",
        )
        self.valid_f1_score_03_classwise = torchmetrics.F1(
            num_classes=self.cfg.num_classes,
            threshold=0.3,
            average=None,
        )        
        self.valid_f1_score_05 = torchmetrics.F1(
            num_classes=self.cfg.num_classes,
            threshold=0.5,
            average="micro",
        )
        
    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["targets"]
        weight = batch["weight"]
        
        if self.cfg.train_mode == "base":
            outputs = self.model(x,y, weight)
            loss = train_utils.loss_fn(outputs, self.cfg)
        
        elif self.cfg.train_mode == "base_conformer":
            x_len = torch.LongTensor([938] * x.shape[0])
            outputs = self.model(x, y, weight, x_len)
            loss = train_utils.loss_fn(outputs, self.cfg)            
        
        else:
            assert "Not supported train mode" 
            
        self.log("train_loss", loss)
        self.log(
            "train_f1_score_03",
            self.train_f1_score_03(outputs["clipwise_output"], outputs["targets"].int()),
        )
        self.log(
            "train_f1_score_05",
            self.train_f1_score_05(outputs["clipwise_output"], outputs["targets"].int()),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["targets"]
        weight = batch["weight"]
        
        if self.cfg.train_mode == "base":
            outputs = self.model(x,y, weight)
        elif self.cfg.train_mode == "base_conformer":
            x_len = torch.LongTensor([938] * x.shape[0])
            outputs = self.model(x,y, weight, x_len)
                
            
        loss = train_utils.loss_fn(outputs, self.cfg)
        f1_score_03 = self.valid_f1_score_03(outputs["clipwise_output"], outputs["targets"].int())
        f1_score_05 = self.valid_f1_score_05(outputs["clipwise_output"], outputs["targets"].int())

        self.log("valid_loss", loss)
        self.log("valid_f1_score_03", f1_score_03)
        self.log("valid_f1_score_05", f1_score_05)

        return {
            "valid_loss": loss,
            "valid_f1_score_03": f1_score_03,
            "valid_f1_score_05": f1_score_05,
        }

    def validation_epoch_end(self, outs):
        all_losses = [out["valid_loss"] for out in outs]
        all_f1_score_03 = [out["valid_f1_score_03"] for out in outs]
        all_f1_score_05 = [out["valid_f1_score_05"] for out in outs]

        all_losses = torch.stack(all_losses).mean()
        all_f1_score_03 = torch.stack(all_f1_score_03).mean()
        all_f1_score_05 = torch.stack(all_f1_score_05).mean()
        
        self.log("valid_loss", all_losses)
        self.log("valid_f1_score_03", all_f1_score_03)
        self.log("valid_f1_score_05", all_f1_score_05)
        
        return {
            "valid_loss": all_losses,
            "valid_f1_score_03": all_f1_score_03,
            "valid_f1_score_05": all_f1_score_05,
        }        

    def configure_optimizers(self):
        opt = instantiate(self.cfg.optim, params=self.parameters())
        scheduler = instantiate(self.cfg.sched, optimizer=opt)
        return [opt], [{'scheduler': scheduler, 'interval': 'step'}]
    
    
    
    
class SelectiveTrainModule(LightningModule):
    def __init__(self, model, cfg, train_datasets, mixup_datasets):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.train_f1_score_03 = torchmetrics.F1(
            num_classes=self.cfg.num_classes,
            threshold=0.3,
            average="micro",
        )
        self.train_f1_score_05 = torchmetrics.F1(
            num_classes=self.cfg.num_classes,
            threshold=0.5,
            average="micro",
        )
        self.valid_f1_score_03 = torchmetrics.F1(
            num_classes=self.cfg.num_classes,
            threshold=0.3,
            average="micro",
        )
        self.valid_f1_score_03_classwise = torchmetrics.F1(
            num_classes=self.cfg.num_classes,
            threshold=0.3,
            average=None,
        )        
        self.valid_f1_score_05 = torchmetrics.F1(
            num_classes=self.cfg.num_classes,
            threshold=0.5,
            average="micro",
        )
        
        
        self.train_datasets = train_datasets
        self.mixup_datasets = mixup_datasets
        
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_datasets,
            batch_size=self.cfg.train_ds.batch_size,
            num_workers=self.cfg.train_ds.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True)
        
        mixup_loader = torch.utils.data.DataLoader(
            self.mixup_datasets,
            batch_size=self.cfg.train_ds.batch_size,
            num_workers=self.cfg.train_ds.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True)
        
        return {"train": train_loader, "mixup": mixup_loader}

    
#     def val_dataloader(self):
#         valid_loader = torch.utils.data.DataLoader(
#             self.valid_datasets,
#             batch_size=self.cfg.valid_ds.batch_size,
#             num_workers=self.cfg.valid_ds.num_workers,
#             pin_memory=True,
#             shuffle=True,)
        
#         mixup_loader = torch.utils.data.DataLoader(
#             self.mixup_datasets,
#             batch_size=self.cfg.valid_ds.batch_size,
#             num_workers=self.cfg.valid_ds.num_workers,
#             pin_memory=True,
#             shuffle=True,)
        
#         return [loader]
    
        
    def training_step(self, batch, batch_idx):
        x = batch["train"]["image"]
        y = batch["train"]["targets"]
        weight = batch["train"]["weight"]
        
        
        if self.cfg.train_mode == "base":
            outputs = self.model(x,y, weight)
            loss = train_utils.loss_fn(outputs, self.cfg)
        
        elif self.cfg.train_mode == "base_conformer":
            x_len = torch.LongTensor([938] * x.shape[0])
            outputs = self.model(x, y, weight, x_len)
            loss = train_utils.loss_fn(outputs, self.cfg)
            
        elif self.cfg.train_mode == "selective_mixup":
            selective_x = batch["mixup"]["image"]
            selective_y = batch["mixup"]["targets"]
            selective_weight = batch["mixup"]["weight"]
            
            x = torch.concat([x, selective_x])
            y = torch.concat([y, selective_y])
            weight = torch.concat([weight, selective_weight])
            
            outputs = self.model(x,y, weight)
            
            loss = train_utils.loss_fn(outputs, self.cfg)            
        
        else:
            assert "Not supported train mode" 
            
        self.log("train_loss", loss)
        self.log(
            "train_f1_score_03",
            self.train_f1_score_03(outputs["clipwise_output"], outputs["targets"].int()),
        )
        self.log(
            "train_f1_score_05",
            self.train_f1_score_05(outputs["clipwise_output"], outputs["targets"].int()),
        )
        return loss

    def validation_step(self, batch, batch_idx):

        x = batch["image"]
        y = batch["targets"]
        weight = batch["weight"]
        
        if (self.cfg.train_mode == "base") | (self.cfg.train_mode == "selective_mixup"):
            outputs = self.model(x,y, weight)
        elif self.cfg.train_mode == "base_conformer":
            x_len = torch.LongTensor([938] * x.shape[0])
            outputs = self.model(x,y, weight, x_len)
            
            
        loss = train_utils.loss_fn(outputs, self.cfg)
        f1_score_03 = self.valid_f1_score_03(outputs["clipwise_output"], outputs["targets"].int())
        f1_score_05 = self.valid_f1_score_05(outputs["clipwise_output"], outputs["targets"].int())

        self.log("valid_loss", loss)
        self.log("valid_f1_score_03", f1_score_03)
        self.log("valid_f1_score_05", f1_score_05)

        return {
            "valid_loss": loss,
            "valid_f1_score_03": f1_score_03,
            "valid_f1_score_05": f1_score_05,
        }

    def validation_epoch_end(self, outs):
        all_losses = [out["valid_loss"] for out in outs]
        all_f1_score_03 = [out["valid_f1_score_03"] for out in outs]
        all_f1_score_05 = [out["valid_f1_score_05"] for out in outs]

        all_losses = torch.stack(all_losses).mean()
        all_f1_score_03 = torch.stack(all_f1_score_03).mean()
        all_f1_score_05 = torch.stack(all_f1_score_05).mean()
        
        self.log("valid_loss", all_losses)
        self.log("valid_f1_score_03", all_f1_score_03)
        self.log("valid_f1_score_05", all_f1_score_05)
        
        return {
            "valid_loss": all_losses,
            "valid_f1_score_03": all_f1_score_03,
            "valid_f1_score_05": all_f1_score_05,
        }        

    def configure_optimizers(self):
        opt = instantiate(self.cfg.optim, params=self.parameters())
        scheduler = instantiate(self.cfg.sched, optimizer=opt)
        return [opt], [{'scheduler': scheduler, 'interval': 'step'}]    