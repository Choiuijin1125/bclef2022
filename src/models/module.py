import torchmetrics
import torch
import numpy as np
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule
from .. import train_utils


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