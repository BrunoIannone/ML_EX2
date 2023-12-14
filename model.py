from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn as nn

import torch.nn.functional as F
import pytorch_lightning as pl
import time
import torchmetrics
import utils

class WSD(pl.LightningModule): 
    def __init__(self, language_model_name: str, num_labels: int,fine_tune_lm: bool = True,lin_lr: float = 0.0, backbone_lr: float = 0.0,lin_wd: float = 0.0, backbone_wd: float = 0.0, lin_dropout: float = 0.0 ,*args, **kwargs) -> None:
        """WSD init function for transformer-based models

    Args:
        language_model_name (str): Transformer Hugging Face model name
        num_labels (int): Number of total labels 
        fine_tune_lm (bool, optional): If false, transformer parameters won't be updated. Defaults to True.
        lin_lr (float, optional): Linear layer learning rate. Defaults to 0.0.
        backbone_lr (float, optional): Transformer learning rate. Defaults to 0.0.
        lin_wd (float, optional): Linear layer weight decay. Defaults to 0.0.
        backbone_wd (float, optional): Transformer weight decay. Defaults to 0.0.
        lin_dropout (float, optional): Linear layer dropout. Defaults to 0.0.
    
    """
        super().__init__()
        self.num_labels = num_labels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for digits 0-9

        
        if not fine_tune_lm:
            for param in self.parameters():
                param.requires_grad = False
        
        self.val_metric  = torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='micro')
        self.test_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='micro')

        self.save_hyperparameters()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten for fully connected layer
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


    def configure_optimizers(self):
        
        # groups = [
        #   {
        #        "params": self.classifier.parameters(),
        #        "lr": self.lin_lr,
        #        "weight_decay": self.lin_wd,
        #     },
        #    {
        #        "params": self.backbone.parameters(),
        #        "lr": self.backbone_lr,
        #        "weight_decay": self.backbone_wd,
        #    } 
        # ]           
        
        optimizer = torch.optim.AdamW(self.parameters())
               
        return optimizer
    
    def training_step(self,train_batch,batch_idx):
        image,labels = train_batch
        outputs = self(image)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),labels,ignore_index=-100)
        
        self.log_dict({'train_loss':loss},on_epoch=True, batch_size=utils.BATCH_SIZE,on_step=False,prog_bar=True)
        
        return loss
        

    def validation_step(self, val_batch,idx):
        image, labels = val_batch
        outputs = self(image)
        y_pred = outputs.argmax(dim = 1)
       
       
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),labels,ignore_index=-100)
        
        self.val_metric(y_pred,val_batch["labels"])
        self.log_dict({'val_loss':loss,'valid_f1': self.val_metric},batch_size=utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
    
        
    def test_step(self, test_batch,idx):
        
        outputs = self(**test_batch)
        y_pred = outputs.argmax(dim = 1)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),test_batch["labels"].view(-1),ignore_index=-100)

        self.test_metric(y_pred,test_batch["labels"])
        self.log_dict({'test_loss':loss,'test_f1': self.test_metric},batch_size=utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
                      