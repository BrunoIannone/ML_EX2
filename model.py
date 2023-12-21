from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn as nn

import torch.nn.functional as F
import pytorch_lightning as pl
import time
import torchmetrics
import utils

class CarActionModel(pl.LightningModule): 
    def __init__(self,  number_actions: int,lin_lr: float = 0.0, cnn_lr: float = 0.0,lin_wd: float = 0.0, cnn_wd: float = 0.0, lin_dropout: float = 0.0, cnn_dropout:float = 0.0) -> None:
        """Car action model init function

        Args:
            number_actions (int): Number of actions
            lin_lr (float, optional): Linear layer learning rate. Defaults to 0.0.
            cnn_lr (float, optional): CNN learning rate. Defaults to 0.0.
            lin_wd (float, optional): Linear layer weight decay. Defaults to 0.0.
            cnn_wd (float, optional): CNN weight decay. Defaults to 0.0.
            lin_dropout (float, optional): Linear layer dropout . Defaults to 0.0.
            cnn_dropout (float, optional): CNN dropout. Defaults to 0.0.
        """
        super().__init__()
        self.number_actions = number_actions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Adjust input channels for RGB images
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjust the input size for the fully connected layer to accommodate 96x96 images
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, number_actions)  # Adjust output size for 5 classes

        
        self.lin_lr = lin_lr
        self.lin_wd = lin_wd
        
        self.val_metric  = torchmetrics.F1Score(task="multiclass", num_classes=number_actions, average='micro')
        self.test_metric = torchmetrics.F1Score(task="multiclass", num_classes=number_actions, average='micro')

        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


    def configure_optimizers(self):
        
        groups = [
          {
               "params": self.conv1.parameters(),
               "lr": self.lin_lr,
               "weight_decay": self.lin_wd,
            },
            {
               "params": self.conv2.parameters(),
               "lr": self.lin_lr,
               "weight_decay": self.lin_wd,
            },
           {
               "params": self.fc1.parameters(),
               "lr": self.lin_lr,
               "weight_decay": self.lin_wd,
            },
            {
               "params": self.fc2.parameters(),
               "lr": self.lin_lr,
               "weight_decay": self.lin_wd,
            }
        ]           
        
        optimizer = torch.optim.AdamW(self.parameters())
               
        return optimizer
    
    def training_step(self,train_batch,batch_idx):
        image,labels = train_batch
        outputs = self(image)
        
        loss = F.cross_entropy(outputs.view(-1, self.number_actions),labels,ignore_index=-100)
        
        self.log_dict({'train_loss':loss},on_epoch=True, batch_size=utils.BATCH_SIZE,on_step=False,prog_bar=True)
        
        return loss
        

    def validation_step(self, val_batch,idx):
        image, labels = val_batch
        outputs = self(image)
        y_pred = outputs.argmax(dim = 1)
       
       
        loss = F.cross_entropy(outputs.view(-1, self.number_actions),labels,ignore_index=-100)
        
        self.val_metric(y_pred,labels)
        self.log_dict({'val_loss':loss,'valid_f1': self.val_metric},batch_size=utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
    
        
    def test_step(self, test_batch,idx):
        image, labels = test_batch

        outputs = self(image)
        y_pred = outputs.argmax(dim = 1)
        
        loss = F.cross_entropy(outputs.view(-1, self.number_actions),labels,ignore_index=-100)

        self.test_metric(y_pred,labels)
        self.log_dict({'test_loss':loss,'test_f1': self.test_metric},batch_size=utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
                      