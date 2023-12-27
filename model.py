from typing import List
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import torchmetrics
import utils
from torchvision import transforms
import numpy as np
import datetime

from plot_utils import plot_confusion_matrix
class CarActionModel(pl.LightningModule): 
    def __init__(self,  number_actions: int,action_names: List[str] = None, action_labels:List[int] = None,fc_lr: float = 0.0, cnn_lr: float = 0.0,fc_wd: float = 0.0, cnn_wd: float = 0.0, fc_dropout: float = 0.0, cf_matrix_filename: str = "") -> None:
        """Car action model init function

        Args:
            number_actions (int): Number of actions
            action_names(List[str]): Action names list of string. Optional
            action_labels(List[int]): List of action labels in integer values
            fc_lr (float, optional): Linear layer learning rate. Defaults to 0.0.
            cnn_lr (float, optional): CNN learning rate. Defaults to 0.0.
            fc_wd (float, optional): Linear layer weight decay. Defaults to 0.0.
            cnn_wd (float, optional): CNN weight decay. Defaults to 0.0.
            fc_dropout (float, optional): Linear layer dropout . Defaults to 0.0.
            cnn_dropout (float, optional): CNN dropout. Defaults to 0.0.
        """
        super().__init__()
        self.number_actions = number_actions
        self.action_names = action_names
        self.action_labels = action_labels
        self.cf_matrix_filename = cf_matrix_filename

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(36864, 128)
        self.fc2 = nn.Linear(128, 5)
        #self.fc3 = nn.Linear(250, 50)  # Adjust the output size to 5 for 5 classes
        #self.fc4 = nn.Linear(100, 5)  # Adjust the output size to 5 for 5 classes

        self.fc_dropout = nn.Dropout(fc_dropout)

        
        self.fc_lr = fc_lr
        self.fc_wd = fc_wd
        self.cnn_lr = cnn_lr
        self.cnn_wd = cnn_wd
        
        self.val_f1  = torchmetrics.F1Score(task="multiclass", num_classes=number_actions, average='macro')
        self.val_accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = number_actions)
        
        
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=number_actions, average='macro')
        self.test_accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = number_actions)
        self.y_pred = None
        self.test_labels = None
        if self.training:
            print("Ciao")
            self.y_pred = torch.Tensor().to(utils.DEVICE)
            self.test_labels =torch.Tensor().to(utils.DEVICE)
            self.save_hyperparameters()
        print(self.device)
    def forward(self, x):
        #x = self.bn1(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # x = self.conv3(x)
        # x = self.relu(x)
        # x = self.pool3(x)

        # Flatten the output
        x = self.flatten(x)

        # Fully connected layers for classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_dropout(x)
        
        x = self.fc2(x)
        #x = self.relu(x)
        #x = self.fc_dropout(x)

        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.fc_dropout(x)

        #x = self.fc4(x)


        return x

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters())
               
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
        
        self.val_f1(y_pred,labels)
        self.val_accuracy(y_pred,labels)
        
        self.log_dict({'val_loss':loss,'valid_f1': self.val_f1, 'valid_acc':self.val_accuracy },batch_size=utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
    
        
    def test_step(self, test_batch,idx):
        image, labels = test_batch
        outputs = self(image)
        y_pred = outputs.argmax(dim = 1)
        self.y_pred=torch.cat((self.y_pred,y_pred),dim=0)
        self.test_labels= torch.cat((self.test_labels,labels),dim=0)
        loss = F.cross_entropy(outputs.view(-1, self.number_actions),labels,ignore_index=-100)
        self.test_f1(y_pred,labels)
        self.test_accuracy(y_pred,labels)
        self.log_dict({'test_loss':loss,'test_f1': self.test_f1, 'test_acc':self.test_accuracy},batch_size=utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
    
    def on_test_end(self) -> None:
        
        plot_confusion_matrix(self.test_labels.cpu().numpy(),self.y_pred.cpu().numpy(),"Car action",0,str(utils.ROOT_FOOLDER)+"/Saves/conf_mat/",False,True,self.action_names,self.action_labels,cf_matrix_filename=self.cf_matrix_filename)
    
    def predict(self,to_predict):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])
        to_predict = transform(to_predict).unsqueeze(0).cpu()
        
        p = self(to_predict)
        
        _,action = torch.max(p,1)
        return int(action)
        