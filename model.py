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
    def __init__(self,  number_actions: int,action_names: List[str] = None, action_labels:List[int] = None,fc_lr: float = 0.0, cnn_lr: float = 0.0,fc_wd: float = 0.0, cnn_wd: float = 0.0, fc_dropout: float = 0.0, cnn_dropout:float = 0.0) -> None:
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

        
        self.fc_lr = fc_lr
        self.fc_wd = fc_wd
        self.cnn_lr = cnn_lr
        self.cnn_wd = cnn_wd
        
        self.val_metric  = torchmetrics.F1Score(task="multiclass", num_classes=number_actions, average='micro')
        self.test_metric = torchmetrics.F1Score(task="multiclass", num_classes=number_actions, average='micro')
        self.y_pred = torch.Tensor().cuda()
        self.test_labels =torch.Tensor().cuda()
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
               "lr": self.cnn_lr,
               "weight_decay": self.cnn_wd,
            },
            {
               "params": self.conv2.parameters(),
               "lr": self.cnn_lr,
               "weight_decay": self.cnn_wd,
            },
           {
               "params": self.fc1.parameters(),
               "lr": self.fc_lr,
               "weight_decay": self.fc_wd,
            },
            {
               "params": self.fc2.parameters(),
               "lr": self.fc_lr,
               "weight_decay": self.fc_wd,
            }
        ]           
        
        optimizer = torch.optim.AdamW(groups)
               
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
        self.y_pred=torch.cat((self.y_pred,y_pred),dim=0)
        self.test_labels= torch.cat((self.test_labels,labels),dim=0)
        loss = F.cross_entropy(outputs.view(-1, self.number_actions),labels,ignore_index=-100)
        self.test_metric(y_pred,labels)
        self.log_dict({'test_loss':loss,'test_f1': self.test_metric},batch_size=utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
    def on_test_end(self) -> None:
        # print(self.number_actions,type(self.number_actions))
        # conf_mat = torchmetrics.ConfusionMatrix(task = "multiclass",num_classes =self.number_actions).to(utils.DEVICE)
        # # Set the figure size in inches for 720p resolution
        # fig, ax = plt.subplots(figsize=(1280/100, 720/100), dpi=100)
        # conf_mat(torch.Tensor(self.y_pred).to(utils.DEVICE),torch.Tensor(self.test_labels).to(utils.DEVICE))
        # conf_mat.plot(ax=ax,labels = self.action_names)
        # ax.set(
        #     title='Confusion Matrix ',
        #     xlabel='Predicted',
        #     ylabel='Actual '
        # )
        # plt.xticks(rotation=45)
        # plt.grid(False)
        # plt.savefig(str(utils.ROOT_FOOLDER)+"/Saves/conf_mat/"+str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"))+".png",dpi = 100)
        #print(self.action_labels)
        plot_confusion_matrix(self.test_labels.cpu().numpy(),self.y_pred.cpu().numpy(),"None",0,str(utils.ROOT_FOOLDER)+"/Saves/conf_mat/",False,True,self.action_names,self.action_labels)
    def predict(self,to_predict):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])
        to_predict = transform(to_predict).unsqueeze(0).cuda()
        
        p = self(to_predict)
        #print(np.argmax(p.detach()))

        #action = int(np.argmax(p.detach()))  # adapt to your model
        _,action = torch.max(p,1)
        return int(action)
        #print("ACTION", action)