import utils
import model
#res = utils._build_couples(utils.TRAINING_DIR_PATH)
#print(len(res))
from itertools import product
from data_processor import DataProcessor
from car_action_datamodule import CarActionDataModule
import tqdm
from model import CarActionModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from termcolor import colored
import os
import subprocess




hyp_comb = list(product(utils.FC_LR, utils.FC_WD, utils.FC_DROPOUT, utils.CNN_LR,utils.CNN_WD,utils.CNN_DROPOUT))
import pytorch_lightning as pl

data_processor = DataProcessor(utils.ROOT_FOOLDER/"train.csv",utils.ROOT_FOOLDER/"test.csv",0.3,0)
for hyperparameter in tqdm.tqdm(hyp_comb,colour="yellow", desc="Tried combinations"):
    
    fc_lr = hyperparameter[0]
    fc_wd = hyperparameter[1]
    fc_dropout = hyperparameter[2]

    cnn_lr = hyperparameter[3]
    cnn_wd = hyperparameter[4]
    cnn_dropout = hyperparameter[5]
    
    print(colored(str(("FC_LR:", fc_lr,"FC_WD:", fc_wd,"FC_DROPOUT:", fc_dropout,"CNN_LR:", cnn_lr,  "CNN_WD: ", cnn_wd,"CNN_DROPOUT:", cnn_dropout)), "yellow"))
    
    print(colored("Built data","green"))

    logger = TensorBoardLogger(save_dir=str(utils.LOG_SAVE_DIR_NAME),name= str(fc_lr) + ", " + str(fc_wd) + ", " + str(fc_dropout)+ ", " + str(cnn_lr)+ ", " + str(cnn_wd) + ", " + str(cnn_dropout))
    trainer = pl.Trainer(log_every_n_steps=5,max_epochs = utils.NUM_EPOCHS,callbacks=[EarlyStopping(monitor="val_loss", patience=5,mode='min'), ModelCheckpoint(filename= str(fc_lr) + ", " + str(fc_wd) + ", " + str(fc_dropout)+ ", " + str(cnn_lr)+ ", " + str(cnn_wd) + ", " + str(cnn_dropout),monitor='valid_f1',save_top_k=1,every_n_epochs=1,mode='max',save_weights_only=False,verbose=True,dirpath=utils.CKPT_SAVE_DIR_NAME)],logger=logger)#,accelerator='cuda')
    #print(list(zip(data_processor.x_train,data_processor.y_train)))
    print(colored("Built logger and trainer","green"))
    car_action_datamodule = CarActionDataModule(list(zip(data_processor.x_train,data_processor.y_train)),list(zip(data_processor.x_eval,data_processor.y_eval)),list(zip(data_processor.test_samples,data_processor.test_labels)))
    #print(colored(len(data_processor.labels_name),"red"))
    labels_name = ["Nothing", "Left", "Right", "Gas", "Brake"]
    car_action_model = CarActionModel(len(data_processor.labels_name),labels_name, [int(label) for label in data_processor.labels_name],fc_lr,cnn_lr,fc_wd,cnn_wd,fc_dropout,cnn_dropout)
    print(colored("Starting training...","green"))
    trainer.fit(car_action_model,datamodule = car_action_datamodule)

    print(colored("Starting testing...","green"))
    car_action_model.eval()
    trainer.test(car_action_model,datamodule = car_action_datamodule)#,ckpt_path="best")
    
    original_path = str(utils.CKPT_SAVE_DIR_NAME/str(str(fc_lr) + ", " + str(fc_wd) + ", " + str(fc_dropout)+ ", " + str(cnn_lr)+ ", " + str(cnn_wd) + ", " + str(cnn_dropout)))

    utils.save_last_ckpt_path(original_path)
    subprocess.run(['python', "play_policy_template.py"])
    print(colored("Pipeline over","red"))
