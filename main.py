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



hyp_comb = list(product(utils.LEARNING_RATE, utils.transformer_learning_rate, [0], utils.LIN_DROPOUT,utils.weight_decay,utils.transformer_weight_decay))
#train_samples = utils.build_couples(utils.TEST_DIR_PATH)
import pytorch_lightning as pl

data_processor = DataProcessor(utils.ROOT_FOOLDER/"train.csv",utils.ROOT_FOOLDER/"test.csv",0.3,0)
for hyperparameter in tqdm.tqdm(hyp_comb,colour="yellow", desc="Tried combinations"):
    
    lin_lr = hyperparameter[0]
    cnn_lr = hyperparameter[1]
    cnn_dropout = hyperparameter[2]
    lin_dropout = hyperparameter[3]
    lin_wd = hyperparameter[4]
    cnn_wd = hyperparameter[5]
    
    print(colored(str(("LIN_LR:", lin_lr,"CNN_LR:", cnn_lr,"DROPOUT_EMBED:", cnn_dropout, "LINEAR_DROPOUT:", lin_dropout, "LINEAR_WD:", lin_wd, "CNN_WD: ", cnn_wd)), "yellow"))
    
    
    
    print(colored("Built coarse data","green"))

    logger = TensorBoardLogger(str(utils.LOG_SAVE_DIR_NAME) + str(lin_lr) + ", " + str(cnn_lr) + ", " + str(cnn_dropout)+ ", " + str(lin_dropout)+ ", " + str(lin_wd) + ", " + str(cnn_wd))
    trainer = pl.Trainer(log_every_n_steps=50,max_epochs = utils.NUM_EPOCHS,callbacks=[EarlyStopping(monitor="val_loss", patience=5,mode='min'), ModelCheckpoint(filename= str(lin_lr) + ", " + str(cnn_wd) + ", " + str(cnn_dropout)+ ", " + str(lin_dropout)+ ", " + str(lin_wd) + ", " + str(cnn_wd),monitor='valid_f1',save_top_k=1,every_n_epochs=1,mode='max',save_weights_only=False,verbose=True,dirpath=utils.CKPT_SAVE_DIR_NAME)],logger=logger,accelerator='cpu')
    #print(list(zip(data_processor.x_train,data_processor.y_train)))
    print(colored("Built logger and trainer","green"))
    car_action_datamodule = CarActionDataModule(list(zip(data_processor.x_train,data_processor.y_train)),list(zip(data_processor.x_eval,data_processor.y_eval)),list(zip(data_processor.test_samples,data_processor.test_labels)))
    car_action_model = CarActionModel(len(data_processor.labels_name),lin_lr,cnn_lr,lin_wd,cnn_wd,lin_dropout,cnn_dropout)
    print(colored("Starting transformer coarse training...","green"))
    trainer.fit(car_action_model,datamodule = car_action_datamodule)

    print(colored("Starting transformer coarse testing...","green"))
    car_action_model.eval()
    trainer.test(car_action_model,datamodule = car_action_datamodule)#,ckpt_path="best")

    print(colored("Pipeline over","red"))