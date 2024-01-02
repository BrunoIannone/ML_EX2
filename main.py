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
import threading
import torch
from pytorch_lightning.profilers import PyTorchProfiler
#import tracemalloc 
#tracemalloc.start() 
import pytorch_lightning as pl
from pympler.tracker import SummaryTracker
from pympler import muppy
import objgraph
biggest_vars = muppy.sort(muppy.get_objects())[-3:]
objgraph.show_backrefs(biggest_vars, filename='backref.png')


hyp_comb = list(product(utils.FC_LR, utils.FC_WD, utils.FC_DROPOUT, utils.CNN_LR,utils.CNN_WD,utils.NUM_EPOCHS,
                        utils.conv1_out_dim,utils.conv1_kernel_dim,utils.conv1_stride_dim,
                        utils.POOL1_KERNEL_DIM,utils.POOL1_STRIDE_DIM,
                        utils.conv2_out_dim,utils.conv2_kernel_dim,utils.conv2_stride_dim,
                        utils.POOL2_KERNEL_DIM,utils.POOL2_STRIDE_DIM,
                        utils.conv3_out_dim,utils.conv3_kernel_dim,utils.conv3_stride_dim,
                        utils.POOL3_KERNEL_DIM,utils.POOL3_STRIDE_DIM))

data_processor = DataProcessor(utils.ROOT_FOOLDER/"train_crop.csv",utils.ROOT_FOOLDER/"test_crop.csv",0.3,0)

car_action_datamodule = CarActionDataModule(list(zip(data_processor.x_train,data_processor.y_train)),list(zip(data_processor.x_eval,data_processor.y_eval)),list(zip(data_processor.test_samples,data_processor.test_labels)))
#snapshot = tracemalloc.take_snapshot() 
#top_stats = snapshot.statistics('lineno') 
  
#for stat in top_stats[:10]: 
   #print(stat)
for hyperparameter in tqdm.tqdm(hyp_comb,colour="yellow", desc="Tried combinations"):
    
    fc_lr = hyperparameter[0]
    fc_wd = hyperparameter[1]
    fc_dropout = hyperparameter[2]

    cnn_lr = hyperparameter[3]
    cnn_wd = hyperparameter[4]
    num_epochs = hyperparameter[5]
    
    conv1_out_dim = hyperparameter[6]
    conv1_kernel_dim = hyperparameter[7]
    conv1_stride_dim = hyperparameter[8]

    pool1_kernel_dim = hyperparameter[9]
    pool1_stride_dim = hyperparameter[10]

    conv2_out_dim = hyperparameter[11]
    conv2_kernel_dim = hyperparameter[12]
    conv2_stride_dim = hyperparameter[13]

    pool2_kernel_dim = hyperparameter[14]
    pool2_stride_dim = hyperparameter[15]

    conv3_out_dim = hyperparameter[16]
    conv3_kernel_dim = hyperparameter[17]
    conv3_stride_dim = hyperparameter[18]

    pool3_kernel_dim = hyperparameter[19]
    pool3_stride_dim = hyperparameter[20]


    filename = str(conv1_out_dim) + ", " + str(conv1_kernel_dim) + ", "+ str(conv1_stride_dim) + ", "+ str(pool1_kernel_dim)+ ", "+ str(pool1_stride_dim) + ", " + str(conv2_out_dim) + ", " + str(conv2_kernel_dim) + ", "+ str(conv2_stride_dim) + ", "+ str(pool2_kernel_dim)+ ", "+ str(pool2_stride_dim) + ", "+str(conv3_out_dim) + ", " + str(conv3_kernel_dim) + ", "+ str(conv3_stride_dim) + ", "+ str(pool3_kernel_dim)+ ", "+ str(pool3_stride_dim) + ", " + str(fc_dropout)
    print(filename)
    if(filename+".ckpt" in os.listdir(utils.CKPT_SAVE_DIR_NAME)):
        print(colored("SKIPPO","red"))
        continue
    
    #print(colored(str(("FC_LR:", fc_lr,"FC_WD:", fc_wd,"FC_DROPOUT:", fc_dropout,"CNN_LR:", cnn_lr,  "CNN_WD: ", cnn_wd)), "yellow"))
    
    print(colored("Built data","green"))

    logger = TensorBoardLogger(save_dir=str(utils.LOG_SAVE_DIR_NAME),name= filename)
    #profiler = PyTorchProfiler(on_trace_ready = torch.profiler.tensorboard_trace_handler("/home/bruno/Desktop/ML_EX2/Saves/tb_logs/profiler0"),trace_memory = True)

    trainer = pl.Trainer(log_every_n_steps=140,max_epochs = num_epochs,callbacks=[EarlyStopping(monitor="val_loss", patience=3,mode='min'), ModelCheckpoint(filename= filename,monitor='val_loss',save_top_k=1,every_n_epochs=1,mode='min',save_weights_only=False,verbose=True,dirpath=utils.CKPT_SAVE_DIR_NAME)],logger=logger)#,accelerator='cuda')
    #print(list(zip(data_processor.x_train,data_processor.y_train)))
    print(colored("Built logger and trainer","green"))
    #print(colored(len(data_processor.labels_name),"red"))
    labels_name = ["Nothing", "Left", "Right", "Gas", "Brake"]
    car_action_model = CarActionModel(len(data_processor.labels_name),labels_name, [int(label) for label in data_processor.labels_name],fc_lr,cnn_lr,fc_wd,cnn_wd,fc_dropout,cf_matrix_filename= filename,
                                      conv1_out_dim=conv1_out_dim,conv1_kernel_dim=conv1_kernel_dim, conv1_stride_dim=conv1_stride_dim,
                                      pool1_kernel_dim=pool1_kernel_dim, pool1_stride_dim=pool1_stride_dim,
                                      
                                      conv2_out_dim=conv2_out_dim,conv2_kernel_dim=conv2_kernel_dim, conv2_stride_dim=conv2_stride_dim,
                                      pool2_kernel_dim=pool2_kernel_dim, pool2_stride_dim=pool2_stride_dim,
                                      
                                      conv3_out_dim=conv3_out_dim,conv3_kernel_dim=conv3_kernel_dim, conv3_stride_dim=conv3_stride_dim,
                                      pool3_kernel_dim=pool3_kernel_dim, pool3_stride_dim=pool3_stride_dim)
    
    print(colored("Starting training...","green"))
    try:
        tracker = SummaryTracker()

        trainer.fit(car_action_model,datamodule = car_action_datamodule)
        tracker.print_diff()

        #snapshot = tracemalloc.take_snapshot() 
        #top_stats = snapshot.statistics('lineno') 
  
        #for stat in top_stats[:10]: 
            #print(stat)
    except torch.cuda.OutOfMemoryError:
        print(colored("Out of memory detected, skipping","red"))
        continue
    except MemoryError:
        print(colored("Out of CPU memory detected, skipping", "red"))
        # Handle CPU out-of-memory error here
        # You might want to reduce the batch size or take other actions to free up CPU memory
        continue
    print(colored("Starting testing...","green"))
    #car_action_model.eval()
    #trainer.test(car_action_model,datamodule = car_action_datamodule)#,ckpt_path="best")
    
    #original_path = str(utils.CKPT_SAVE_DIR_NAME/str(filename))

    #utils.save_last_ckpt_path(original_path)
    #command_thread = threading.Thread(target=subprocess.Popen(['python', "play_policy_template.py"]))
    
    
subprocess.run(['bash',"alert.sh"])
print(colored("Pipeline over","red"))
# current, peak = tracemalloc.get_traced_memory()
# print(f"Current memory usage: {current / 10**6} MB")
# print(f"Peak memory usage: {peak / 10**6} MB")
#snapshot = tracemalloc.take_snapshot() 
#top_stats = snapshot.statistics('lineno') 
  
#for stat in top_stats[:10]: 
   #print(stat)
#tracemalloc.stop() 
#objgraph.show_most_common_types()
