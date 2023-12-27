import os
from termcolor import colored
from pathlib import Path
import csv
import torch

ROOT_FOOLDER = Path(os.path.dirname(__file__))
TRAINING_DIR_PATH = Path(os.path.join(os.path.dirname(__file__) + "/Data_crop" + "/train"))
LOG_SAVE_DIR_NAME = ROOT_FOOLDER/"Saves/logs/"
CKPT_SAVE_DIR_NAME= ROOT_FOOLDER/"Saves/ckpt/"
TEST_DIR_PATH = Path(os.path.join(os.path.dirname(__file__) + "/Data_crop" + "/test"))


NUM_EPOCHS =  [20]#[10, 20, 30]
NUM_WORKERS = 11
BATCH_SIZE = 512

FC_LR = [1e-3]#, 1e-4]#, 1e-5]
CNN_LR = [1e-3]#, 1e-4]#, 1e-5]

CNN_WD = [0]#,0.01,0.1]
FC_WD = [0]#,0.01,0.1]

FC_DROPOUT = [0.5]#, 0.6]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_couples(dir):  # training_folder, eval_folder, test_folder):
    # Implement your logic to build front and rear couples here
    """_summary_

    Args:
        root_folder (str): _description_

    Returns:
        _type_: _description_
    """
    # Get a list of all items (files and subfolders) in the root folder
    res = []
    for action_folder in os.listdir(dir):
        #print(colored(action_folder, "red"))
        for action_image in os.listdir(dir / action_folder):
            # print(image_folder)
            res.append((dir/action_folder/action_image,action_folder))



    csv_file_path = 'predictions2.csv'
    # Open the CSV file in write mode
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)
        #Write the header if needed (optional)
        csv_writer.writerow(['Image','Label'])
        for elem in res:

            # Write the predictions to the CSV file
            csv_writer.writerow((str(elem[0]),elem[1]))

    print(f'Train data been written to {csv_file_path}.')


def save_last_ckpt_path(original_path):
     with open("last_ckpt.txt","r+") as f:
        #print(original_path)
        v = f.readline().strip()
        #print(v)
        if v == "":
            #print("qui")
            f.write("v = 0\n")
            f.write(original_path+".ckpt\n")
        
        else:    
            v  = int(v.split("=")[1])
            #print(v)
            path = f.readline().strip()
            #print(path)
        
            path_args = path.split("-")
            #print(path_args)
            if len(path_args)>1 and path_args[0]+".ckpt" == original_path+".ckpt":
                #print("ELLE",path_args[0])
                f.seek(0)
                f.writelines(["v = " + str(v+1) + "\n",original_path + "-v"+str(v+1) +".ckpt"+ "\n"])
                f.truncate()

                #f.write("v = " + str(v+1) + "\n")
                #f.write(original_path + "-v"+str(v+1))

            elif len(path_args) == 1 and path_args[0] == original_path+".ckpt":
                #print("AUMENTO V")
                f.seek(0)
                f.writelines(["v = " + str(v+1)+"\n",original_path + "-v"+str(v+1)+".ckpt"+"\n"])
                f.truncate()

                #f.write("v = " + str(v+1))
                #f.write(original_path + "-v"+str(v+1)+".ckpt")

            else:
                f.seek(0)
                f.writelines(["v = 0" + "\n",original_path+".ckpt"+"\n"])
                f.truncate()

from PIL import Image

def manipola_immagine(immagine_path,augmentation_operation,dest):
    
    for image in os.listdir(immagine_path):
        print(image)
    # Carica l'immagine
        img = Image.open(immagine_path/image)

        # Estrai la barra di 12 pixel dal basso
        barra = img.crop((0, 84, 96, 96))

        # Flippa il resto dell'immagine
        resto = img.crop((0, 0, 96, 84))
        for transformation, transformation_name in augmentation_operation:
            resto_temp = resto
            if transformation != None:
                resto_temp = resto.transpose(transformation)

            # Crea una nuova immagine con la barra inserita
            nuova_immagine = Image.new("RGB", (96, 84))
            nuova_immagine.paste(resto_temp, (0, 0))
            #nuova_immagine.paste(barra, (0, 84))

            # Salva l'immagine risultante
            nuova_immagine.save(dest/Path(image.split(".")[0] + transformation_name +".png"))

