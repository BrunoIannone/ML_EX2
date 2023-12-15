import os
from termcolor import colored
from pathlib import Path
import csv
DATA_FOLDER = os.path.dirname(__file__) 
TRAINING_DIR_PATH = Path(os.path.join(os.path.dirname(__file__) + "/Data" + "/train"))

TEST_DIR_PATH = Path(os.path.join(os.path.dirname(__file__) + "/Data" + "/test"))


BATCH_SIZE = 16
NUM_WORKERS = 12
LEARNING_RATE = [1e-2,1e-3]
LEARNING_RATE2 = [1e-2,1e-3]
weight_decay = [0,0.001,0.01]
transformer_learning_rate = [1e-5]
transformer_weight_decay = [0,0.001,0.1]
LIN_DROPOUT = [0.5,0.8]
NUM_EPOCHS = 100


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
        print(colored(action_folder, "red"))
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


