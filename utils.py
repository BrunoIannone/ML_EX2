import os
from termcolor import colored
from pathlib import Path


TRAINING_DIR_PATH = Path(os.path.join(os.path.dirname(__file__) + "/Data" + "/train"))



BATCH_SIZE = 16
NUM_WORKERS = 12
LEARNING_RATE = [1e-2,1e-3]
LEARNING_RATE2 = [1e-2,1e-3]
weight_decay = [0,0.001,0.01]
transformer_learning_rate = [1e-5]
transformer_weight_decay = [0,0.001,0.1]
LIN_DROPOUT = [0.5,0.8]
NUM_EPOCHS = 100


def _build_couples(dir):  # training_folder, eval_folder, test_folder):
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

        return res