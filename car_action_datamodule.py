from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import car_action_dataset
import utils

class CarActionDataModule(LightningDataModule):
    """Datamodule for car action dataset.

    
    """
    def __init__(self,training_data:dict, valid_data:dict, test_data:dict):
        """Init function for car action datamodule

        Args:
            training_data (List[tuple]): List of tuple (image,label)
            valid_data (List[tuple]): List of tuple (image,label)
            test_data (List[tuple]): List of tuple (image,label)
        """
        super().__init__()
        
        self.training_data = training_data
        self.valid_data = valid_data
        self.test_data = test_data
        

    def setup(self, stage: str):
        
        self.train_dataset = car_action_dataset.CarActionDataset(self.training_data)
        self.valid_dataset = car_action_dataset.CarActionDataset(self.valid_data)
        self.test_dataset = car_action_dataset.CarActionDataset(self.test_data)

    def train_dataloader(self):
        
        return DataLoader(
            self.train_dataset,
            batch_size = utils.BATCH_SIZE,
            num_workers = utils.NUM_WORKERS,
            shuffle = True,
            #collate_fn=utils.collate_fn
        ) 
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size = utils.BATCH_SIZE,
            num_workers = utils.NUM_WORKERS,
            shuffle = False,
            #collate_fn=utils.collate_fn
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = utils.BATCH_SIZE,
            num_workers = utils.NUM_WORKERS,
            shuffle = False,
            #collate_fn=utils.collate_fn
        )