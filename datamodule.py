from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import dataset
import utils

class WsdDataModule(LightningDataModule):
    """Datamodule for WSD with tensor-based models

    
    """
    def __init__(self,training_data:dict, valid_data:dict, test_data:dict,labels_to_idx:dict):
        """Init function for WSD datamodule with tensor-based models

        Args:
            training_data (dict): {sample:{List[sample_dicts]}} for training
            valid_data (dict): {sample:{List[sample_dicts]}} for valid
            test_data (dict): {sample:{List[sample_dicts]}} for test
            labels_to_idx (dict):  dictionary with structure {label:index} 
        """
        super().__init__()
        
        self.training_data = training_data
        self.valid_data = valid_data
        self.test_data = test_data
        

    def setup(self, stage: str):
        
        self.train_dataset = dataset.Dataset(self.training_data)
        self.valid_dataset = dataset.Dataset(self.valid_data)
        self.test_dataset = dataset.Dataset(self.test_data)

    def train_dataloader(self):
        
        return DataLoader(
            self.train_dataset,
            batch_size = utils.BATCH_SIZE,
            num_workers = utils.NUM_WORKERS,
            shuffle = False,
            collate_fn=utils.collate_fn
        ) 
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size = utils.BATCH_SIZE,
            num_workers = utils.NUM_WORKERS,
            shuffle = False,
            collate_fn=utils.collate_fn
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = utils.BATCH_SIZE,
            num_workers = utils.NUM_WORKERS,
            shuffle = False,
            collate_fn=utils.collate_fn
        )