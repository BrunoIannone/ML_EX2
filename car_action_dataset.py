from torch.utils.data import Dataset
#import stud.transformer_utils as transformer_utils
from torchvision.io import read_image

from typing import List
import time
import os
class CarActionDataset(Dataset):
    """Car action dataset class
    """
    def __init__(self, samples: List[tuple]):
        """Constructor for car action dataset

        Args:
            samples (List[tuple]): List of tuples (image_path, action_label)
            
        """
        
        self.samples = samples 
        
        
        
        
    

    def __len__(self):
        """Return samples length

        Returns:
            int: length of samples list (number of samples)
        """
        return len(self.samples)

    def __getitem__(self, index: int):
        
        """Get item for dataloader input

        Args:
            index (int): index-th sample to access

        Returns:
            tuple: (image, labels) open image_path and return the tuple (image,label) related to the index-th element 
        """
        
        #convert index-th sample senses in indices
        
        image = read_image(self.samples[index][0])
        
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        
            
        return image, self.labels[index][1]
    
        