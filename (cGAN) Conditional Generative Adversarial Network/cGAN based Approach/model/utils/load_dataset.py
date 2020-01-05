from torch.utils.data.dataset import Dataset
from PIL import Image
import torch

#check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    '''
    Custom dataset to combine hazy images with their clean counterpart
    '''
    def __init__(self, hazy_list, clear_list, transform,train=True):
        self.image_paths = hazy_list
        self.target_paths = clear_list
        self.transforms = transform
        
    def __getitem__(self, index):
        '''
        transform the hazy and the clean image.

        Paremeters :
         - index (int): number between 0 and lenght of the list

        Returns:
            - transformed hazy and clean image
        '''
        hazy_image = Image.open(self.image_paths[index])
        clear_image = Image.open(self.target_paths[index])
        t_hazy_image = self.transforms(hazy_image).to(device) #move data to cuda if available
        t_clear_image = self.transforms(clear_image).to(device) #move data to cuda if available
        return t_hazy_image, t_clear_image
    
    def __len__(self):
        '''
        returns the lenth of the dataset
        '''
        return len(self.image_paths)