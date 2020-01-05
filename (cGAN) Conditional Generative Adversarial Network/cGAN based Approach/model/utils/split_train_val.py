import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from .load_dataset import CustomDataset

def split_train_val(train_size, image_size, hazy_list, clear_list, batch_size):
    '''
        Divide the dataset into train/test

    Parameters:
        - train_size (float) : ratio to split train and validation 
        - image_size (int) : size of an input image
        - hazy_list (list,str) :  list of path  hazy images
        - clear_list (list,str) : list of path for clear images
        - batch_size (int) : size of each batch

    Returns:
        - Two generators (train_loader, test_loader)
    '''

    transform = transforms.Compose([
    transforms.Resize((image_size,image_size)) ,
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    indices = list(range(len(hazy_list)))
    split = int(np.floor(train_size*len(hazy_list)))
    train_idx, val_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SequentialSampler(val_idx)

    train_dataset = CustomDataset(hazy_list, clear_list,transform, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)

    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0)

    return train_loader, test_loader