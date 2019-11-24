#!/usr/bin/env python
# coding: utf-8

# In[350]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import os
import glob
from torch.utils.data.dataset import Dataset
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler

#importing pretrained vgg19 model to calculate perceptual loss
from torchvision.models import vgg16


# In[351]:


image_size = 224
batch_size = 10


# In[399]:


hazy_img_path ='C:\\Users\\utkar\\Downloads\\ITS\\hazy\\'
clear_img_path = 'C:\\Users\\utkar\\Downloads\\ITS\\clear\\'
hazy_lst = os.listdir(hazy_img_path)
clear_lst = os.listdir(clear_img_path)

hazy_list = []
clear_list = []
for each_hazy in hazy_lst:
    hazy_list.append(hazy_img_path+each_hazy)
    clear_list.append(clear_img_path+each_hazy.split('_')[0]+'.png')


# In[400]:


transform = transforms.Compose([
    transforms.Resize(image_size) ,
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
])


# In[420]:


class CustomDataset(Dataset):
    def __init__(self, hazy_list, clear_list, transform,train=True):
        self.image_paths = hazy_list
        self.target_paths = clear_list
        self.transforms = transform
        
    def __getitem__(self, index):
        hazy_image = Image.open(self.image_paths[index])
        clear_image = Image.open(self.target_paths[index])
        t_hazy_image = self.transforms(hazy_image)
        t_clear_image = self.transforms(clear_image)
        return t_hazy_image, t_clear_image
    
    def __len__(self):
        return len(self.image_paths)


# In[421]:


train_size = 0.9
indices = list(range(len(data_list)))
split = int(np.floor(train_size*len(data_list)))
train_idx, val_idx = indices[:split], indices[split:]

train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(val_idx)

train_dataset = CustomDataset(hazy_list, clear_list,transform, train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)

test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0)


# In[422]:


# x, y = next(iter(train_dataset))
# type(x)


# In[423]:


# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.model = nn.Sequential( 
#             nn.ConvTranspose2d(3,512,4,1,0,bias = False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(512,256,4,2,1,bias = False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(256,128,4,2,1,bias = False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(128,64,4,2,1,bias = False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(64,3,4,2,1,bias = False),
#             nn.Tanh()
#         )
        
#     def forward(self,x):
#         x = self.model(x)
#         return x


# In[445]:


#create a discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,64,4,2,1, bias = False),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(64,128,4,2,1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace = True),

            nn.Conv2d(128,256,4,2,1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace = True),

            nn.Conv2d(256,512,4,2,1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(512,1,4,1,0),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        return output
        #return output.view(-1)
    


# In[446]:


disc = Discriminator()
disc


# In[426]:


# h_img = torch.Tensor(cv2.imread('./hazy_data/1.jpg'))
# h_img = h_img.view(1,h_img.shape[0],h_img.shape[1],h_img.shape[2])
# h_img = h_img.permute(0,3,1,2)
# c_img = torch.Tensor(cv2.imread('./hazy_data/2.jpg'))
# c_img = c_img.view(1,c_img.shape[0],c_img.shape[1],c_img.shape[2])
# c_img = c_img.permute(0,3,1,2)


# In[427]:


# perceptual_loss = 0
# perceptual_loss_lst = []
# percep_criterion = nn.MSELoss()
# for clear_layer, hazy_layer in zip(clear_vgg_model,gen_vgg_model) :
#     if isinstance(clear_layer,nn.ReLU) and isinstance(hazy_layer,nn.ReLU):
#         #print(clear_layer.modules().item())
#         perceptual_loss += percep_criterion(clear_layer, hazy_layer)

# print(perceptual_loss)


# In[428]:


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        vgg16_model = vgg16(pretrained = True)
        self.vgg16 = nn.Sequential(*list(vgg16_model.features.children())[:9])
        
    def forward(self,x):
        x = self.vgg16(x)
        return x


# In[429]:


# perceptual_criterion = nn.MSELoss()
# feature_extractor = FeatureExtractor()
# feature_extractor.eval()
# gen_feature = feature_extractor(h_img)
# real_feature = feature_extractor(c_img)

# perceptual_loss =  perceptual_criterion(gen_feature,real_feature)


# In[430]:


#print('Perceptual Loss is :-',perceptual_loss.item())


# In[431]:


# #Content Loss
# content_criterion = nn.L1Loss()
# content_loss = content_criterion(gen_img, real_img)

# print('Conten Loss is :-', content_loss)


# In[432]:


# #GAN Loss
# gan_criterion = nn.BCELoss()


# In[433]:


class DoubleConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        
    def forward(self,x):
        return self.model(x)

class DoubleConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2,inplace = True),
                
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2,inplace = True))

    def forward(self,x):
        return self.model(x)


# In[465]:


#Unet
class GenUNet(nn.Module):
    def __init__(self):
        super().__init__()
        

        self.dconv_down1 = DoubleConvDownBlock(3, 64)
        self.dconv_down2 = DoubleConvDownBlock(64, 128)
        self.dconv_down3 = DoubleConvDownBlock(128, 256)
        self.dconv_down4 = DoubleConvDownBlock(256, 512)        

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = DoubleConvUpBlock(256 + 512, 256)
        self.dconv_up2 = DoubleConvUpBlock(128 + 256, 128)
        self.dconv_up1 = DoubleConvUpBlock(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 3,1)
        
        self.maxpool = nn.MaxPool2d(2)

        
    def forward(self, x):
        
        dconv1 = self.dconv_down1(x)
        x = self.maxpool(dconv1)
        
        dconv2 = self.dconv_down2(x)
        x = self.maxpool(dconv2)
        
        dconv3 = self.dconv_down3(x)
        x = self.maxpool(dconv3)
        
        dconv4 = self.dconv_down4(x)
        
        x = self.upsample(x)
        x = torch.cat([x, dconv3], dim=1)
        x = self.dconv_up3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, dconv2], dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, dconv1], dim=1)
        x = self.dconv_up1(x)
        
        x = self.conv_last(x)
        x = F.tanh(x)
        
        return x    


# In[466]:


gen = GenUNet()
disc = Discriminator()


# In[467]:


disc


# In[468]:


#Training DCGans
gan_criterion = nn.BCELoss()
content_criterion = nn.L1Loss()
perceptual_criterion = nn.MSELoss()
feature_extractor = FeatureExtractor()
feature_extractor.eval()


# In[469]:


optimizerD = optim.Adam(disc.parameters(), lr = 0.0002, betas = (0.5,0.999))
optimizerG = optim.Adam(gen.parameters(), lr = 0.0002, betas = (0.5,0.999))


# In[470]:


alpha = 1
beta = 1
epochs = 25
for e in range(epochs):
    for i, data in enumerate(train_loader):
            
        hazy_images, clear_images = data
        
        disc.zero_grad()
        
        #training discriminator with real images
        target = Variable(torch.ones(clear_images.shape[0], 1,11,15))
        print(target.shape)
        output = disc.forward(clear_images)
        print(output.shape)
        errorD_real = criterion(output, target)
        
        
        #training discriminator with fake images
        fake_images = gen(hazy_images)
        target = Variable(torch.zeros(clear_images.shape[0], 1,11,15))
        output = disc.forward(fake_images)
        errorD_fake = gan_criterion(output, target)
        
        #Total Discriminator Error
        errorD = errorD_real + errorD_fake
        errorD.backward()
        optimizerD.step()
        
        
        #update the weights of Generator, now the target variable will be 1 as we want the generator to generate real images
        gen.zero_grad()
        
        target = Variable(torch.ones(real_images.size()[0], 1,11,15))
        output = disc(fake_images)
        gan_loss = gan_criterion(output, target)
        
        #content loss
        content_loss = content_criterion(fake_images, clear_images)
        
        #perceptual loss
        gen_feature = feature_extractor(fake_images)
        real_feature = feature_extractor(clear_images)
        perceptual_loss =  perceptual_criterion(gen_feature,real_feature).detach()
           
        #total Generator loss
        errorG = ganloss + alpha*content_loss + beta*perceptual_loss       
        errorG.backward()
        optimizerG.step()
        
        
        print('{:4f} Generator Loss :- {:6f}, Discriminator Loss :- {:6f}'.format(e, errorG.items(), errorD.items()))        


# In[ ]:


data_list[588]


# In[ ]:





# In[ ]:




