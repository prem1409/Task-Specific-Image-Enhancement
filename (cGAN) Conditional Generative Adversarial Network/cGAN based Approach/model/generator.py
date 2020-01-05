import torch
import torch.nn as nn
from .utils.conv_down import DoubleConvDownBlock
from .utils.conv_up import DoubleConvUpBlock

class Generator(nn.Module):
    '''
    Generator class of type nn.Module
    '''
    def __init__(self):
        '''
        Initialize the encoder-decoder architecture with 3 layers of encoder, 1 bottle-neck dense block and then
        3 layer of decoder. 
        '''
        super().__init__()
        

        self.dconv_down1 = DoubleConvDownBlock(3, 64)
        self.dconv_down2 = DoubleConvDownBlock(64, 128)
        self.dconv_down3 = DoubleConvDownBlock(128, 256)
        self.dconv_down4 = DoubleConvDownBlock(256, 512)        

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = DoubleConvUpBlock(256 + 512, 256)
        self.dconv_up2 = DoubleConvUpBlock(128 + 256, 128)
        self.dconv_up1 = DoubleConvUpBlock(128 + 64, 64)
        
        self.conv_dense = nn.Conv2d(64, 3, 1)

        self.maxpool = nn.MaxPool2d(2)

        
    def forward(self, x):
        '''
        Forward Propogation, Skip connections are introduced between encoder and decoder.

        Parameters:
            - x (tensor) : Input to the model of shape -> Input to the model of shape -> batch_size*in_channels*image_size_width*image_size_height
        Returns:
            - A tensor
        '''

        residual = x
        
        dconv1 = self.dconv_down1(x)
        x = self.maxpool(dconv1)
        
        dconv2 = self.dconv_down2(x)
        x = self.maxpool(dconv2)
        
        dconv3 = self.dconv_down3(x)
        x = self.maxpool(dconv3)
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)
        x = torch.cat([x, dconv3], dim=1)
        x = self.dconv_up3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, dconv2], dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, dconv1], dim=1)
        x = self.dconv_up1(x)
        
        x = self.conv_dense(x)
        x-= residual

        x = torch.tanh(x)
        
        return x
    
#initialize the weights using Xavier initialization    
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)