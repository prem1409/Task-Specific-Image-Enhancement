import torch
import torch.nn.functional as F
import torch.nn as nn


class DoubleConvUpBlock(nn.Module):
    '''
    Dense Block, used as a decoder for the Generator
    '''
    def __init__(self, in_channels, out_channels):
        '''
        initialize the Dense block
        Parameters:
            - in_channels (int) : number of input channels
            - out_channels (int) : number of output channels

        Returns:

            - An object of type nn.Module

        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels//4,3, padding =1, bias = False)
        self.conv2 = nn.Conv2d(out_channels//4,out_channels//4,3, padding =1, bias = False)
        self.conv3 = nn.Conv2d(out_channels//2,out_channels//4,3, padding =1, bias = False)
        self.conv4 = nn.Conv2d(3*out_channels//4,out_channels//4,3, padding =1, bias = False)

        self.batch_norm = nn.BatchNorm2d(in_channels)


    def forward(self,x):
        '''
        Forward Propogation

        Parameters:
            - x (tensor) : Input to the model of shape -> Input to the model of shape -> batch_size*in_channels*image_size_width*image_size_height
        Returns:
            - A tensor
        '''

        x = self.batch_norm(x)

        conv1 = self.conv1(x)
        conv1 = F.leaky_relu(conv1, 0.2, inplace = True)
        x = conv1      

        conv2 = self.conv2(x)
        conv2 = F.leaky_relu(conv2, 0.2, inplace = True)
        x = torch.cat([x, conv2], dim = 1)
        
        
        conv3 = self.conv3(x)
        conv3 = F.leaky_relu(conv3, 0.2, inplace = True)
        x = torch.cat([x, conv3], dim = 1)
        
        
        conv4 = self.conv4(x)
        conv4 = F.leaky_relu(conv4, 0.2, inplace = True)
        x = torch.cat([x, conv4], dim = 1)
        
        return x

