import torch.nn as nn
from torch.nn.utils import spectral_norm

#create a discriminator
class Discriminator(nn.Module):
    '''
    Discriminator class of type nn.Module
    '''

    def __init__(self):
        '''
        Initialize the discriminator with 4 convolution blocks
        '''
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,64,4,2,1, bias = False),
            nn.LeakyReLU(0.2,inplace = True),
            
            spectral_norm(nn.Conv2d(64,128,4,2,1, bias = False)),
            nn.LeakyReLU(0.2,inplace = True),

            spectral_norm(nn.Conv2d(128,256,4,2,1, bias = False)),
            nn.LeakyReLU(0.2,inplace = True),

            spectral_norm(nn.Conv2d(256,512,4,2,1, bias = False)),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(512,1,4,1,0),
            nn.Sigmoid()
        )
    def forward(self, input):
        '''
        Forward Propogation

        Parameters:
            - x (tensor) : Input to the model of shape -> Input to the model of shape -> batch_size*in_channels*image_size_width*image_size_height
        Returns:
            - A tensor
        '''

        output = self.main(input)
        return output
        #return output.view(-1)