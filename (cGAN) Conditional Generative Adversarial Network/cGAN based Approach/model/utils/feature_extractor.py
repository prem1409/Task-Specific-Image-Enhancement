import torch.nn as nn
from torchvision.models import vgg16

class FeatureExtractor(nn.Module):
    '''
    VGG16 Network, used to calculate perceptual loss for the Generator.
    - Both Clear, generated Haze-free is passed and difference is estimated.
    '''

    def __init__(self):
        '''
            Initialize the FeatureExtractor, Constructor
        Parameters:
            - None

        Returns:
            - Object of type nn.Module
        '''
        super().__init__()
        
        vgg16_model = vgg16(pretrained = True)
        self.vgg16 = nn.Sequential(*list(vgg16_model.features.children())[:15])
        
    def forward(self,x):
        '''
            Forward propogation
        Paremeters:
            - x (tensor) : Input to the model of shape -> batch_size*no. of channels*image_size_width*image_size_height

        Returns:
            - A tensor 


        '''
        x = self.vgg16(x)
        return x