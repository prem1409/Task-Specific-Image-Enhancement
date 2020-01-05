from torch import optim
import torch.nn as nn
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
import os
from torchvision import datasets, transforms
from model.generator import Generator, init_weights
from model.discriminator import Discriminator
from model.utils.feature_extractor import FeatureExtractor
from model.utils.split_train_val import split_train_val
import configparser


#check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#read the configuration file
config = configparser.ConfigParser()
config.read('cGAN based Approach/config.ini')

hazy_img_path = config['Train']['hazy_img_path']
clear_img_path = config['Train']['clear_img_path']
results_path = config['Train']['results_path']
batch_size = int(config['Train']['batch_size'])
train_size = float(config['Train']['train_size'])
epochs = int(config['Train']['epochs'])

checkpoint_tar = config['Common']['checkpoint_tar']
image_size = int(config['Common']['image_size'])

hazy_lst = os.listdir(hazy_img_path)
clear_lst = os.listdir(clear_img_path)

hazy_list = []
clear_list = []
for each_hazy in hazy_lst:
    hazy_list.append(hazy_img_path+each_hazy)
    clear_list.append(clear_img_path+each_hazy.split('_')[0]+'.png')

#divide the dataset into train/val
train_loader, test_loader = split_train_val(train_size, image_size, hazy_list, clear_list, batch_size)



#inititialize the Generator,Discriminator and move them to GPU if available
gen = Generator()
gen.apply(init_weights)
gen.to(device)

disc = Discriminator()
disc.apply(init_weights)
disc.to(device)


#initialize the gan,content,perceptual and brightness loss
gan_criterion = nn.BCELoss().to(device)
content_criterion = nn.L1Loss().to(device)
perceptual_criterion = nn.MSELoss().to(device)
brightness_criterion = nn.L1Loss().to(device)

#set the feature extractor to evaluation mode as it will be used only to calculate perceptual loss
feature_extractor = FeatureExtractor()
feature_extractor.eval()
feature_extractor.to(device)

#initialize the optimizers for Generator and Discriminator
optimizerD = optim.Adam(disc.parameters(), lr = 0.0003, betas = (0.5,0.999))
optimizerG = optim.Adam(gen.parameters(), lr = 0.0001, betas = (0.5,0.999))


alpha = 0.5
beta = 1.8
gamma = 1.97
delta = 0.069

resume_epoch = 0


for e in range(resume_epoch,epochs):
    for i, data in enumerate(train_loader):
            
        hazy_images, clear_images = data
        
        #to prevent accumulation of gradients
        disc.zero_grad()
        
        #training discriminator with real images
        target = Variable(torch.ones(clear_images.shape[0], 1,11,11)*0.90,requires_grad=False).to(device) #added 0.9 for label smoothning
        output = disc.forward(clear_images)
        errorD_real = gan_criterion(output, target)
        
        #print(hazy_images.shape)
        
        #training discriminator with fake images
        fake_images = gen(hazy_images)
        target = Variable(torch.zeros(clear_images.shape[0], 1,11,11),requires_grad=False).to(device)
        output = disc.forward(fake_images.detach())
        errorD_fake = gan_criterion(output, target)
        
        #Total Discriminator Error
        errorD = errorD_real + errorD_fake
        errorD.backward()
        optimizerD.step()
        
        
        #update the weights of Generator, now the target variable will be 1 as we want the generator to generate real images
        gen.zero_grad()
        
        target = Variable(torch.ones(clear_images.size()[0], 1,11,11),requires_grad=False).to(device)
        output = disc(fake_images).detach()
        gan_loss = gan_criterion(output, target)
        
        #content loss with total regularization parameter
        content_loss = content_criterion(fake_images, clear_images)
        
        #perceptual loss
        gen_feature = feature_extractor(fake_images)
        real_feature = feature_extractor(clear_images).detach()
        perceptual_loss =  perceptual_criterion(gen_feature,real_feature)


        #estimate the value component of HSV from RGB image and calculate the brightness loss
        vue_clear_images = torch.max(torch.max(clear_images[:,0],clear_images[:,1]), clear_images[:,2])
        vue_fake_images = torch.max(torch.max(fake_images[:,0],fake_images[:,1]), fake_images[:,2])
        brightness_loss = brightness_criterion(vue_clear_images, vue_fake_images)

           
        #total Generator loss
        print('gan loss :- {:5f}, content loss :- {:5f}, perceptual loss :- {:5f}, brightness loss :- {:5f}'.format(gan_loss, content_loss, perceptual_loss, brightness_loss))
        errorG = alpha*gan_loss + beta*content_loss + gamma*perceptual_loss + delta*brightness_loss  
        
        #Perform backward propogation
        errorG.backward()
        optimizerG.step()
        

        print('Epoch:-{} [{}/{}] Generator Loss :- {:6f}, Discriminator Loss :- {:6f}'.format(e,i,len(train_loader), errorG.item(), errorD.item()))
        
        #save the weights and images every 200th iteration of an epoch
        if i%200 == 0:
            for j, test_data in enumerate(test_loader):
                gen.eval()

                test_hazy_images, test_clear_images = test_data
                vutils.save_image(test_clear_images, results_path+'real_samples.png', normalize = True)

                fake = gen(test_hazy_images)
                vutils.save_image(fake.data, results_path+'fake_samples_epoch{}_{}_{}.png'.format(e,i,j), normalize = True)
                gen.train()


                #save batches
                torch.save({'epochs': e,
                            'gen_state_dict' :gen.state_dict(),
                            'disc_state_dict' :disc.state_dict(),
                            'optimizerG_state_dict': optimizerG.state_dict(),
                            'optimizerD_state_dict': optimizerD.state_dict(),
                            'errorG':errorG,
                            'errorD':errorD
                            }, checkpoint_tar)

                break