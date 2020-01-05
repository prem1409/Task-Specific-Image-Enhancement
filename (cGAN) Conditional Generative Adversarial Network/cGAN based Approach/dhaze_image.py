import torch
from model.generator import Generator
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
import os
import configparser


#check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#read the configuration file
config = configparser.ConfigParser()
config.read('cGAN based Approach/config.ini')

test_data_path = config['Test']['test_data_path']
output_path = config['Test']['output_path']
image_size = int(config['Common']['image_size'])
checkpoint_tar = config['Common']['checkpoint_tar']

#normalize/resize an image
transform = transforms.Compose([
transforms.Resize((image_size,image_size)) ,
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#create a dataloader
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

#Initialize the Generator and load the pretrained weights
gen = Generator()
gen.to(device)

checkpoints = torch.load(checkpoint_tar, map_location= torch.device(device))
gen.load_state_dict(checkpoints['gen_state_dict'])

#pass the hazy image to the generator and save the resultant to an output_path location
for j, images in enumerate(test_loader):
    gen.eval()
    images = images[0].to(device)
    fake = gen(images)
    vutils.save_image(fake.data[0], output_path+'/fake_samples_{}.png'.format(j), normalize = True)


print('....Done....')