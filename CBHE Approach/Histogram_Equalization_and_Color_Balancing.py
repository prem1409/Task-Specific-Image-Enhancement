
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2 
import os
import shutil
import sys
from skimage.measure import compare_ssim
# from scipy.misc import imfilter
from skimage import color, data, restoration
from scipy.signal import convolve2d 
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


from skimage.color import rgb2hsv,hsv2rgb


class ImageEnhancement:
    def gaussian_blurring(self,input_image,kernel_size,sigma):
        """Smoothening of an image.
        @param input_img: The source image.
        @kernel_size: size of the blur filter
        @sigma: Standard deviation for Gaussian kernel.
        @return The blurred image.
        """
        #Applying Gaussian Blur filter
        output_image=cv2.GaussianBlur(input_image,kernel_size,sigma)
        # output_image=cv2.medianBlur(input_image,kernel_size)
        return output_image
    def sampling(self,input_image,width,height):
        """sampling of the image.
        @param input_img: The image to be sampled.
        @width: width to which the image needs to be sampled to.
        @height: height to which the image needs to be sampled to.
        @return The sampled image.
        """
        #Resizing the image
        output_image=cv2.resize(input_image,None,fx=width,fy=height)
        return output_image
    def get_dark_channel(self,img, *, size):
        """Get dark channel for an image.
        @param img: The source image.
        @param size: Patch size.
        @return The dark channel of the image.
        """
        #Extract the dark/hazy part from the image
        minch = np.amin(img, axis=2)
        box = cv2.getStructuringElement(cv2.MORPH_RECT, (size // 2, size // 2))
        return cv2.erode(minch, box)
    def get_atmospheric_light(self,img, *, size, percent):
        """Estimate atmospheric light for an image.
        @param img: the source image.
        @param size: Patch size for calculating the dark channel.
        @param percent: Percentage of brightest pixels in the dark channel
        considered for the estimation.
        @return The estimated atmospheric light.
        """
        #Get the atmospheric light factor from the image
        m, n, _ = img.shape

        flat_img = img.reshape(m * n, 3)
        flat_dark = self.get_dark_channel(img, size=size).ravel()
        count = math.ceil(m * n * percent / 100)
        indices = np.argpartition(flat_dark, -count)[:-count]

        return np.amax(np.take(flat_img, indices, axis=0), axis=0)
    def get_transmission(self,img, atmosphere, *, size, omega, radius, epsilon):
        """Estimate transmission map of an image.
        @param img: The source image.
        @param atmosphere: The atmospheric light for the image.
        @param omega: Factor to preserve minor amounts of haze [1].
        @param radius: (default: 40) Radius for the guided filter [2].
        @param epsilon: (default: 0.001) Epsilon for the guided filter [2].
        @return The transmission map for the source image.
        """
        division = np.float64(img) / np.float64(atmosphere)
        raw = (1 - omega * self.get_dark_channel(division, size=size)).astype(np.float32)
        return cv2.ximgproc.guidedFilter(img, raw, radius, epsilon)


    def get_scene_radiance(self, img,*,size=15,omega=0.95,trans_lb=0.1,percent=0.1,radius=40,epsilon=0.001):
        """Get recovered scene radiance for a hazy image.
        @param img: The source image to be dehazed.
        @param omega: (default: 0.95) Factor to preserve minor amounts of haze [1].
        @param trans_lb: (default: 0.1) Lower bound for transmission [1].
        @param size: (default: 15) Patch size for filtering etc [1].
        @param percent: (default: 0.1) Percentage of pixels chosen to compute atmospheric light [1].
        @param radius: (default: 40) Radius for the guided filter [2].
        @param epsilon: (default: 0.001) Epsilon for the guided filter [2].
        @return The final dehazed image.
        
        """
        #Applying atmosheric scattering model on the image
        L=356
        atmosphere = self.get_atmospheric_light(img, size=size, percent=percent)
        trans = self.get_transmission(img, atmosphere, size=size, omega=omega, radius=radius, epsilon=epsilon)
        clamped = np.clip(trans, trans_lb, omega)[:, :, None]
        img = np.float64(img)
        return np.uint8(((img - atmosphere) / clamped + atmosphere).clip(0, L - 1))


    def process_imgdir(self,imgdir):
        """Get haze free images in the directory
        @param imgdir: The source image directory.
        @return All the haze free images.
        """
          #Write images into resultdir
        resultdir = os.path.join(imgdir, 'results')
        inputdir = os.path.join(imgdir, 'inputs')
        shutil.rmtree(resultdir)
        os.mkdir(resultdir)
        #Read images from input dir
        for fullname in os.listdir(inputdir):
            filepath = os.path.join(inputdir, fullname)
            if os.path.isfile(filepath):
                basename = os.path.basename(filepath)
                image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    print('Processing %s ...' % basename)
                else:
                    sys.stderr.write('Skipping %s, not RGB' % basename)
                    continue
                dehazed = self.get_scene_radiance(image)
                # side_by_side = np.concatenate((image, dehazed), axis=1)
                cv2.imwrite(os.path.join(resultdir, basename), dehazed)
        return os.path.join(resultdir, basename)

        #Funtion for Color Balancing of the images

    def cal_equalisation(self,img,ratio):
        '''
        @param img: The source hazy image
        @param ratio: RGB average ratio
        @return the array after averaging the color
        '''
        Array = img * ratio
        Array = np.clip(Array, 0, 255)
        return Array


    def RGB_equalisation(self, img):
        '''
        @param img: source hazy image
        @return: Color balanced image
        '''
        img = np.float32(img)
        avg_RGB = []
        for i in range(3):
            avg = np.mean(img[:,:,i])
            avg_RGB.append(avg)
        # print('avg_RGB',avg_RGB)
        a_r = avg_RGB[0]/avg_RGB[2]
        a_g =  avg_RGB[0]/avg_RGB[1]
        ratio = [0,a_g,a_r]
      
        for i in range(1,3):
            img[:,:,i] = self.cal_equalisation(img[:,:,i],ratio[i])
         
        return img
   
# Dehazing and enhancing the dehazed image
    
    def image_enhancement(self,img,file_name):
        """Main function to call all the functions
        @param img: Input image
        @param file_name: The output file name
        @return All the haze free images.
        """
        #Creating output directory if it doesnt exist
        dirname = 'output'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if(os.path.isdir(os.path.join(dir_path, dirname))): 
            if(os.path.exists(os.path.join(dir_path, dirname))):
                pass
        else:
            os.mkdir(os.path.join(dir_path, dirname))
            os.mkdir(os.path.join(dir_path, dirname,"results"))
            os.mkdir(os.path.join(dir_path, dirname,"inputs"))

        #Performing Color Balancing on the images

    
        kernel = (3,3)

        sceneRadiance = self.RGB_equalisation(img)
        sceneRadiance=sceneRadiance.astype(np.uint8)
        #Histogram Equalization of the color balanced image

        R, G, B = cv2.split(sceneRadiance)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)

        equ = cv2.merge((output1_R, output1_G, output1_B))

        #Smoothening of the histogram equalized image using Gaussian Filter       
        gaussian_blurred_image =self.gaussian_blurring(equ,kernel,0)
        plt.subplot(121),plt.imshow(img),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(gaussian_blurred_image),plt.title('Averaging')
        plt.xticks([]), plt.yticks([])
        #Down sizing of the smoothen image
       
        coarse_image =self.sampling(gaussian_blurred_image,0.25,0.25)
    
        
        cv2.imwrite(os.path.join(dir_path, dirname,'inputs','coarse_image.png'),coarse_image)
        plt.imshow(coarse_image)
        plt.title("Coarse Image")

        #Smoothening of the downsampled image to remove any artifacts
        
        
        gaus=self.gaussian_blurring(coarse_image,kernel,0)
        #Umsampling of the smoothened image

        up_sampling=self.sampling(gaus,4,4)
        
        plt.imshow(gaus)

        #Resizing of the image to 256*256 
       
        gaussian_blurred_image=cv2.resize(gaussian_blurred_image,(up_sampling.shape[1],up_sampling.shape[0]))
        
        #Resizing the file to compare it with other methods
        resized_image = resize_img = cv2.resize(gaussian_blurred_image  , (256 , 256))
          
        
        output_path=self.process_imgdir(os.path.join(dir_path, dirname))
        print(output_path)
        dehazed_image=cv2.imread(output_path)
        # plt.imshow(dehazed_image)
        # plt.show()    
        dehazed_image =self.sampling(dehazed_image,4,4)
        output_path="\\".join(output_path.split("\\")[:-1])
        cv2.imwrite(os.path.join(output_path,'dehazed_image.png'),dehazed_image)
        plt.imshow(dehazed_image)

        #Writing the dehazed image into the result folder       
        
        cv2.imwrite(os.path.join(output_path,'finaldehazedimage.png'),resized_image )
        

if __name__=="__main__":
     #Input file path
    path = "dataset\\0042_0.8_0.2.jpg"
    #loading the file
    img = cv2.imread(path)
    print(img)
    #Extracting file name
    file_name=".".join(path.split("\\")[-1].split(".")[:-1])
    #calling the function
    ImageEnhancement().image_enhancement(img,file_name)