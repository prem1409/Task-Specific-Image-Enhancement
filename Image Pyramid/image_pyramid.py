
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
# from PIL import Image
# from resizeimage import resizeimage


class ImageEnhancement:
    def gaussian_blurring(self,input_image,kernel_size,sigma):
        """Get Guassian Blurred image.
        @param input_image: The source image.
        @param kernel_size: Size of the filter
        @param sigma:Control variation around its mean value
        @return The gaussian blurred image
        """
        #Applying Gaussian Blur filter
        output_image=cv2.GaussianBlur(input_image,kernel_size,sigma)
        return output_image
    def sampling(self,input_image,width,height):
        """Resizing the image
        @param input_image: The source image.
        @param width:Width of new image
        @param height:Height of new image
        @return The resized image
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
        #Get transmission map from the image
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

        L=356
        #Applying atmosheric scattering model on the image
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
        #Read images from input dir
        inputdir = os.path.join(imgdir, 'inputs')
        shutil.rmtree(resultdir)
        os.mkdir(resultdir)
        #Read files from input images
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
                #Extract haze from the scene and then save the image
                dehazed = self.get_scene_radiance(image)
                cv2.imwrite(os.path.join(resultdir, basename), dehazed)
        return os.path.join(resultdir, basename)
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
        #Extracting edges using Canny's Edge Detection
        edges = cv2.Canny(img,80,255)
        cv2.imwrite(os.path.join(dir_path, dirname,'inputs','edges.png'),edges)
        kernel = (3,3)
        #Applying image pyramid technique
        #Applying Gaussian blur filter over the image
        gaussian_blurred_image =self.gaussian_blurring(img,kernel,0)
        cv2.imwrite(os.path.join(dir_path, dirname,'inputs','gaussian_blurred_image.png'),gaussian_blurred_image)
        plt.subplot(121),
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),
        plt.xticks([]), plt.yticks([])
        #Downsizing the image to 1/4th of its original size
        coarse_image =self.sampling(gaussian_blurred_image,0.25,0.25)        
        cv2.imwrite(os.path.join(dir_path, dirname,'inputs','coarse_image.png'),coarse_image)
        #Upsampling the image to its original size
        up_sampling=self.sampling(coarse_image,4,4)
        cv2.imwrite(os.path.join(dir_path, dirname,'inputs','up_sampling.png'),up_sampling)
        #Applying Gaussian Blur filtering
        gaus=self.gaussian_blurring(up_sampling,kernel,0)
        cv2.imwrite(os.path.join(dir_path, dirname,'inputs','gaus2.png'),gaus)
        #Resizing the image for image subtraction
        gaussian_blurred_image=cv2.resize(img,(gaus.shape[1],gaus.shape[0]))
        #Convert into grayscale
        gaus_gray=cv2.cvtColor(gaus,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(dir_path, dirname,'inputs','gausgray.png'),gaus_gray)
        #Converting to grayscale
        dst_gray=cv2.cvtColor(gaussian_blurred_image,cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(gaus_gray, dst_gray, full=True)
        diff = (diff * 255).astype("uint8")
        #Image Subtraction
        detail_image = cv2.subtract(gaus,gaussian_blurred_image)
        cv2.imwrite(os.path.join(dir_path, dirname,'inputs','detailed.png'),detail_image)
        print(detail_image.shape)
        output_path=self.process_imgdir(os.path.join(dir_path, dirname))
        dehazed_image=cv2.imread(output_path)
        # dehazed_image =self.sampling(dehazed_image,4,4)
        output_path="\\".join(output_path.split("\\")[:-1])
        print(dehazed_image.shape)
        cv2.imwrite(os.path.join(output_path,'dehazed_image.png'),dehazed_image)  
        #Adding two images
        dst = cv2.addWeighted(detail_image,1,dehazed_image,1,0)  
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        dst = cv2.filter2D(dst, -1, kernel)
        #Converting images to lightness,chroma ,hue for increasing the brightness
        lab= cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        #Applying CLAHE Algorithm for contrast amplification which is limited and to reduce the problem of noise amplification
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l) 
        limg = cv2.merge((cl,a,b))
        #Convert back to rgb
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)        
        psf = np.ones((5, 5)) / 25
        #Applying mean denoising filtering
        dst=cv2.fastNlMeansDenoisingColored(final,None,10,10,7,21)
        edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        print(edges.shape)
        edges=cv2.resize(edges,(dst.shape[1],dst.shape[0]))
        #Increasing the brightness of the image
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        value = 30 #whatever value you want to add
        lim=255-value
        
        s[s>lim]=255
        s[s<lim]+=value
        value1=30
        lim1=255-value1
        v[v>lim1]=255
        v[v<lim1]+=value1
        hsv = cv2.merge((h, s, v))
        dst = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #Writing the output file 
        dst = cv2.addWeighted(dst,1,edges,1,0)
        cv2.imwrite(os.path.join(output_path,file_name+'.png'),dst)
        #Resizing the file to compare it with other methods
        resized = cv2.resize(dst, (256,256), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output_path,'result_resized.png'),resized)
if __name__=="__main__":
    #Input file path
    path="dataset\\1_1_0.90179.png"
    #loading the file
    img = cv2.imread(path)
    print(img)
    #Extracting file name
    file_name=".".join(path.split("\\")[-1].split(".")[:-1])
    #calling the function
    ImageEnhancement().image_enhancement(img,file_name)