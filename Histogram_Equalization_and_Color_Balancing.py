
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
        output_image=cv2.GaussianBlur(input_image,kernel_size,sigma)
        # output_image=cv2.medianBlur(input_image,kernel_size)
        return output_image
    def sampling(self,input_image,width,height):
        output_image=cv2.resize(input_image,None,fx=width,fy=height)
        return output_image
    def get_dark_channel(self,img, *, size):
        """Get dark channel for an image.
        @param img: The source image.
        @param size: Patch size.
        @return The dark channel of the image.
        """
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
        L=356
        atmosphere = self.get_atmospheric_light(img, size=size, percent=percent)
        trans = self.get_transmission(img, atmosphere, size=size, omega=omega, radius=radius, epsilon=epsilon)
        clamped = np.clip(trans, trans_lb, omega)[:, :, None]
        img = np.float64(img)
        return np.uint8(((img - atmosphere) / clamped + atmosphere).clip(0, L - 1))


    def process_imgdir(self,imgdir):
        resultdir = os.path.join(imgdir, 'results')
        inputdir = os.path.join(imgdir, 'inputs')
        shutil.rmtree(resultdir)
        os.mkdir(resultdir)
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


    def cal_equalisation(self,img,ratio):
        Array = img * ratio
        Array = np.clip(Array, 0, 255)
        return Array

    def RGB_equalisation(self, img):
        img = np.float32(img)
        avg_RGB = []
        for i in range(3):
            avg = np.mean(img[:,:,i])
            avg_RGB.append(avg)
        # print('avg_RGB',avg_RGB)
        a_r = avg_RGB[0]/avg_RGB[2]
        a_g =  avg_RGB[0]/avg_RGB[1]
        ratio = [0,a_g,a_r]
        # print(img.shape)
        for i in range(1,3):
            img[:,:,i] = self.cal_equalisation(img[:,:,i],ratio[i])
            # print(img[:,:,i])
        # print(img.shape)
        return img
   
    def image_enhancement(self,img):

    #Adding Histogram equalization and Color balacing

        edges = cv2.Canny(img,80,255)
        plt.imshow(edges)
        plt.show()
        kernel = (3,3)

        sceneRadiance = self.RGB_equalisation(img)
        sceneRadiance=sceneRadiance.astype(np.uint8)
       
        R, G, B = cv2.split(sceneRadiance)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)

        equ = cv2.merge((output1_R, output1_G, output1_B))

        
        gaussian_blurred_image =self.gaussian_blurring(equ,kernel,0)
        plt.subplot(121),plt.imshow(img),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(gaussian_blurred_image),plt.title('Averaging')
        plt.xticks([]), plt.yticks([])
        # plt.show()
        coarse_image =self.sampling(gaussian_blurred_image,0.25,0.25)
        dirname = 'test'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        os.mkdir(os.path.join(dir_path, dirname))
        os.mkdir(os.path.join(dir_path, dirname,"results"))
        os.mkdir(os.path.join(dir_path, dirname,"inputs"))
        
        cv2.imwrite(os.path.join(dir_path, dirname,'inputs','coarse_image.png'),coarse_image)
        plt.imshow(coarse_image)
        plt.title("Coarse Image")
        # plt.show()
        up_sampling=self.sampling(coarse_image,4,4)
        gaus=self.gaussian_blurring(up_sampling,kernel,0)
        # gaus=up_sampling
        plt.imshow(gaus)
        # plt.show()
        gaussian_blurred_image=cv2.resize(gaussian_blurred_image,(gaus.shape[1],gaus.shape[0]))
        gaus_gray=cv2.cvtColor(gaus,cv2.COLOR_BGR2GRAY)
        dst_gray=cv2.cvtColor(gaussian_blurred_image,cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(gaus_gray, dst_gray, full=True)
        diff = (diff * 255).astype("uint8")
        detail_image = cv2.subtract(gaus,gaussian_blurred_image)
        plt.imshow(detail_image)
        # plt.show()    


        
        output_path=self.process_imgdir(os.path.join(dir_path, dirname))
        print(output_path)
        dehazed_image=cv2.imread(output_path)
        plt.imshow(dehazed_image)
        # plt.show()    
        dehazed_image =self.sampling(dehazed_image,4,4)
        output_path="\\".join(output_path.split("\\")[:-1])
        cv2.imwrite(os.path.join(output_path,'dehazed_image.png'),dehazed_image)
        plt.imshow(dehazed_image)

        
        # R, G, B = cv2.split(img)

        # output1_R = cv2.equalizeHist(R)
        # output1_G = cv2.equalizeHist(G)
        # output1_B = cv2.equalizeHist(B)

        # equ = cv2.merge((output1_R, output1_G, output1_B))
        
       
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
        
      

        # plt.show()    
        dst = cv2.addWeighted(detail_image,1,dehazed_image,1,0)
        plt.imshow(dst)
        # plt.show()    
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        dst = cv2.filter2D(dst, -1, kernel)
        
        lab= cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)       
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l) 
        limg = cv2.merge((cl,a,b))

        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        cv2.imwrite(os.path.join(output_path,'final_output123.png'),gaussian_blurred_image )
        psf = np.ones((5, 5)) / 25
        dst=cv2.fastNlMeansDenoisingColored(gaussian_blurred_image,None,10,10,7,21)
        
        edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        print(edges.shape)
        edges=cv2.resize(edges,(dst.shape[1],dst.shape[0]))
        print(dst.shape)
        dst = cv2.addWeighted(dst,1,edges,1,0)
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
        hsv[:,:,2] += value1
        hsv[:,2,:] += value
        hsv = cv2.merge((h, s, v))
        dst = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        cv2.imwrite(os.path.join(output_path,'final_output.png'),dst)

if __name__=="__main__":
    img = cv2.imread("C:\\Users\\jatin\Desktop\\803 Project\\Testing_image\\0154_0.8_0.2.jpg")
    ImageEnhancement().image_enhancement(img)