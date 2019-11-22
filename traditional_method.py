
import numpy as np
from matplotlib import pyplot as plt
import math
import numpy as np
import cv2 as cv
import os
import shutil
import sys
from skimage.measure import compare_ssim
# from scipy.misc import imfilter
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

L = 256

def gaussian_blurring(input_image,kernel_size,sigma):
    output_image=cv.GaussianBlur(input_image,kernel_size,sigma)
    return output_image
def sampling(input_image,width,height):
    output_image=cv.resize(input_image,None,fx=width,fy=height)
    return output_image

def get_dark_channel(img, *, size):
    """Get dark channel for an image.
    @param img: The source image.
    @param size: Patch size.
    @return The dark channel of the image.
    """
    minch = np.amin(img, axis=2)
    box = cv.getStructuringElement(cv.MORPH_RECT, (size // 2, size // 2))
    return cv.erode(minch, box)


def get_atmospheric_light(img, *, size, percent):
    """Estimate atmospheric light for an image.
    @param img: the source image.
    @param size: Patch size for calculating the dark channel.
    @param percent: Percentage of brightest pixels in the dark channel
    considered for the estimation.
    @return The estimated atmospheric light.
    """
    m, n, _ = img.shape

    flat_img = img.reshape(m * n, 3)
    flat_dark = get_dark_channel(img, size=size).ravel()
    count = math.ceil(m * n * percent / 100)
    indices = np.argpartition(flat_dark, -count)[:-count]

    return np.amax(np.take(flat_img, indices, axis=0), axis=0)


def get_transmission(img, atmosphere, *, size, omega, radius, epsilon):
    """Estimate transmission map of an image.
    @param img: The source image.
    @param atmosphere: The atmospheric light for the image.
    @param omega: Factor to preserve minor amounts of haze [1].
    @param radius: (default: 40) Radius for the guided filter [2].
    @param epsilon: (default: 0.001) Epsilon for the guided filter [2].
    @return The transmission map for the source image.
    """
    division = np.float64(img) / np.float64(atmosphere)
    raw = (1 - omega * get_dark_channel(division, size=size)).astype(np.float32)
    return cv.ximgproc.guidedFilter(img, raw, radius, epsilon)


def get_scene_radiance(img,
                       *,
                       size=15,
                       omega=0.95,
                       trans_lb=0.1,
                       percent=0.1,
                       radius=40,
                       epsilon=0.001):
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
    atmosphere = get_atmospheric_light(img, size=size, percent=percent)
    trans = get_transmission(img, atmosphere, size=size, omega=omega, radius=radius, epsilon=epsilon)
    clamped = np.clip(trans, trans_lb, omega)[:, :, None]
    img = np.float64(img)
    return np.uint8(((img - atmosphere) / clamped + atmosphere).clip(0, L - 1))


def process_imgdir(imgdir):
    resultdir = os.path.join(imgdir, 'results')
    inputdir = os.path.join(imgdir, 'inputs')
    shutil.rmtree(resultdir)
    os.mkdir(resultdir)
    for fullname in os.listdir(inputdir):
        filepath = os.path.join(inputdir, fullname)
        if os.path.isfile(filepath):
            basename = os.path.basename(filepath)
            image = cv.imread(filepath, cv.IMREAD_COLOR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                print('Processing %s ...' % basename)
            else:
                sys.stderr.write('Skipping %s, not RGB' % basename)
                continue
            dehazed = get_scene_radiance(image)
            # side_by_side = np.concatenate((image, dehazed), axis=1)
            cv.imwrite(os.path.join(resultdir, basename), dehazed)
    return os.path.join(resultdir, basename)


if __name__=="__main__":
    img = cv.imread("dataset/1_3_0.84256.png")
    edges = cv.Canny(img,0,150)
    plt.imshow(edges)
    plt.show()
    # cv.imshow("image",img)
    # cv.waitKey(0)
    kernel = (3,3)
    print(type(kernel))
    dst =gaussian_blurring(img,kernel,0)
    # dst=img
    print(img.shape)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()
    coarse_image =sampling(img,0.25,0.25)
    dirname = 'test'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.mkdir(os.path.join(dir_path, dirname))
    os.mkdir(os.path.join(dir_path, dirname,"results"))
    os.mkdir(os.path.join(dir_path, dirname,"inputs"))
    
    cv.imwrite(os.path.join(dir_path, dirname,'inputs','coarse_image.png'),coarse_image)
    plt.imshow(coarse_image)
    plt.title("Coarse Image")
    plt.show()
    up_sampling=sampling(coarse_image,4,4)
    # gaus=gaussian_blurring(up_sampling,kernel,0)
    gaus=up_sampling
    plt.imshow(gaus)
    plt.show()
    gaus_gray=cv.cvtColor(gaus,cv.COLOR_BGR2GRAY)
    dst_gray=cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    print(gaus_gray.shape)
    print(dst_gray.shape)
    (score, diff) = compare_ssim(gaus_gray, dst_gray, full=True)
    diff = (diff * 255).astype("uint8")
    detail_image = cv.subtract(gaus,dst)
    # detail_image = cv.cvtColor(detail_image,cv.COLOR_GRAY2RGB)
    plt.imshow(detail_image)
    plt.show()    
    output_path=process_imgdir(os.path.join(dir_path, dirname))

    dehazedd_image=cv.imread(output_path)

    print("Output Path",output_path)
    print(dehazedd_image.shape)
    plt.imshow(dehazedd_image)
    plt.show()    
    dehazedd_image =sampling(dehazedd_image,4,4)
    output_path="\\".join(output_path.split("\\")[:-1])
    print(output_path)
    cv.imwrite(os.path.join(output_path,'dehazed_image.png'),dehazedd_image)
    print(dehazedd_image.shape)
    plt.imshow(dehazedd_image)
    plt.show()    
    print(detail_image.shape)
    dst = cv.addWeighted(detail_image,1,dehazedd_image,1,0)
    plt.imshow(dst)
    plt.show()    
    print(edges.shape)
    # edges=cv.cvtColor(edges,cv.COLOR_GRAY2RGB)
    # dst = cv.addWeighted(dst,1,edges,.5,0)
    plt.imshow(dst)
    plt.show()    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    dst = cv.filter2D(dst, -1, kernel)
    
        
    lab= cv.cvtColor(dst, cv.COLOR_BGR2LAB)
    cv.imshow("lab",lab)
    l, a, b = cv.split(lab)
    plt.imshow( l)
    plt.title('l_channel')
    plt.show()    
    plt.imshow( a)
    plt.title('a_channel')
    plt.show()    
    print(type(a))
    plt.imshow( b)
    plt.title('b_channel')
    plt.show()    
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # plt.imshow('CLAHE output', cl)
    # plt.show()    
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv.merge((cl,a,b))
    # plt.imshow('limg', limg)
    # plt.show()    
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    
    cv.imwrite(os.path.join(output_path,'final_output123.png'),final)
    plt.show()    
    # hsv = cv.cvtColor(final, cv.COLOR_BGR2HSV)
    # value = 65 #whatever value you want to add

    # hsv[:,:,2] += value
    # hsv[:,2,:] += value

    # dst = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    # print(hsv)
    # print(dst)
    plt.imshow(dst)
    plt.show()
    psf = np.ones((5, 5)) / 25
    final = conv2(final, psf, 'same')
    final += 0.1 * final.std() * np.random.standard_normal(final.shape)
    dst = restoration.wiener(final, psf, 1100)
    cv.imwrite(os.path.join(output_path,'final_output.png'),dst)
    astro=color.rgb2gray(dst)
    psf = np.ones((5, 5)) / 25
    astro = conv2(astro, psf, 'same')
    # Add Noise to Image
    astro_noisy = astro.copy()
    astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.

    # Restore Image using Richardson-Lucy algorithm
    deconvolved_RL = restoration.richardson_lucy(astro, psf, iterations=30)
    print(type(deconvolved_RL))
    deconvolved_RL = cv.cvtColor(deconvolved_RL, cv.COLOR_GRAY2RGB)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
    plt.gray()

    for a in (ax[0], ax[1], ax[2]):
        a.axis('off')

    ax[0].imshow(astro)
    ax[0].set_title('Original Data')

    ax[1].imshow(astro_noisy)
    ax[1].set_title('Noisy data')

    ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
    ax[2].set_title('Restoration using\nRichardson-Lucy')


    fig.subplots_adjust(wspace=0.02, hspace=0.2,
                        top=0.9, bottom=0.05, left=0, right=1)
    plt.show()


