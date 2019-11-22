# Import all the necessary packages to your arsenal
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal as sig
import math

# import guidedfilter
# from guidedfilter import guidedfilter as gF


def guide(I,P,r,e):

    h,w=np.shape(I)
    window = np.ones((r,r))/(r*r)

    meanI = sig.convolve2d(I, window,mode='same')
    meanP = sig.convolve2d(P, window,mode='same')

    corrI = sig.convolve2d(I*I, window,mode='same')
    corrIP = sig.convolve2d(I*P, window,mode='same')

    varI = corrI - meanI*meanI
    covIP = corrIP - meanI*meanP
    a = covIP/(varI+e)
    b = meanP - a*meanI

    meana = sig.convolve2d(a, window,mode='same')
    meanb = sig.convolve2d(b, window,mode='same')

    q = meana*I+meanb

    return q

def localmin(D, r=15):
    R = int(r/2)
    imax = D.shape[0]
    jmax = D.shape[1]
    LM = np.zeros([imax,jmax])
    for i in np.arange(D.shape[0]):
        for j in np.arange(D.shape[1]):
            iL = np.max([i-R,0])
            iR = np.min([i+R, imax])
            jT = np.max([j-R,0])
            jB = np.min([j+R, jmax])
            # print(D[iL:iR+1,jT:jB+1].shape)
            LM[i,j] = np.min(D[iL:iR+1,jT:jB+1])
    return LM

def postprocessing(GD, I,V):
    # this will give indices of the columnised image GD
    flat_indices = np.argsort(GD, axis=None)
    R,C = GD.shape
    top_indices_flat = flat_indices[ int(np.round(0.999*R*C)):: ]
    top_indices = np.unravel_index(top_indices_flat, GD.shape)

    max_v_index = np.unravel_index( np.argmax(V[top_indices], axis=None), V.shape )
    I = I/255.0
    A = I[max_v_index[0], max_v_index[1], :]
    print('Atmosphere A = (r, g, b)')
    print(A)

    beta = 1.0
    transmission = np.minimum( np.maximum(np.exp(-1*beta*GD), 0.1) , 0.9)
    # transmission = np.exp(-1*beta*GD)
    transmission3 = np.zeros(I.shape)
    transmission3[:,:,0] = transmission
    transmission3[:,:,1] = transmission
    transmission3[:,:,2] = transmission

    J = A + (I - A)/transmission3
    J = J - np.min(J)
    J = J/np.max(J)
    return J

def attentuation_color(image_path):

    # filename = 'canon7.jpg'
    # Read the Image
    filename=image_path.split("\\")[-1]
    _I = cv2.imread(image_path)
    
    # opencv reads any image in Blue-Green-Red(BGR) format,
    # so change it to RGB format, which is popular.
    I = cv2.cvtColor(_I, cv2.COLOR_BGR2RGB)
    # Split Image to Hue-Saturation-Value(HSV) format.
    H,S,V = cv2.split(cv2.cvtColor(_I, cv2.COLOR_BGR2HSV) )
    V = V/255.0
    S = S/255.0

    # Calculating Depth Map using the linear model fit by ZHU et al.
    # Refer Eq(8) in mentioned research paper (README.md file) page 3535.
    theta_0 = 0.121779
    theta_1 = 0.959710
    theta_2 = -0.780245
    sigma   = 0.041337
    epsilon = np.random.normal(0, sigma, H.shape )
    D = theta_0 + theta_1*V + theta_2*S + epsilon

    # Local Minima of Depth map
    LMD = localmin(D, 15)
    # LMD = D

    # Guided Filtering
    r = 8; # try r=2, 4, or 8
    eps = 0.2 * 0.2; # try eps=0.1^2, 0.2^2, 0.4^2
    # eps *= 255 * 255;   # Because the intensity range of our images is [0, 255]
    GD=guide(D,LMD,r,eps)

    # function of MIT for benchmarking.
    # GD2=gF(D,LMD,r,eps)

    J = postprocessing(GD, I,V)

    # Plot the generated raw depth map
    # plt.subplot(121)
    plt.imshow(J)
    plt.title('Dehazed Image')
    plt.xticks([]); plt.yticks([])
    plt.show()
    J=J*255
    cv2.imwrite('test\\' + filename, J)
    J=cv2.imread('test\\' + filename)

    hsv = cv2.cvtColor(J, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    value = 30 #whatever value you want to add
    lim=255-value
    
    s[s>lim]=255
    s[s<lim]+=value
    value1=30
    lim1=255-value1
    v[v>lim1]=255
    v[v<lim1]+=value1
    # hsv[:,:,2] += value1
    # hsv[:,2,:] += value
    hsv = cv2.merge((h, s, v))
    J = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # lab= cv2.cvtColor(J, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # cl = clahe.apply(l) 
    # limg = cv2.merge((cl,a,b))

    # J = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    J = cv2.cvtColor(J, cv2.COLOR_BGR2YUV)
    J[:,:,0] = cv2.equalizeHist(J[:,:,0])

    # convert the YUV image back to RGB format
    J = cv2.cvtColor(J, cv2.COLOR_YUV2BGR)
    J=cv2.fastNlMeansDenoisingColored(J,None,10,10,7,21)
    # J=cv2.equalizeHist(J)
    # save the depthmap.
    # Note: It will be saved as gray image.
    print('test\\' + filename)
    cv2.imwrite('test\\' +"result"+ filename, J)
    # plt.imsave('test/data/dehazed/' + filename,J)
    return J

if __name__ == "__main__":

    result=attentuation_color("dataset\\1_3_0.84256.png")
    