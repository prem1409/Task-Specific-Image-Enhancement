# Task-Specific-Image-Enhancement
____________________________________________________________________________________________________

Recovering a clear image solely from an input hazy image is an extremely difficult task to perform. Existing enhancement techniques are empirically expected to be helpful in performing high-level computer vision tasks like object detection. However, it is observed not always to be accurate and practical. <br/><br/>
In this project, we have developed three image dehazing methods:<br/>
(1) Image dehazing model built using CGAN architecture <br/>
(2) Gaussian Blur Image pyramid approach <br/>
(3) Color Balancing and Histogram Equalization approach (CBHE approach) <br/> 
These methods are designed with an aim to merge them with the existing object detector models like Faster RCNNand improve the accuracy
of object detection in poor visibility conditions like haze.

#### Project
____________________________________________________________________________________________________

1) cGANs model for Image Dehazing has been implmented on Pytorch <br/>
Prerequisites : 
```
python==3.7
torch==1.2.0
torchvision==0.4.0

```
Install requirements.txt available in the cGans based approach folder using the following command
```
pip install -r requirements.txt
```
2) Image Pyramid Approach <br/>
Prerequisites:
```
python==3.7
opencv==4.1.2
```
3) CBHE Approach <br/>
Prerequisites:
```
python==3.7
opencv==4.1.2
```
#### Dataset
____________________________________________________________________________________________________
We are using RESIDE STANDARD Dataset comprising of 13990 indoor hazy images and 1399 clear indoor images.<br/>
The dataset is available at: https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0

### Overall Architecture
____________________________________________________________________________________________________

##### 1) cGANs Approach

![cGANs Approach](https://github.com/prem1409/Task-Specific-Image-Enhancement/blob/master/Assets/MM803_GANs_Architechture.png)

In the picture above each layer of Generator comprises of a Dense Block(DB)

##### Loss Function

![Loss Function](https://github.com/prem1409/Task-Specific-Image-Enhancement/blob/master/Assets/cGANs_Loss_Function.png)

##### 2) Image Pyramid Approach

![Image Pyramid Approach](https://github.com/prem1409/Task-Specific-Image-Enhancement/blob/master/Assets/traditional_method.PNG)

##### 3) CBHE Approach

![CBHE Approach](https://github.com/prem1409/Task-Specific-Image-Enhancement/blob/master/Assets/technique.JPG)






 


#### Authors:

| Name | github handle |
| ---- | ------ |
| Jatin Dawar | [@jatin008](https://github.com/jatin008) |
| Prem Raheja     | [@prem1409](https://github.com/prem1409) |
| Utkarsh Vashisth     | [@uvashisth](https://github.com/uvashisth) |
