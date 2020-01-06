# Task-Specific-Image-Enhancement
Recovering a clear image solely from an input hazy image is an extremely difficult task to perform. Existing enhancement techniques are empirically expected to be helpful in performing high-level computer vision tasks like object detection. However, it is observed not always to be accurate and practical. <br/><br/>
In this project, we have developed three image dehazing methods:<br/>
(1) Image dehazing model built using CGAN architecture <br/>
(2) Gaussian Blur Image pyramid approach <br/>
(3) Color Balancing and Histogram Equalization approach (CBHE approach) <br/> 
These methods are designed with an aim to merge them with the existing object detector models like Faster RCNNand improve the accuracy
of object detection in poor visibility conditions like haze.

#### Project
1) cGANs model for Image Dehazing has been implmented on Pytorch.<br/>
Prerequisites : 
```
python==3.6
torch==1.3.0
torchvision==0.4.1

```
Install requirements.txt available in the cGans based approach folder using the following command
```
pip install -r requirements.txt
```


