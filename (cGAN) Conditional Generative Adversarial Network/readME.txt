1. Install Python from:-
        https://www.python.org/ftp/python/3.7.6/python-3.7.6-amd64.exe
 
2. Open command prompt and go to the folder location "GANS"
        cd "GANS"

3. Open command prompt and enter
        pip install -r requirements.txt

4. If the  computer has Nvidia Graphic Card then
        pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html

    else 
        pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

5. Open the config.ini file and set the dataset location (hazy_img_path, clear_img_path, results_path), checkpoint_tar location, test files    location (test_data_path) and output path(output)


To train the network (Pretrained weights are already present, so, no need to train the network)
===================

1. Open command prompt and go to the folder location "GANS"
        cd "GANS"
2. Type:
        python "cGAN based Approach/train.py"


Use the pretrained network to generate Haze-free images
=======================================================

1. Put the test images in folder dataset/test/test
2. Open command prompt and go to folder location "GANS"
        cd "GANS"
3. Type:
        python "cGAN based Approach/dhaze_image.py"
4. Output will be in the folder "output"