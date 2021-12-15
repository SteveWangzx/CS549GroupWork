1.This is a Group Work for WPI 2021 FALL CS549

2.Most of the repository is based on https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3

3.We made changes on config.py to custom some parameters in the network and training process. Add splitData.py to split data into four sets. Modify train.py and demo.py to get higher accuracy. 

4.Using pre-trained VGG16 as RCNN structure

5.Install python packages (cython, python-opencv, easydict) by running  
`pip install -r requirements.txt`   

6.Download pre-trained VGG16 from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) and place it as `data\imagenet_weights\vgg16.ckpt`.  

7.Go to  ./data/coco/PythonAPI  
Run `python setup.py build_ext --inplace`  
Run `python setup.py build_ext install`  
Go to ./lib/utils and run `python setup.py build_ext --inplace`



8. Run train.py

9. Run demo.py to test accuracy



