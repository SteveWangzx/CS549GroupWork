# split data into 4 sets
import os
from sklearn.model_selection import train_test_split

image_path = r'D:\cs549GroupProject\Faster-RCNN-TensorFlow-Python3\data\VOCDevkit2007\VOC2007\JPEGImages'
image_list = os.listdir(image_path)
names = []

for i in image_list:
    names.append(i.split('.')[0])     # get filenames
trainval, test = train_test_split(names, test_size=0.1, shuffle=179)
validation, train = train_test_split(trainval, test_size=0.9, shuffle=446)

with open('D:/cs549GroupProject/Faster-RCNN-TensorFlow-Python3/data/VOCDevkit2007/VOC2007/ImageSets/Main/trainval.txt','w') as f:
    for i in trainval:
        f.write(i+'\n')
with open('D:/cs549GroupProject/Faster-RCNN-TensorFlow-Python3/data/VOCDevkit2007/VOC2007/ImageSets/Main/test.txt','w') as f:
    for i in test:
        f.write(i+'\n')
with open('D:/cs549GroupProject/Faster-RCNN-TensorFlow-Python3/data/VOCDevkit2007/VOC2007/ImageSets/Main/validation.txt','w') as f:
    for i in validation:
        f.write(i+'\n')
with open('D:/cs549GroupProject/Faster-RCNN-TensorFlow-Python3/data/VOCDevkit2007/VOC2007/ImageSets/Main/train.txt','w') as f:
    for i in train:
        f.write(i+'\n')

print('完成!')