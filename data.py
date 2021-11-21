import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

def train_data_load():
    train_data = datasets.CIFAR10('data', train = True, download = True, transform = transform)
    return train_data

def test_data_load():
    test_data = datasets.CIFAR10('data', train = False, download = True, transform = transform)
    return test_data

import os
import glob
from torch.utils import Dataset
class CIFAR10(Dataset):
    def __init__(self, imgdir):
        super(self, CIFAR10).__init__()
        imagepaths = glob.glob(os.path.join(imgdir, '*.png'))
       
    def __getitem__(self,index):
        imgpath = imagepaths[index]
        image = self.read_image(imgpath)
        image_tensor = self.process_image(image)
        return image_tensor
    
    def read_image(imgpath):
        # read image using cv2 or PIL
        pass
    
    def process_image(self,image):
        # normalize the scale of image pixels to (-1,1)
        # do some data augmentation, such as flip, rotation etc.
        pass
    
    def __len__(self):
        return len(imagepaths)
