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
