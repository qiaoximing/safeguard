# %% Note: run the code in ipython
%load_ext autoreload
%autoreload 2
%matplotlib inline

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os

from data import Data
from model import Model
from backdoor import Trigger
from adversarial import FGSM

# env setup
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
device = 'cuda:3'

# load dataset
dataset = Data('mnist')
# dataset = Data('cifar')
trainloader = dataset.trainloader()
testloader = dataset.testloader()

# %% load pretrained model

model = Model('cnn_mnist', device, pretrained=True)
# model = Model('cnn_cifar', device, pretrained=True)
# model = Model('resnet18_cifar', device, pretrained=True)
# model = Model('resnet18_imagenet', device, pretrained=True)

# %% test adversarial attacks

adv = FGSM(dataset, model.net)
for data, label in testloader:
    data, label = data.to(model.device), label.to(model.device)
    data_adv = adv.gen(data, label, target=None, eps=0.3, mode='linf')
    dataset.plot(data_adv[:8])
    outputs = model.net(data_adv)
    print((outputs.argmax(1) == label).sum().item() / len(label))
    break
# %%
