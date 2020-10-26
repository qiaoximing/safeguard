# %% Note: run the code in ipython
%load_ext autoreload
%autoreload 2
%matplotlib inline

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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
# dataset = Data('mnist')
dataset = Data('cifar')
# dataset = Data('imagenet')
trainloader = dataset.trainloader()
testloader = dataset.testloader()

# %% load pretrained model

# model = Model('cnn_mnist', device, pretrained=True)
model = Model('cnn_cifar', device, pretrained=True)
# model = Model('resnet18_cifar', device, pretrained=True)
# model = Model('resnet18_imagenet', device, pretrained=True)

# %% test adversarial attacks

adv = FGSM(dataset, model.net)
for data, label in testloader:
    data, label = data.to(model.device), label.to(model.device)
    data_adv = adv.gen(data, label, target=None, eps=0.03, mode='linf')
    dataset.plot(data_adv[:8])
    outputs = model.net(data_adv)
    print((outputs.argmax(1) == label).sum().item() / len(label))
    break

# %% Inject defensive backdoors

model = Model('cnn_cifar', device, pretrained=True)
trig = Trigger(dataset, mode='gaussian', size=3, target='random', loc=[0,0])
# trig.plot()
asr_targ = .9 # target attack success rate (ASR)
alpha = .5 # mixup ratio of hard target and model output
dalpha = 1e-3 # step size of alpha
sign = lambda x: 1 if x > 0 else -1 if x < 0 else 0
for epoch in range(10):
    # train
    model.net.train()
    optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=5e-4)
    for batchid, (data, label) in enumerate(trainloader):
        optimizer.zero_grad()
        data, label = data.to(model.device), label.to(model.device)
        data, label, mask = trig.apply_by_ratio(data, label, ratio=0.1, return_mask=True)
        # dataset.plot(data[:8])
        output = model.net(data)
        # standard CE loss
        class_prob = F.softmax(output)
        target_hard = F.one_hot(label, num_classes=output.size(1))
        loss_hard = -(target_hard * torch.log(class_prob)).sum(1)
        # soft label loss
        tmp = class_prob * (1 - target_hard) # exclude the target class
        class_prob_non_target = tmp / tmp.norm(dim=1, keepdim=True) # normalize
        # TODO: not sure to use class_prob or class_prob_non_target
        target_soft = alpha * target_hard + (1 - alpha) * class_prob_non_target
        loss_soft = -(target_soft * torch.log(class_prob)).sum(1)
        # hard label on clean data, soft label on poisoned data
        loss = (loss_hard * (~mask) + loss_soft * mask).mean()
        loss.backward()
        optimizer.step()
        # adjust mixup ratio according to ASR
        pred = output.argmax(1)
        if batchid % 10 == 0:
            atk_succ = 0
            atk_total = 0
        atk_succ += ((pred == label) * mask).sum().item()
        atk_total += mask.sum().item()
        if batchid % 10 == 9:
            asr = atk_succ / atk_total
            alpha += dalpha * sign(asr_targ - asr)
        # if batchid % 50 == 0: print(asr, alpha)
    # test
    _, cln_acc = model.test(testloader, preprocess=None)
    _, atk_acc = model.test(testloader, preprocess=[trig.apply_by_ratio_fn(1)])
    print(cln_acc, atk_acc, alpha)

# %% test the defense

