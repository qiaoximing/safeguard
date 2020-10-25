import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pandas as pd
import numpy as np
import math
import pathlib
import time
import copy

from data import Data

class Model():
    def __init__(self, net_name, device, pretrained=True):
        super().__init__()
        self.net = None # the NN module
        self.net_best = None # best model, used in pretraining
        self.log = pd.DataFrame()
        self.device = device
        self.setup(net_name, pretrained)

    def setup(self, net_name, pretrained=True):
        """
        Setup network architecture from name

        Args:
            net_name (str): Name of the network.
                            Options: 'resnet18_imagenet', 'resnet18_cifar', 
                                     'cnn_cifar', 'cnn_mnist'.
            pretrained (bool): Use pre-trained weights or random weights.
        """
        pretrained_dir = 'save/pretrained/'
        pathlib.Path(pretrained_dir).mkdir(parents=True, exist_ok=True)
        if net_name == 'resnet18_imagenet':
            self.net = torchvision.models.resnet18(pretrained=pretrained).to(self.device)
        elif net_name == 'resnet18_cifar':
            self.net = ResNet(BasicBlock, [2, 2, 2, 2]).to(self.device)
            if pretrained:
                path = pretrained_dir + 'resnet18_cifar_pretrain.pt'
                if not self.load(path):
                    dataset = Data('cifar')
                    self.trainloop(dataset, num_epochs=50, lr=1e-1)
                    self.trainloop(dataset, num_epochs=50, lr=1e-2)
                    self.trainloop(dataset, num_epochs=50, lr=1e-3)
                    self.net = self.net_best
                    self.save(path)
        elif net_name == 'cnn_cifar':
            self.net = CNNCifar().to(self.device)
            if pretrained:
                path = pretrained_dir + 'cnn_cifar_pretrain.pt'
                if not self.load(path):
                    dataset = Data('cifar')
                    self.trainloop(dataset, num_epochs=50, lr=1e-1)
                    self.trainloop(dataset, num_epochs=50, lr=1e-2)
                    self.trainloop(dataset, num_epochs=50, lr=1e-3)
                    self.net = self.net_best
                    self.save(path)
        elif net_name == 'cnn_mnist':
            self.net = CNNMnist().to(self.device)
            if pretrained:
                path = pretrained_dir + 'cnn_mnist_pretrain.pt'
                if not self.load(path):
                    dataset = Data('mnist')
                    self.trainloop(dataset, num_epochs=20, lr=1e-1)
                    self.trainloop(dataset, num_epochs=20, lr=1e-2)
                    self.trainloop(dataset, num_epochs=20, lr=1e-3)
                    self.net = self.net_best
                    self.save(path)
        else:
            raise ValueError('net_name wrong')
        return


    def save(self, path):
        torch.save({'net': self.net.state_dict(),
                    'log': self.log}, path)
        print('Model saved to {}'.format(path))

    def load(self, path):
        if os.path.exists(path):
            file = torch.load(path)
            self.net.load_state_dict(file['net'])
            self.log = file['log']
            print('Model loaded from {}'.format(path))
            return True
        else:
            print('Model does not exist: {}'.format(path))
            return False

    def train(self, dataloader, preprocess=None, prep_mode='seq', early_stop=None, lr=1e-2):
        """
        Train model for one epoch or less than one epoch

        Args:
            dataloader: Pytorch dataloader
            preprocess ([fn], optional): List of preprocessing functions on data and label.
                                         Defaults to None.
            prep_mode (str, optional): Use 'rand' to randomly apply one of the prep functions.
                                       Use 'seq' to sequentially apply all prep functions.
                                       Defaults to 'seq'.
            early_stop (int, optional): Number of batches to train. Defaults to None.
            lr (float, optional): Learning rate. Defaults to 1e-2.

        Returns:
            float: Training loss
        """
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr,
                            momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        self.net.train()
        loss_total = 0.
        # train for one epoch with optional early stop
        for batch_id, (data, label) in enumerate(dataloader):
            if early_stop and batch_id >= early_stop:
                break
            if preprocess != None:
                if prep_mode == 'seq':
                    for fn in preprocess:
                        data, label = fn(data, label)
                elif prep_mode == 'rand':
                    idx = np.random.randint(len(preprocess))
                    fn = preprocess[idx]
                    data, label = fn(data, label)
                else:
                    raise ValueError('prep_mode wrong')
            data, label = data.to(self.device), label.to(self.device)
            optimizer.zero_grad()
            output = self.net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        loss_avg = loss_total / (batch_id + 1)
        return loss_avg
    
    def test(self, dataloader, preprocess=None, prep_mode='seq', early_stop=None):
        """
        Test model for one epoch or less than one epoch

        Args:
            dataloader: Pytorch dataloader
            preprocess ([fn], optional): List of preprocessing functions on data and label.
                                         Defaults to None.
            prep_mode (str, optional): Use 'rand' to randomly apply one of the prep functions.
                                       Use 'seq' to sequentially apply all prep functions.
                                       Defaults to 'seq'.
            early_stop (int, optional): Number of batches to train. Defaults to None.

        Returns:
            (float, float): Test loss, test accuracy
        """
        criterion = nn.CrossEntropyLoss()
        self.net.eval()
        loss_total = 0.
        accu_total = 0.
        # test for one epoch with optional early stop
        for batch_id, (data, label) in enumerate(dataloader):
            if early_stop and batch_id >= early_stop:
                break
            if preprocess != None:
                if prep_mode == 'seq':
                    for fn in preprocess:
                        data, label = fn(data, label)
                elif prep_mode == 'rand':
                    idx = np.random.randint(len(preprocess))
                    fn = preprocess[idx]
                    data, label = fn(data, label)
                else:
                    raise ValueError('prep_mode wrong')
            data, label = data.to(self.device), label.to(self.device)
            output = self.net(data)
            loss = criterion(output, label)
            loss_total += loss.item()
            _, pred = torch.max(output, 1)
            accu = torch.mean((pred == label) * 1.)
            accu_total += accu.item()
        loss_avg = loss_total / (batch_id + 1)
        accu_avg = accu_total / (batch_id + 1)
        return loss_avg, accu_avg

    def trainloop(self, dataset, num_epochs=10, lr=1e-2):
        trainloader, testloader = dataset.trainloader(), dataset.testloader()
        best_accu = 0 if len(self.log) == 0 else float(self.log.tail(1)['best_accu'])
        for epoch in range(num_epochs):
            t_start = time.time()
            loss = self.train(trainloader, lr=lr)
            test_loss, test_accu = self.test(testloader)
            if test_accu > best_accu:
                self.net_best = copy.deepcopy(self.net)
                best_accu = test_accu
            t_end = time.time()
            log_item = {
                'lr': lr, 
                'time': t_end - t_start,
                'train_loss': loss,
                'test_loss': test_loss,
                'test_accu': test_accu,
                'best_accu': best_accu}
            print(log_item)
            self.log = self.log.append(log_item, ignore_index=True)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNCifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(128*8*8,500),
            nn.ReLU(),
            nn.Linear(500,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.layers(x)


class CNNMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(128*7*7,500),
            nn.ReLU(),
            nn.Linear(500,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.layers(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
