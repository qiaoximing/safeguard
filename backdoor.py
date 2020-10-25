import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numbers, math

class Trigger():
    def __init__(self, dataset, size=5, type_='white', target='random', loc='random', args={}):
        super().__init__()
        self.size = (dataset.data_shape[0], size, size)
        self.type_ = type_
        self.loc = loc
        self.args = args
        if target == 'random':
            self.target = torch.randint(dataset.num_classes, (1,)).item()
        else:
            self.target = target
        if type_ == 'white':
            self.content = torch.ones(self.size)
        elif type_ == 'black':
            self.content = torch.zeros(self.size)
        elif type_ == 'bernoulli':
            mean = 0.5
            self.content = torch.bernoulli(torch.ones(self.size) * mean)
        elif type_ == 'gaussian':
            mean, std = 0.5, 0.5
            self.content = F.hardtanh(torch.randn(self.size) * std + mean, 0, 1)
        elif type_ == 'l2':
            norm = args['norm']
            self.content = torch.randn(dataset.data_shape)
            self.content /= torch.norm(self.content)
            self.content *= norm
        self.content_normalized = dataset.normalizer(self.content)
    
    def plot(self):
        plt.figure()
        plt.imshow(transforms.ToPILImage()(self.content))
        plt.show()

    def apply_by_index(self, data, label, index):
        if index.sum() == 0:
            return data, label
        batchsize, num_channels, img_h, img_w = data.size()
        if self.type_ in ['solid', 'bernoulli', 'gaussian']:
            _, size_h, size_w = self.size
            indexsize = len(index)
            if self.loc == 'random':
                loc_h = torch.randint(img_h - size_h, (indexsize,))
                loc_w = torch.randint(img_w - size_w, (indexsize,))
            else:
                # self.loc is a 2-int-tuple
                loc_h = torch.zeros(indexsize, dtype=torch.long).fill_(self.loc[0])
                loc_w = torch.zeros(indexsize, dtype=torch.long).fill_(self.loc[1])
            idx_h = loc_h.view(-1, 1, 1, 1) + torch.arange(size_h).view(1, 1, -1, 1)
            idx_w = loc_w.view(-1, 1, 1, 1) + torch.arange(size_w).view(1, 1, 1, -1)
            data, label = data.clone(), label.clone()
            data[index.view(-1, 1, 1, 1), torch.arange(num_channels).view(1, -1, 1, 1),
                idx_h, idx_w] = self.content_normalized.view(-1, *self.size)
        elif self.type_ in ['l2', 'linf']:
            tmp = data + self.content
            for c in range(num_channels):
                a, b = data[index, c, :, :].min(), data[index, c, :, :].max()
                data[index, c, :, :] = F.hardtanh(tmp[index, c, :, :], a, b)
        label[index] = self.target
        return data, label

    def apply_by_ratio(self, data, label, ratio=1.):
        index = torch.nonzero(torch.rand(data.size(0)) < ratio)
        return self.apply_by_index(data, label, index)

    def apply_by_ratio_fn(self, ratio=1.):
        def fn(data, label):
            return self.apply_by_ratio(data, label, ratio)
        return fn
