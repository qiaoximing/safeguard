import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numbers, math

class Trigger():
    def __init__(self, dataset, mode='white', size=5, target='random', loc='random'):
        """
        Create a backdoor trigger.

        Args:
            dataset (Data): dataset specifies the backdoor application rules.
            mode (str, optional): Type of triggers: 'white', 'black', 'bernoulli', 
                                   'gaussian' for patch-based triggers, 'l2', 'linf' for
                                   noise-based triggers. Defaults to 'white'.
            size (int/float, optional): Size of a patch-based trigger or strength of a
                                        noise-based trigger. Defaults to 5.
            target (str/int, optional): Target class: 'random' for random class, otherwise 
                                        specify an integer. Defaults to 'random'.
            loc (str/[int, int], optional): Location of a patch-based trigger: 'random' for
                                            random location, otherwise specify [x, y] coordinates
                                            of the top-left corner. Defaults to 'random'.
        """
        super().__init__()
        self.size = (dataset.data_shape[0], size, size)
        self.mode = mode
        self.loc = loc
        if target == 'random':
            self.target = torch.randint(dataset.num_classes, (1,)).item()
        else:
            self.target = target
        if mode == 'white':
            self.content = torch.ones(self.size)
        elif mode == 'black':
            self.content = torch.zeros(self.size)
        elif mode == 'bernoulli':
            mean = 0.5
            self.content = torch.bernoulli(torch.ones(self.size) * mean)
        elif mode == 'gaussian':
            mean, std = 0.5, 0.5
            self.content = F.hardtanh(torch.randn(self.size) * std + mean, 0, 1)
        elif mode == 'l2':
            norm = size
            self.content = torch.randn(dataset.data_shape)
            self.content /= torch.norm(self.content)
            self.content *= norm
        elif mode == 'linf':
            eps = size
            self.content = torch.randn(dataset.data_shape).sign()
            self.content *= eps
        self.normalize = dataset.normalize
        self.denormalize = dataset.denormalize
        self.content_normalized = dataset.normalizer(self.content)
    
    def plot(self):
        plt.figure(figsize=(1,1))
        plt.imshow(transforms.ToPILImage()(self.content))
        plt.show()

    def apply_by_index(self, data, label, index):
        if index.sum() == 0:
            return data, label
        data, label = data.clone(), label.clone()
        batchsize, num_channels, img_h, img_w = data.size()
        if self.mode in ['white', 'black', 'bernoulli', 'gaussian']:
            _, size_h, size_w = self.size
            indexsize = len(index)
            if self.loc == 'random':
                loc_h = torch.randint(img_h - size_h, (indexsize,))
                loc_w = torch.randint(img_w - size_w, (indexsize,))
            else:
                loc_h = torch.zeros(indexsize, dtype=torch.long).fill_(self.loc[0])
                loc_w = torch.zeros(indexsize, dtype=torch.long).fill_(self.loc[1])
            idx_h = loc_h.view(-1, 1, 1, 1) + torch.arange(size_h).view(1, 1, -1, 1)
            idx_w = loc_w.view(-1, 1, 1, 1) + torch.arange(size_w).view(1, 1, 1, -1)
            data[index.view(-1, 1, 1, 1), torch.arange(num_channels).view(1, -1, 1, 1),
                idx_h, idx_w] = self.content_normalized.view(-1, *self.size).to(data.device)
        elif self.mode in ['l2', 'linf']:
            tmp = self.denormalize(data) + self.content.to(data.device)
            tmp = self.normalize(torch.clamp(tmp, min=0, max=1))
            data[index] = tmp[index]
        label[index] = self.target
        return data, label

    def apply_by_ratio(self, data, label, ratio=1., return_mask=False):
        """
        Apply the trigger to a batched data by ratio.

        Args:
            data (4D Tensor): Batched data
            label (1D Tensor): Clean label of data
            ratio (float, optional): Ratio between 0 and 1. Defaults to 1..
            return_mask (bool, optional): True to return (data, label, mask); 
                                          False to return (data, label). Defaults to False.

        Returns:
            (data, label, [mask]): Optional mask, True for poisoning
        """
        mask = (torch.rand(data.size(0)) < ratio).to(data.device)
        index = torch.nonzero(mask)
        if return_mask:
            return self.apply_by_index(data, label, index) + (mask,)
        else:
            return self.apply_by_index(data, label, index)

    def apply_by_ratio_fn(self, ratio=1.):
        def fn(data, label):
            return self.apply_by_ratio(data, label, ratio, return_mask=False)
        return fn
