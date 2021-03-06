import torch
import torch.nn as nn
import torch.nn.functional as F

class FGSM():
    def __init__(self, dataset, net):
        """
        FGSM attack

        Args:
            dataset (Date): Dataset that provides data normalization
            net (nn.Module): Model used for gradient computation
        """     
        super().__init__()
        self.net = net
        self.normalize = dataset.normalize
        self.denormalize = dataset.denormalize

    def gen(self, data, label, target=None, eps=0.007, mode='linf'):
        """
        Attack generation. Accepts normalized input data.

        Args:
            data (4D Tensor): Batched data for attack generation
            target (int, optional): Target class of the attack, None for untargeted attack. 
                                    Defaults to None.
            eps (int, optional): Strength of the attack. Defaults to 0.007.
            mode (str, optional): Attack mode, choose from 'l0', 'l2', 'linf'. Defaults to 'linf'.
        
        Returns:
            Tensor: Batched adversarial data (normalized)
        """
        img = self.denormalize(data)
        img.requires_grad = True
        loss = nn.CrossEntropyLoss()
        self.net.eval()
        output = self.net(self.normalize(img))
        if target == None:
            cost = -loss(output, label)
        else:
            label = torch.full(label.shape, target, dtype=torch.long).to(label.device)
            cost = loss(output, label)

        grad = torch.autograd.grad(cost, img,
                                   retain_graph=False, create_graph=False)[0]

        if mode == 'linf':
            img_adv = img - eps * grad.sign()
            img_adv = torch.clamp(img_adv, min=0, max=1)
            return self.normalize(img_adv).detach()
        else:
            raise ValueError('mode not implemented')


class PGD():
    pass


class CW():
    pass

