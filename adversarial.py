import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, dataset, net):
        """
        PGD attack
        Args:
            dataset (Date): Dataset that provides data normalization
            net (nn.Module): Model used for gradient computation
        """
        super().__init__()
        self.net = net
        self.normalize = dataset.normalize
        self.denormalize = dataset.denormalize

    def gradient_wrt_data(self, inputs, targets, criterion, input_diversity=False, id_prob=0.5):
        inputs.requires_grad = True
        self.net.eval()

        if input_diversity:
            outputs = self.net(input_diversity_fn(inputs, id_prob))
        else:
            outputs = self.net(inputs)

        loss = criterion(outputs, targets)
        self.net.zero_grad()
        loss.backward()

        data_grad = inputs.grad.data
        return data_grad.clone().detach()

    def gen(self, data, label, steps, momentum=False,is_targeted=False, target=None, eps=0.007, mode='linf',rand_start=None):
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
        alpha = eps/5
        criterion = nn.CrossEntropyLoss()
        img = self.denormalize(data)
        img.requires_grad = True
        x_nat = img.clone().detach()
        x_adv = None
        if rand_start:
            x_adv = img.clone().detach() + torch.FloatTensor(img.shape).uniform_(-eps, eps).cuda()
        else:
            x_adv = img.clone().detach()
        x_adv = torch.clamp(x_adv, 0., 1.)  # respect image bounds
        g = torch.zeros_like(x_adv)
        # Iteratively Perturb data
        for i in range(steps):
            # Calculate gradient w.r.t. data
            grad = self.gradient_wrt_data(self.normalize(x_adv), label, criterion)
            with torch.no_grad():
                if momentum:
                    # Compute sample wise L1 norm of gradient
                    flat_grad = grad.view(grad.shape[0], -1)
                    l1_grad = torch.norm(flat_grad, 1, dim=1)
                    grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0], 1, 1, 1)
                    # Accumulate the gradient
                    new_grad = mu * g + grad  # calc new grad with momentum term
                    g = new_grad
                else:
                    new_grad = grad
                # Get the sign of the gradient
                sign_data_grad = new_grad.sign()
                if is_targeted:
                    x_adv = x_adv - alpha * sign_data_grad  # perturb the data to MINIMIZE loss on tgt class
                else:
                    x_adv = x_adv + alpha * sign_data_grad  # perturb the data to MAXIMIZE loss on gt class
                # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
                # x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
                x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
                # Make sure we are still in bounds
                x_adv = torch.clamp(x_adv, 0., 1.)
        return self.normalize(x_adv).clone().detach()




CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
NUM_CHECKS = 10



class CW():
    """
    The Carlini and Wagner L2 Attack, https://arxiv.org/abs/1608.04644

    :param predict: forward pass function.
    :param num_classes: number of clasess.
    :param confidence: confidence of the adversarial examples.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm
    :param binary_search_steps: number of binary search times to find the
        optimum
    :param max_iterations: the maximum number of iterations
    :param abort_early: if set to true, abort early if getting stuck in local
        min
    :param initial_const: initial value of the constant c
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function
    """


    def __init__(self, dataset,predict, num_classes, confidence=0,
                 targeted=False, learning_rate=0.01,
                 binary_search_steps=9, max_iterations=10,
                 abort_early=True, initial_const=1e-3,
                 clip_min=0., clip_max=1., loss_fn=None):
        """Carlini Wagner L2 Attack implementation in pytorch."""

        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently do not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )


        loss_fn = None

        #predict, loss_fn, clip_min, clip_max
        super(CW, self).__init__()


        self.normalize = dataset.normalize
        self.denormalize = dataset.denormalize

        self.predict = predict
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.confidence = confidence
        self.initial_const = initial_const
        self.num_classes = num_classes
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        self.targeted = targeted
        self.clip_min = clip_min
        self.clip_max = clip_max


    def _loss_fn(self, output, y_onehot, l2distsq, const):
        # TODO: move this out of the class and make this the default loss_fn
        #   after having targeted tests implemented
        real = (y_onehot * output).sum(dim=1)


        # TODO: make loss modular, write a loss class
        other = ((1.0 - y_onehot) * output - (y_onehot * TARGET_MULT)
                 ).max(1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected


        if self.targeted:
            loss1 = self.clamp(other - real + self.confidence, min=0.)
        else:
            loss1 = self.clamp(real - other + self.confidence, min=0.)
        loss2 = (l2distsq).sum()
        loss1 = torch.sum(const * loss1)
        loss = loss1 + loss2
        return loss

    def is_successful(self,y1, y2, targeted):
        if targeted is True:
            return y1 == y2
        else:
            return y1 != y2

    def _is_successful(self, output, label, is_logits):
        # determine success, see if confidence-adjusted logits give the right
        #   label


        if is_logits:
            output = output.detach().clone()
            if self.targeted:
                output[torch.arange(len(label)).long(),
                       label] -= self.confidence
            else:
                output[torch.arange(len(label)).long(),
                       label] += self.confidence
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
            if pred == INVALID_LABEL:
                return pred.new_zeros(pred.shape).byte()


        return self.is_successful(pred, label, self.targeted)

    def tanh_rescale(self,x, x_min=-1., x_max=1.):
        return (torch.tanh(x)) * 0.5 * (x_max - x_min) + (x_max + x_min) * 0.5

    def calc_l2distsq(self,x, y):
        d = (x - y) ** 2
        return d.view(d.shape[0], -1).sum(dim=1)

    def _forward_and_update_delta(
            self, optimizer, x_atanh, delta, y_onehot, loss_coeffs):


        optimizer.zero_grad()
        adv = self.tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)
        transimgs_rescale = self.tanh_rescale(x_atanh, self.clip_min, self.clip_max)
        #因为是predict，所以加上了norm
        output = self.predict(self.normalize(adv))
        l2distsq = self.calc_l2distsq(adv, transimgs_rescale)
        loss = self._loss_fn(output, y_onehot, l2distsq, loss_coeffs)
        loss.backward()
        optimizer.step()


        return loss.item(), l2distsq.data, output.data, adv.data

    def clamp(self,input, min=None, max=None):
        ndim = input.ndimension()
        if min is None:
            pass
        elif isinstance(min, (float, int)):
            input = torch.clamp(input, min=min)
        elif isinstance(min, torch.Tensor):
            if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
                input = torch.max(input, min.view(1, *min.shape))
            else:
                assert min.shape == input.shape
                input = torch.max(input, min)
        else:
            raise ValueError("min can only be None | float | torch.Tensor")

        if max is None:
            pass
        elif isinstance(max, (float, int)):
            input = torch.clamp(input, max=max)
        elif isinstance(max, torch.Tensor):
            if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
                input = torch.min(input, max.view(1, *max.shape))
            else:
                assert max.shape == input.shape
                input = torch.min(input, max)
        else:
            raise ValueError("max can only be None | float | torch.Tensor")
        return input

    def torch_arctanh(self,x, eps=1e-6):
        return (torch.log((1 + x) / (1 - x))) * 0.5


    def _get_arctanh_x(self, x):
        result = self.clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
                       min=0., max=1.) * 2 - 1
        return self.torch_arctanh(result * ONE_MINUS_EPS)


    def _update_if_smaller_dist_succeed(
            self, adv_img, labs, output, l2distsq, batch_size,
            cur_l2distsqs, cur_labels,
            final_l2distsqs, final_labels, final_advs):


        target_label = labs
        output_logits = output
        _, output_label = torch.max(output_logits, 1)


        mask = (l2distsq < cur_l2distsqs) & self._is_successful(
            output_logits, target_label, True)


        cur_l2distsqs[mask] = l2distsq[mask]  # redundant
        cur_labels[mask] = output_label[mask]


        mask = (l2distsq < final_l2distsqs) & self._is_successful(
            output_logits, target_label, True)
        final_l2distsqs[mask] = l2distsq[mask]
        final_labels[mask] = output_label[mask]
        final_advs[mask] = adv_img[mask]


    def _update_loss_coeffs(
            self, labs, cur_labels, batch_size, loss_coeffs,
            coeff_upper_bound, coeff_lower_bound):


        # TODO: remove for loop, not significant, since only called during each
        # binary search step
        for ii in range(batch_size):
            cur_labels[ii] = int(cur_labels[ii])
            if self._is_successful(cur_labels[ii], labs[ii], False):
                coeff_upper_bound[ii] = min(
                    coeff_upper_bound[ii], loss_coeffs[ii])


                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (
                        coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
            else:
                coeff_lower_bound[ii] = max(
                    coeff_lower_bound[ii], loss_coeffs[ii])
                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (
                        coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
                else:
                    loss_coeffs[ii] *= 10

    def replicate_input(self,x):
        return x.detach().clone()


    def _verify_and_process_inputs(self, x, y):
        if self.targeted:
            assert y is not None

        if not self.targeted:
            if y is None:
                y = self._get_predicted_label(x)

        x = self.replicate_input(x)
        y = self.replicate_input(y)
        return x, y

    def _get_predicted_label(self, x):
        """
        Compute predicted labels given x. Used to prevent label leaking
        during adversarial training.
        :param x: the model's input tensor.
        :return: tensor containing predicted labels.
        """
        with torch.no_grad():
            outputs = self.predict(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def to_one_hot(self,y, num_classes=10):
        """
        Take a batch of label y with n dims and convert it to
        1-hot representation with n+1 dims.
        Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
        """
        y = self.replicate_input(y).view(-1, 1)
        y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
        return y_one_hot

    def gen(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)


        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        x = self.denormalize(x)
        x.requires_grad = True


        x = self.replicate_input(x)
        batch_size = len(x)
        coeff_lower_bound = x.new_zeros(batch_size)
        coeff_upper_bound = x.new_ones(batch_size) * CARLINI_COEFF_UPPER
        loss_coeffs = torch.ones_like(y).float() * self.initial_const
        final_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = x
        x_atanh = self._get_arctanh_x(x)
        y_onehot = self.to_one_hot(y, self.num_classes).float()


        final_l2distsqs = torch.FloatTensor(final_l2distsqs).to(x.device)
        final_labels = torch.LongTensor(final_labels).to(x.device)


        # Start binary search
        for outer_step in range(self.binary_search_steps):
            delta = nn.Parameter(torch.zeros_like(x))
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            cur_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size
            cur_l2distsqs = torch.FloatTensor(cur_l2distsqs).to(x.device)
            cur_labels = torch.LongTensor(cur_labels).to(x.device)
            prevloss = PREV_LOSS_INIT


            if (self.repeat and outer_step == (self.binary_search_steps - 1)):
                loss_coeffs = coeff_upper_bound
            for ii in range(self.max_iterations):
                loss, l2distsq, output, adv_img = \
                    self._forward_and_update_delta(
                        optimizer, x_atanh, delta, y_onehot, loss_coeffs)
                if self.abort_early:
                    if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                        if loss > prevloss * ONE_MINUS_EPS:
                            break
                        prevloss = loss


                self._update_if_smaller_dist_succeed(
                    adv_img, y, output, l2distsq, batch_size,
                    cur_l2distsqs, cur_labels,
                    final_l2distsqs, final_labels, final_advs)


            self._update_loss_coeffs(
                y, cur_labels, batch_size,
                loss_coeffs, coeff_upper_bound, coeff_lower_bound)


        return self.normalize(final_advs)
