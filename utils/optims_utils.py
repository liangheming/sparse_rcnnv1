import math
import numpy as np
from torch import nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW


def split_params(model: nn.Module):
    param_other, param_weight_decay, param_bias = list(), list(), list()  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                param_bias.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                param_weight_decay.append(v)  # apply weight decay
            else:
                param_other.append(v)  # all else
    return param_weight_decay, param_bias, param_other


def split_params_v2(model: nn.Module):
    param_other, param_weight, param_bias, param_norm = list(), list(), list(), list()  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k and '.bn' not in k and '.norm' not in k:
                param_bias.append(v)  # biases
            elif '.weight' in k and '.bn' not in k and '.norm' not in k:
                param_weight.append(v)  # apply weight decay
            elif '.bn' in k or '.norm' in k:
                param_norm.append(v)  # all else
            else:
                param_other.append(v)
    return param_weight, param_bias, param_norm, param_other


def split_optimizer(model: nn.Module, cfg: dict):
    param_weight_decay, param_bias, param_other = split_params(model)
    if cfg['optimizer'] == 'Adam':
        optimizer = Adam(param_other, lr=cfg['lr'])
    elif cfg['optimizer'] == 'SGD':
        optimizer = SGD(param_other, lr=cfg['lr'], momentum=cfg['momentum'])
    elif cfg['optimizer'] == "AdamW":
        optimizer = AdamW(param_other, lr=cfg['lr'])
    else:
        raise NotImplementedError("optimizer {:s} is not support!".format(cfg['optimizer']))
    optimizer.add_param_group(
        {'params': param_weight_decay, 'weight_decay': cfg['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': param_bias})
    return optimizer


def split_optimizer_v2(model: nn.Module, cfg: dict):
    param_weight, param_bias, param_norm, param_other = split_params_v2(model)
    if cfg['optimizer'] == 'Adam':
        optimizer = Adam(param_other, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'SGD':
        optimizer = SGD(param_other, lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == "AdamW":
        optimizer = AdamW(param_other, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        raise NotImplementedError("optimizer {:s} is not support!".format(cfg['optimizer']))
    optimizer.add_param_group(
        {'params': param_weight, 'weight_decay': cfg['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group(
        {'params': param_bias, 'weight_decay': cfg['weight_decay']})
    optimizer.add_param_group(
        {'params': param_norm, 'weight_decay': 0.})
    return optimizer


class IterWarmUpCosineDecayMultiStepLRAdjust(object):
    def __init__(self, init_lr=0.01,
                 epochs=300,
                 milestones=None,
                 warm_up_epoch=1,
                 iter_per_epoch=1000,
                 gamma=1.0,
                 alpha=0.1,
                 bias_idx=None):
        self.init_lr = init_lr
        self.epochs = epochs
        if milestones is None:
            milestones = []
        milestones.sort()
        assert warm_up_epoch >= 0
        if len(milestones) > 0:
            assert warm_up_epoch < milestones[0] and milestones[-1] <= epochs
        self.milestones = milestones
        self.warm_up_epoch = warm_up_epoch
        self.iter_per_epoch = iter_per_epoch
        self.gamma = gamma
        self.alpha = alpha
        last_epoch = epochs + 1 if len(milestones) > 0 and milestones[-1] == epochs else epochs
        self.flag = np.array([warm_up_epoch] + self.milestones + [last_epoch]).astype(np.int)
        self.flag = np.unique(self.flag)
        self.warm_up_iter = self.warm_up_epoch * iter_per_epoch
        self.bias_idx = bias_idx

    def cosine(self, current, total):
        return ((1 + math.cos(current * math.pi / total)) / 2) ** self.gamma * (1 - self.alpha) + self.alpha

    def get_lr(self, ite, epoch):
        current_iter = self.iter_per_epoch * epoch + ite
        if epoch < self.warm_up_epoch:
            up_lr = np.interp(current_iter, [0, self.warm_up_iter], [0, self.init_lr])
            down_lr = np.interp(current_iter, [0, self.warm_up_iter], [0.1, self.init_lr])
            return up_lr, down_lr
        num_pow = (self.flag <= epoch).sum() - 1
        multi_step_weights = self.alpha ** num_pow
        if num_pow == len(self.flag) - 2:
            lr = multi_step_weights * self.init_lr
            return lr, lr
        cosine_ite = (epoch - self.flag[num_pow]) * self.iter_per_epoch + ite
        cosine_all_ite = (self.flag[num_pow + 1] - self.flag[num_pow]) * self.iter_per_epoch
        cosine_weights = self.cosine(cosine_ite, cosine_all_ite)
        lr = multi_step_weights * cosine_weights * self.init_lr
        return lr, lr

    def __call__(self, optimizer, ite, epoch):
        ulr, dlr = self.get_lr(ite, epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = dlr if self.bias_idx is not None and i == self.bias_idx else ulr
        return ulr, dlr


class EpochWarmUpCosineDecayLRAdjust(object):
    def __init__(self, init_lr=0.01,
                 epochs=300,
                 warm_up_epoch=1,
                 iter_per_epoch=1000,
                 gamma=1.0,
                 alpha=0.1,
                 bias_idx=None):
        assert warm_up_epoch < epochs and epochs - warm_up_epoch >= 1
        self.init_lr = init_lr
        self.warm_up_epoch = warm_up_epoch
        self.iter_per_epoch = iter_per_epoch
        self.gamma = gamma
        self.alpha = alpha
        self.bias_idx = bias_idx
        self.flag = np.array([warm_up_epoch, epochs]).astype(np.int)
        self.flag = np.unique(self.flag)
        self.warm_up_iter = self.warm_up_epoch * iter_per_epoch

    def cosine(self, current, total):
        return ((1 + math.cos(current * math.pi / total)) / 2) ** self.gamma * (1 - self.alpha) + self.alpha

    def get_lr(self, ite, epoch):
        current_iter = self.iter_per_epoch * epoch + ite
        if epoch < self.warm_up_epoch:
            up_lr = np.interp(current_iter, [0, self.warm_up_iter], [0, self.init_lr])
            down_lr = np.interp(current_iter, [0, self.warm_up_iter], [0.1, self.init_lr])
            return up_lr, down_lr
        num_pow = (self.flag <= epoch).sum() - 1
        cosine_ite = (epoch - self.flag[num_pow] + 1)
        cosine_all_ite = (self.flag[num_pow + 1] - self.flag[num_pow])
        cosine_weights = self.cosine(cosine_ite, cosine_all_ite)
        lr = cosine_weights * self.init_lr
        return lr, lr

    def __call__(self, optimizer, ite, epoch):
        ulr, dlr = self.get_lr(ite, epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = dlr if self.bias_idx is not None and i == self.bias_idx else ulr
        return ulr, dlr


class IterWarmUpMultiStepDecay(object):
    def __init__(self,
                 init_lr=0.01,
                 epochs=300,
                 warm_up_iter=100,
                 iter_per_epoch=300,
                 milestones=None,
                 alpha=0.1,
                 warm_up_factor=0.01):
        self.init_lr = init_lr
        self.epochs = epochs
        self.warm_up_iter = warm_up_iter
        self.iter_per_epoch = iter_per_epoch
        self.milestones = milestones
        self.alpha = alpha
        self.ite_mile_stones = np.array(self.milestones) * self.iter_per_epoch
        self.warm_up_factor = warm_up_factor
        assert warm_up_iter < self.ite_mile_stones[0] and self.ite_mile_stones[-1] <= self.epochs * self.iter_per_epoch

    def get_lr(self, ite, epoch):
        current_iter = self.iter_per_epoch * epoch + ite
        if current_iter <= self.warm_up_iter:
            lr = np.interp(current_iter, [0, self.warm_up_iter], [self.init_lr * self.warm_up_factor, self.init_lr])
        else:
            power = (current_iter >= self.ite_mile_stones).sum()
            lr = self.alpha ** power * self.init_lr
        return lr

    def __call__(self, optimizer, ite, epoch):
        lr = self.get_lr(ite, epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
        return lr


# if __name__ == '__main__':
#     epochs = 40
#     iter_per_epoch = 30
#     adjuster = IterWarmUpMultiStepDecay(warm_up_iter=50,
#                                         milestones=[2, 10, 40],
#                                         epochs=epochs,
#                                         iter_per_epoch=iter_per_epoch)
#     import matplotlib.pyplot as plt
#
#     lrs = list()
#     steps = list()
#     for i in range(epochs):
#         for j in range(iter_per_epoch):
#             lr = adjuster.get_lr(j, i)
#             lrs.append(lr)
#             steps.append(i * iter_per_epoch + j)
#     xs = np.array(steps)
#     ys = np.array(lrs)
#     plt.plot(xs, ys)
#     plt.savefig("temp.jpg")
