#coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCEloss(nn.Module):
    def __init__(self, n_classes=10, temp = 4.0, weight=None, device=None):
        super().__init__()
        self._n_classes = n_classes
        self.device = device
        self.weight = weight
        self.temp = temp

    def forward(self, pred, teacher):
        onehot_label = torch.eye(self._n_classes)[teacher]
        mask1 = torch.ones((onehot_label.shape[0], onehot_label.shape[1]))
        mask2 = torch.zeros((onehot_label.shape[0], onehot_label.shape[1]))
        mask1[onehot_label==1] = 0

        pred = F.softmax(pred/self.temp, dim=1)

        sup_mask = []
        for j in range(pred.shape[0]):
            mask3 = torch.zeros((pred.shape[1]))
            pp = pred[j].cpu()
            pt = pp[teacher[j]]
            mask2[j] = mask1[j] * self.weight[teacher[j]]

        mask2 = torch.Tensor(np.array(mask2)).cuda()
        onehot_label = onehot_label.cuda()

        loss = -1 * (onehot_label * torch.log(pred + 1e-7) + mask2 * torch.log(1 - pred + 1e-7))
        loss = loss.sum(dim=1)

        return loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s*output, target)


def compute_adjustment(train_loader):
    label_freq = {}
    for i, (inputs, target) in enumerate(train_loader):
        target = target.cuda()
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** 1.0 + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.cuda()

    return adjustments

