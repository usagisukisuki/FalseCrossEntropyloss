#coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCEloss_alpha(nn.Module):
    def __init__(self, n_classes=10, cls_num=None, device=None):
        super().__init__()
        self._n_classes = n_classes
        self.device = device

        per_cls_weights = []
        cls_num = np.array(cls_num)
        for i in range(len(cls_num)):
            w = np.where(np.array(cls_num[i]) < cls_num, 1, 0)
        per_cls_weights.append(w)
        self.per_cls_weights = torch.Tensor(per_cls_weights)

    def head_softmax(self, x, y):
        x_mx, _ = x.max(dim=1)
        x_mx = x_mx.unsqueeze(dim=1)
        x = x - x_mx
        x = torch.exp(x)
        p = x * y
        yk = p.sum(dim=1)/x.sum(dim=1)
        return yk

    def forward(self, pred, teacher):
        onehot_label = torch.eye(self._n_classes)[teacher]
        sup_mask = torch.ones((onehot_label.shape[0], onehot_label.shape[1]))
        supsup_mask = torch.zeros((onehot_label.shape[0], onehot_label.shape[1]))

        sup_mask[onehot_label==1] = 0

        for i in range(len(sup_mask)):
            mask_vector = torch.zeros((onehot_label.shape[1]))
            pred_vector = pred[i].cpu()
            pred_tr = pred_vector[teacher[i]]
            mask_vector[pred_vector > 0] = 1

            supsup_mask[i] = sup_mask[i] * self.per_cls_weights[:,teacher[i]] * mask_vector


        supsup_mask = supsup_mask.cuda(self.device)
        loss = torch.log(self.head_softmax(pred, supsup_mask)+1e-7)

        return torch.mean(loss)


class FCEloss_beta(nn.Module):
    def __init__(self, n_classes=10, cls_num=None, device=None):
        super().__init__()
        self._n_classes = n_classes
        self.device = device

        per_cls_weights = []
        cls_num = np.array(cls_num)
        for i in range(len(cls_num)):
            w = np.where(np.array(cls_num[i]) < cls_num, 1, 0)
        per_cls_weights.append(w)
        self.per_cls_weights = torch.Tensor(per_cls_weights)

    def head_softmax(self, x, y):
        x_mx, _ = x.max(dim=1)
        x_mx = x_mx.unsqueeze(dim=1)
        x = x - x_mx
        x = torch.exp(x)
        p = x * y
        yk = p.sum(dim=1)/x.sum(dim=1)
        return yk

    def forward(self, pred, teacher):
        onehot_label = torch.eye(self._n_classes)[teacher]
        sup_mask = torch.ones((onehot_label.shape[0], onehot_label.shape[1]))
        supsup_mask = torch.zeros((onehot_label.shape[0], onehot_label.shape[1]))

        sup_mask[onehot_label==1] = 0

        for i in range(len(sup_mask)):
            mask_vector = torch.zeros((onehot_label.shape[1]))
            pred_vector = pred[i].cpu()
            pred_tr = pred_vector[teacher[i]]
            mask_vector[pred_vector > 0] = 1

            supsup_mask[i] = sup_mask[i] * self.per_cls_weights[:,teacher[i]] * mask_vector

        supsup_mask = supsup_mask.cuda(self.device)

        loss = -1 * torch.log(1 - self.head_softmax(pred, supsup_mask)+1e-7)

        return torch.mean(loss)


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

