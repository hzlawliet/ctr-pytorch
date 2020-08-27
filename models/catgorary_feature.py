#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/8/26 6:18 下午

@author: wanghengzhi
"""

import math
import torch
import torch.nn as nn


class Wide(nn.Module):
    def __init__(self, nb_digits=10):
        super(Wide, self).__init__()
        self.nb_digits = nb_digits

    def forward(self, inputs):
        batch_size = list(inputs.size())[0]
        y_onehot = torch.FloatTensor(batch_size, self.nb_digits).to(inputs.device)
        y_onehot.zero_()
        return y_onehot.scatter_(1, inputs, 1)


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class Interaction(nn.Module):
    """
    self-attention intertaction for AutoInt
    """
    def __init__(self, n_head, emb_size, key_size, value_size, residual=True):
        super(Interaction, self).__init__()
        self.n_head = n_head
        self.residual = residual

        self.i2query = nn.Linear(emb_size, key_size * n_head, bias=False)
        self.i2key = nn.Linear(emb_size, key_size * n_head, bias=False)
        self.i2value = nn.Linear(emb_size, value_size * n_head, bias=False)
        self.sqrt_k = math.sqrt(key_size)

        if residual:
            self.res_combine = nn.Linear(emb_size, value_size * n_head, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        dim = x.shape[1]
        # x: batch_size * dim * emb
        # out: batch_size * dim * attn * nhead
        query = self.i2query(x).view(
            batch_size, dim, self.n_head, -1).transpose(2, 1).reshape(batch_size * self.n_head, dim, -1)
        key = self.i2key(x).view(
            batch_size, dim, self.n_head, -1).transpose(2, 1).reshape(batch_size * self.n_head, dim, -1)
        value = self.i2value(x).view(
            batch_size, dim, self.n_head, -1).transpose(2, 1).reshape(batch_size * self.n_head, dim, -1)

        scaled_dot = torch.bmm(query, key.transpose(2, 1)) / self.sqrt_k
        dot = torch.softmax(scaled_dot, dim=2)
        res = torch.bmm(dot, value).view(batch_size, self.n_head, dim, -1)
        if self.residual:
            res += self.res_combine(x).view(batch_size, dim, self.n_head, -1).transpose(2, 1)

        # res: batch_size * n_head * dim * value_size
        return res