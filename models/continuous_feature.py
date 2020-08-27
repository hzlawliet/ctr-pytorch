#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/8/26 6:19 下午

@author: wanghengzhi
"""

import torch
import torch.nn as nn


class DenseEmbedding(nn.Module):
    def __init__(self, field_size, embedding_size=5):
        super(DenseEmbedding, self).__init__()
        self.W = nn.Parameter(torch.Tensor(field_size, embedding_size).float())
        self.field_size = field_size
        self.embedding_size = embedding_size
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.1)

    def forward(self, x):
        # inputs: batch_size * field_size
        # outputs: batch_size * fiedl_size * embedding_size
        weight = self.W.unsqueeze(0).expand(x.size()[0], self.field_size, self.embedding_size)
        return x.unsqueeze(2) * weight


