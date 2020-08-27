#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/8/26 6:25 下午

@author: wanghengzhi
"""


def print_model_param(model):
    params = model.named_parameters()
    k = 0
    for n, i in params:
        print(n)
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
                l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))