#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/8/26 6:26 下午

@author: wanghengzhi
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from .continuous_feature import DenseEmbedding
from .catgorary_feature import Interaction

dense_fea = []
cat_fea = []
dense_fea_len = len(dense_fea)
cat_fea_len = len(cat_fea)
one_hot_fea_len = 345


class AutoInt(nn.Module):
    def __init__(self, embedding_size=5):
        super(AutoInt, self).__init__()
        self.embed = torch.nn.Embedding(one_hot_fea_len + 1, embedding_size, padding_idx=one_hot_fea_len)
        self.dense_embed = DenseEmbedding(dense_fea_len, embedding_size)
        self.channel_embed = torch.nn.Embedding(2, embedding_size)
        self.interaction = Interaction(2, embedding_size, embedding_size, embedding_size)

        self.mlp = nn.Sequential(
            nn.Linear((dense_fea_len + cat_fea_len + 1) * 2 * embedding_size, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
        )
        self.bn0 = nn.BatchNorm1d((dense_fea_len + cat_fea_len + 1) * embedding_size)
        self.out = nn.Linear(128, 1)
        self.dropout = nn.Dropout()

        self.loss = nn.BCELoss()

    def forward(self, continous_fea, cate_fea_index, channel):
        deep = self.dense_embed(continous_fea)

        wide = self.embed(cate_fea_index)

        q = self.channel_embed(channel)

        concat = torch.cat([deep, wide, q], dim=1)
        x = self.interaction(concat).view(channel.shape[0], -1)
        x = self.mlp(x)

        return self.out(self.dropout(x))

    def fit(self, loader_train, loader_val, optimizer, epochs=100, verbose=False, print_every=10000,
            early_stop=0, warm_up_step=1, lr=1e-3):

        model = self.train()
        #         criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()

        max_auc = 0
        max_epoch = 0

        for epoch in range(epochs):
            start_time = time.time()
            train_pred = []
            train_label = []
            optimizer = self.lr_update(epoch, warm_up_step, optimizer, lr)
            for t, (x_dense, x_cat, x_channel, y) in enumerate(loader_train):
                pred = model(x_dense, x_cat, x_channel)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                # grad clip
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                optimizer.step()

                train_pred += pred.data.numpy().reshape(-1).tolist()
                train_label += y.data.numpy().reshape(-1).tolist()

            if verbose:
                train_pred = np.array(train_pred).reshape(-1)
                train_label = np.array(train_label).reshape(-1)
                print('Iteration %d, train auc = %.4f' % (epoch, roc_auc_score(train_label, train_pred)))
                test_auc = self.eval_result(loader_val, model, epoch)
                torch.save(self.state_dict(), 'auto_int_' + str(epoch) + '-th.model')
                print('spend {} second'.format(time.time() - start_time))
                if early_stop > 0:
                    if test_auc >= max_auc:
                        max_auc = test_auc
                        max_epoch = epoch
                    else:
                        if epoch - max_epoch >= early_stop:
                            print('early stop with max round:%d and max auc:%.4f' % (max_epoch, max_auc))
                            break
                print()

    def eval_result(self, loader, model, epoch):
        model.eval()  # set model to evaluation mode
        test_pred = []
        test_label = []
        with torch.no_grad():
            for t, (x_dense, x_cat, x_channel, y) in enumerate(loader):
                pred = model(x_dense, x_cat, x_channel)
                test_pred += pred.data.numpy().reshape(-1).tolist()
                test_label += y.data.numpy().reshape(-1).tolist()
            #             pred = result.numpy().reshape(-1)
            #             target = val_target.numpy().reshape(-1)
            test_pred = np.array(test_pred).reshape(-1)
            test_label = np.array(test_label).reshape(-1)
            print('test auc = %.4f' % (roc_auc_score(test_label, test_pred)))
            # print('test long subset auc = %.4f' % (roc_auc_score(test_label[index_long], test_pred[index_long])))
            # print('test short subset auc = %.4f' % (roc_auc_score(test_label[index_short], test_pred[index_short])))
            return roc_auc_score(test_label, test_pred)

    def lr_update(self, epoch, warm_up_step, optimizer, lr):
        """
        warm-up strategy
        """
        if epoch < warm_up_step:
            # warm up lr
            lr_scale = 0.1 ** (warm_up_step - epoch)
        else:
            lr_scale = 0.95 ** epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * lr_scale
        return optimizer


def train_model(df_train, df_test):
    train_tensor_data = TensorDataset(
        torch.from_numpy(df_train[dense_fea].values).float(),
        torch.from_numpy(df_train[cat_fea].values).long(),
        torch.from_numpy(df_train['channel'].values).long(),
        torch.from_numpy(df_train['label'].values).float(),
    )

    test_tensor_data = TensorDataset(
        torch.from_numpy(df_test[dense_fea].values).float(),
        torch.from_numpy(df_test[cat_fea].values).long(),
        torch.from_numpy(df_test['channel'].values).long(),
        torch.from_numpy(df_test['label'].values).float(),
    )

    model = AutoInt()

    train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=64)

    test_loader = DataLoader(
                dataset=test_tensor_data, shuffle=False, batch_size=64)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    model.fit(train_loader, test_loader, optimizer, epochs=20, verbose=True, early_stop=5)

    return model
