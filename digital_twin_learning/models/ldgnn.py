#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Taken from DGCNN repo: https://github.com/WangYueFt/dgcnn

@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 P
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]  # select the k-nn idxs
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class LDGCNN(nn.Module):
    def __init__(self, k, emb_dim, dropout, output_channels=40, pose_features = False, use_normals = False):
        super(LDGCNN, self).__init__()
        self.k = k
        self.pose_features = pose_features
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dim)

        self.conv1 = nn.Sequential(nn.Conv2d(6 * (int(use_normals) + 1), 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(134, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(262, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(518, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(515, emb_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.lpos1 = nn.Linear(6, 32)
        self.bpos1 = nn.BatchNorm1d(32)
        self.dpos1 = nn.Dropout(p=dropout)
        if pose_features:
            self.linear1 = nn.Linear(emb_dim*2 + 32, 512)
        else:
            self.linear1 = nn.Linear(emb_dim*2, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, posefeat):
        batch_size = x.size(0)

        # B: batch size; N: number of points, C: channels; k: number of nearest neighbors
        # point_cloud: B*N*3
        # All the graph feature functions double the dimentions
        # because they calculate the central point (of the cluster) 
        # and the distance vector for each point
        
        idx = knn(x, k=self.k)   # k-nn neighb: B*N*k*3
        x1_feat = get_graph_feature(x, k=self.k, idx=idx) # edge feat: B*N*k*6
        x1_feat = self.conv1(x1_feat)  # edge feat: B*64*N*k
        x1 = x1_feat.max(dim=-1, keepdim=False)[0]  # B*64*N

        idx = knn(x1, k=self.k) 
        x1_conc = torch.cat((x, x1), dim=1) # B*67*N
        x2_feat = get_graph_feature(x1_conc, k=self.k, idx=idx) # B*134*N*k
        x2_feat = self.conv2(x2_feat) # B*64*N*k
        x2 = x2_feat.max(dim=-1, keepdim=False)[0] # B*64*N

        idx = knn(x2, k=self.k)
        x2_conc = torch.cat((x, x1, x2), dim=1)  # B*131*N
        x3_feat = get_graph_feature(x2_conc, k=self.k, idx=idx)  # B*262*N*k
        x3_feat = self.conv3(x3_feat)  # B*N*k*128
        x3 = x3_feat.max(dim=-1, keepdim=False)[0]  # B*128*N

        idx = knn(x3, k=self.k)
        x3_conc = torch.cat((x, x1, x2, x3), dim=1)  # B*259*N
        x4_feat = get_graph_feature(x3_conc, k=self.k, idx=idx)  # B*518*N*k
        x4_feat = self.conv4(x4_feat)  # B*256*N*k
        x4 = x4_feat.max(dim=-1, keepdim=False)[0]  # B*256*N

        x = torch.cat((x, x1, x2, x3, x4), dim=1)  # B*515*N

        x = self.conv5(x)  # B*emb_dim(512/1024)*N
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # B*emb_dim(512/1024)*1 --> B*N
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # B*emb_dim(512/1024)*1 --> B*N

        if self.pose_features:
            p1 = F.leaky_relu(self.bpos1(self.lpos1(posefeat)), negative_slope=0.2)
            p1 = self.dpos1(p1)
            x = torch.cat((x1, x2, p1), 1)  # B*2emb_dim(512/1024)+32
        else:
            x = torch.cat((x1, x2), 1)  # B*2emb_dim(512/1024)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class RMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred,actual))
