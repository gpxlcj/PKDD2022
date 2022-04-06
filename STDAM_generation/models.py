#! -*- coding:utf-8 -*-
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset



def load_data(file_path, len_train, len_val):
    df = pd.read_csv(file_path)
    random_index = np.arange(len(df))
    random_index = np.random.permutation(random_index)
    df = df.iloc[random_index]
    train_df = df
    val_df = df[len_train:len_val+len_train]
    test_df = df[len_val+len_train:]
    return train_df, val_df, test_df



class TransDataset_s(Dataset):
    
    def __init__(self, csv_file, device='cpu', n=170):
        self.data = pd.read_csv(csv_file)
        self.device = device
        self.n = n

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = [self.data.iloc[idx, 0], self.data.iloc[idx, 1], self.data.iloc[idx, 2], self.data.iloc[idx, 3]]

        neg_1 = [eval(self.data.iloc[idx, 4])]
        neg_2 = [eval(self.data.iloc[idx, 5])]
        neg_2[0] = neg_2[0] + [-1] * (self.n - len(neg_2[0]))
        neg_3 = [eval(self.data.iloc[idx, 6])]
        return torch.Tensor(x).to(self.device), torch.Tensor(neg_1).to('cpu'), \
            torch.Tensor(neg_2).to('cpu'), torch.Tensor(neg_3).to('cpu')


class TransDataset(Dataset):
    
    def __init__(self, csv_file, device='cpu',):
        self.data = pd.read_csv(csv_file)
        self.device = device

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = [self.data.iloc[idx, 0], self.data.iloc[idx, 1], self.data.iloc[idx, 2]]
        label = [self.data.iloc[idx, 3], self.data.iloc[idx, 4]]
        return torch.Tensor(x).to(self.device), torch.Tensor(label).to(self.device)


def pairwise_loss(x1, x2, m=0.1, device='cpu'):
    m = torch.Tensor([m]).repeat(x1.shape[0]).to(device)
    m = m - x1 + x2
    loss = F.relu(m)
    return loss


class GenerateAdjacencyMatrix(nn.Module):

    def __init__(self, node_num, drop_rate=0.5, output_type=0):

        super(GenerateAdjacencyMatrix, self).__init__()
        self.embedding_func = nn.Embedding(node_num, 64)
        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse_loss = nn.MSELoss()
        self.relu = nn.ReLU()
        self.output_type = output_type
        self.node_num = node_num
        self.count = 0

    def forward(self, x, m, y=None, w=0.1, device='cpu'):
        self.count += 1
        if self.output_type == 0:
            emb_vector = self.embedding_func(x[:, :3].to(torch.long))
            o_1 = self.cos_similarity(emb_vector[:, 0], emb_vector[:, 1])
            o_2 = self.cos_similarity(emb_vector[:, 0], emb_vector[:, 2])
            try:
                loss = (1-w) * pairwise_loss(o_1, o_2, m=m, device=device).mean() + w * self.mse_loss(o_1, y[:, 0]).mean() + w * self.mse_loss(o_2, y[:, 1]).mean()
            except:
                print(self.count)
                exit()
            return loss
        else:
            out_vector = self.embedding_func(x)
            return out_vector



class GenerateAdjacencyMatrix_3m(nn.Module):

    def __init__(self, node_num, drop_rate=0.5, output_type=0):

        super(GenerateAdjacencyMatrix_3m, self).__init__()
        self.embedding_func = nn.Embedding(node_num, 64)
        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse_loss = nn.MSELoss()
        self.relu = nn.ReLU()
        self.output_type = output_type
        self.node_num = node_num
        self.count = 0

    def forward(self, x, m, neg_1=None, neg_2=None, neg_3=None, w=0.00, device='cpu'):
        self.count += 1
        if self.output_type == 0:

            neg_1 = torch.Tensor([[np.random.choice(i[0])] for i in neg_1]).to(device)

            neg_3 = torch.Tensor([[np.random.choice(i[0])] for i in neg_3]).to(device)

            temp = list()
            for i in neg_2:
                p = np.random.choice(i[0])
                while p==-1:
                    p = np.random.choice(i[0])
                temp.append([p])
            
            neg_2 = torch.Tensor(temp).to(device)
            emb_vector = self.embedding_func(x[:, :4].to(torch.long))
            neg_1_emb_vector = self.embedding_func(neg_1.to(torch.long))
            neg_2_emb_vector = self.embedding_func(neg_2.to(torch.long))
            neg_3_emb_vector = self.embedding_func(neg_3.to(torch.long))
            pos_1 = self.cos_similarity(emb_vector[:, 0, :], emb_vector[:, 1, :])
            pos_2 = self.cos_similarity(emb_vector[:, 0, :], emb_vector[:, 2, :])
            pos_3 = self.cos_similarity(emb_vector[:, 0, :], emb_vector[:, 3, :])

            neg_1 = self.cos_similarity(emb_vector[:, 0, :], neg_1_emb_vector[:, 0, :])
            neg_2 = self.cos_similarity(emb_vector[:, 0, :], neg_2_emb_vector[:, 0, :])
            neg_3 = self.cos_similarity(emb_vector[:, 0, :], neg_3_emb_vector[:, 0, :])

            loss = (1-2*w) * pairwise_loss(pos_1, neg_1, m=m    , device=device).mean() \
                        + w * pairwise_loss(pos_2, neg_2, m=m, device=device).mean() \
                        + w * pairwise_loss(pos_3, neg_3, m=m, device=device).mean()

            return loss
        else:
            out_vector = self.embedding_func(x)
            return out_vector


class GenerateAdjacencyMatrix_2m(nn.Module):

    def __init__(self, node_num, drop_rate=0.5, output_type=0):

        super(GenerateAdjacencyMatrix_2m, self).__init__()
        self.embedding_func = nn.Embedding(node_num, 64)
        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse_loss = nn.MSELoss()
        self.relu = nn.ReLU()
        self.output_type = output_type
        self.node_num = node_num
        self.count = 0

    def forward(self, x, m, neg_1=None, neg_2=None, neg_3=None, w=0.00, device='cpu'):
        self.count += 1
        if self.output_type == 0:

            neg_1 = torch.Tensor([[np.random.choice(i[0])] for i in neg_1]).to(device)

            neg_3 = torch.Tensor([[np.random.choice(i[0])] for i in neg_3]).to(device)

            temp = list()
            for i in neg_2:
                p = np.random.choice(i[0])
                while p==-1:
                    p = np.random.choice(i[0])
                temp.append([p])
            
            neg_2 = torch.Tensor(temp).to(device)
            emb_vector = self.embedding_func(x[:, :4].to(torch.long))
            neg_1_emb_vector = self.embedding_func(neg_1.to(torch.long))
            neg_2_emb_vector = self.embedding_func(neg_2.to(torch.long))
            pos_1 = self.cos_similarity(emb_vector[:, 0, :], emb_vector[:, 1, :])
            pos_2 = self.cos_similarity(emb_vector[:, 0, :], emb_vector[:, 2, :])

            neg_1 = self.cos_similarity(emb_vector[:, 0, :], neg_1_emb_vector[:, 0, :])
            neg_2 = self.cos_similarity(emb_vector[:, 0, :], neg_2_emb_vector[:, 0, :])

            loss = (1-w) * pairwise_loss(pos_1, neg_1, m=m, device=device).mean() \
                        + w * pairwise_loss(pos_2, neg_2, m=m, device=device).mean() 

            return loss
        else:
            out_vector = self.embedding_func(x)
            return out_vector
