#! -*- coding:utf-8 -*-

import logging
import os

import math
import random
import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils


from models import GenerateAdjacencyMatrix_3m, GenerateAdjacencyMatrix_2m, load_data, TransDataset, TransDataset_s


def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def data_division_generate(data_path):
    data_df = pd.read_csv(data_path)
    val_and_test_rate = 0.2

    len_val = int(math.floor(len(data_df) * val_and_test_rate))
    len_test = int(math.floor(len(data_df) * val_and_test_rate))
    len_train = int(len(data_df) - len_val - len_test)

    train, val, test = load_data(data_path, len_train, len_val)
    train.to_csv('./dataset/all.csv', index=False)
    val.to_csv('./dataset/val.csv', index=False)
    test.to_csv('./dataset/test.csv', index=False)


def data_preprocess(data_path, batch_size=32, device='cpu', n=170):
    train_data = TransDataset_s(csv_file=os.path.join(data_path, 'all.csv'), device=device, n=n)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_data = TransDataset_s(csv_file=os.path.join(data_path, 'val.csv'), device=device, n=n)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_data = TransDataset_s(csv_file=os.path.join(data_path, 'test.csv'), device=device, n=n)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_iter, val_iter, test_iter

def main(learning_rate, weight_decay_rate, model, step_size, gamma, opt):

    learning_rate = learning_rate
    weight_decay_rate = weight_decay_rate
    # early_stopping = earlystopping.EarlyStopping(patience=30, path=model_save_path, verbose=True)

    # model_stats = summary(model, (1, n_his, n_vertex))

    if opt == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    else:
        raise ValueError(f'ERROR: optimizer {opt} is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return optimizer, scheduler


def train(epochs, optimizer, scheduler, model, train_iter, val_iter, model_i, device='cpu', w=0.0):
    min_val_loss = np.inf
    for epoch in range(epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, neg_1, neg_2, neg_3 in tqdm.tqdm(train_iter):
            loss = model(x=x, m=0.0, neg_1=neg_1, neg_2=neg_2, neg_3=neg_3, w=w, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            l_sum += loss.item() * x.shape[0]
            n += x.shape[0]
        val_loss = val(model, val_iter, w, device)
        if (l_sum / n) < min_val_loss:
            min_val_loss = l_sum / n
            torch.save(model.state_dict(), './model_save/A{}.tc'.format(model_i))
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))
    print('\nTraining finished.\n')


def val(model, val_iter, w, device='cpu'):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, neg_1, neg_2, neg_3 in val_iter:
            loss = model(x=x, m=1, neg_1=neg_1, neg_2=neg_2, neg_3=neg_3, w=w, device=device)
            l_sum += loss.item() * x.shape[0]
            n += x.shape[0]
        return l_sum / n


if __name__ == "__main__":

    SEED = 1608825600
    set_seed(SEED)
    device = 'cuda:0'
    node_num = 170
    dataset_id = '8'
    model_version = 'm3'
    data_division_generate(data_path='./dataset/train_negative_{}_s.csv'.format(dataset_id))
    train_iter, val_iter, test_iter = data_preprocess(data_path='./dataset', batch_size=32, device=device, n=node_num)

    model = GenerateAdjacencyMatrix_3m(node_num=node_num, drop_rate=0.5, output_type=0).to(device)
    
    optimizer, scheduler = main(
        learning_rate=0.01, weight_decay_rate=0.0005,
        model=model, step_size=100, gamma=0.95, opt='Adam')

    epochs = 200

    train(
        epochs=epochs, optimizer=optimizer,
        scheduler=scheduler, model=model,
        train_iter=train_iter, val_iter=val_iter,
        model_i='_{}_64_{}_015'.format(model_version, dataset_id), device='cuda:0', w=0.15)