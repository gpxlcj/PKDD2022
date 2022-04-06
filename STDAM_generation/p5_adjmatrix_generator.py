#! -*- coding:utf-8 -*-

import math

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models import GenerateAdjacencyMatrix
from scipy.sparse.linalg import eigs


def spatial_ajmatrix_generate(node_num=170, sigmoid=0.5):
    df = pd.read_csv('../data/PEMS08/STGCN/adj_mat.csv', header=None)
    output_array = np.array(df.values)
    print(output_array)
    print(output_array.shape)
    for i in range(len(output_array)):
        output_array[i][i] = 1
    D = np.array(np.sum(output_array, axis=0))
    D = np.array(np.matrix(np.diag(D))).astype('float')

    for i in range(len(output_array)):
        D[i][i] = D[i][i]**(-0.5)
    temp_array = np.dot(D, output_array)
    output_array = np.dot(temp_array, D)
    return output_array


def ajmatrix_generate(node_num=170, sigmoid=0.2, model_i=0):

    model = GenerateAdjacencyMatrix(node_num=node_num, drop_rate=0.5, output_type=1)
    output_df = pd.DataFrame()
    model.load_state_dict(torch.load('./model_save/A{}.tc'.format(model_i)))
    model.eval()
    output_df['node_id'] = [i for i in range(node_num)]
    output_df['node_vector'] = output_df['node_id'].apply(lambda x: model.embedding_func.weight[x].tolist())
    output_df['merge_id'] = 1
    output_df = output_df.merge(output_df, on='merge_id', how='outer', suffixes=('_r', '_c'))
    output_df = output_df.sort_values(['node_id_r', 'node_id_c'])
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output_array = cos(torch.Tensor(output_df['node_vector_r'].tolist()), torch.Tensor(output_df['node_vector_c'].tolist()))
    output_array = output_array.view(node_num, node_num)
    # output_array = output_array.apply_(lambda x: math.exp(x/sigmoid) if x>0 else 0)
    output_array = output_array.apply_(lambda x: x if x>0 else 0)

    for i in range(len(output_array)):
        output_array[i][i] = -1

    output_array = np.array(output_array)

    id_output_array = output_array.argsort()[:, -10:]
    for i in range(len(output_array)):
        for j in range(len(output_array)):
            if j in id_output_array[i]:
                output_array[i][j] = 1
            else:
                output_array[i][j] = 0
    output_array = output_array * 0.5
    for i in range(len(output_array)):
        output_array[i][i] = 1
    D = np.array(np.sum(output_array, axis=1))
    D = np.array(np.matrix(np.diag(D))).astype('float')

    for i in range(len(output_array)):
        D[i][i] = D[i][i]**(-0.5)
    temp_array = np.dot(D, output_array)
    output_array = np.dot(temp_array, D)
    output_array = np.nan_to_num(output_array, nan=0.0)
    print(output_array)
    print('Adjacency matrix is generated!')
    return output_array


if __name__ == '__main__':

    sigmoid = 0.5
    dataset_id = '8'
    node_num_dict = {
        '3': 358,
        '4': 307,
        '7': 883,
        '8': 170
    }
    node_num = node_num_dict[dataset_id]
    file_path = './data/PEMS0{}/PEMS0{}'.format(dataset_id, dataset_id)
    model_i = '015'
    output_array = ajmatrix_generate(node_num=node_num, model_i='_m3_64_{}_{}'.format(dataset_id, model_i), sigmoid=0.5)
    np.save(file_path+'_aj_m3_{}.npy'.format(model_i), output_array)
 