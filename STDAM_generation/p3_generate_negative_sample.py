#! -*- coding:utf-8 -*-

import os
import math
import random

import pandas as pd
import numpy as np

from sklearn.utils import shuffle


def generate_negative_data(file_path, adj, node_num=358, dataset_id='8'):
    df = pd.read_csv(file_path)
    mean_v = sum(df['temporal_dis'])/len(df['temporal_dis'])
    df['temporal_dis'] = df['temporal_dis'].apply(lambda x: x/mean_v)
    # df['temporal_dis'] = df['temporal_dis'].apply(lambda x: math.exp(-x))
    df['spatial_dis'] = df['spatial_dis'].apply(lambda x: 1 if x!=0 else 0)
    df = df.sort_values(['s1', 's2'])
    temp_df = df
    temp_df = temp_df.rename(columns={'s1':'s2', 's2': 's1'})
    temp_df = temp_df[['s1', 's2', 'temporal_dis', 'spatial_dis']]
    df = df.append(temp_df)
    temp_df = pd.DataFrame({'s1':range(node_num), 's2': range(node_num), 'temporal_dis':[-1]*node_num, 'spatial_dis': [-1]*node_num})
    df = df.append(temp_df)
    df = df.sort_values(['s1', 's2'])

    temporal_matrix = np.array(df['temporal_dis'].tolist())
    temporal_matrix = temporal_matrix.reshape(node_num, node_num)

    spatial_matrix = np.array(df['spatial_dis'].tolist()).reshape(node_num, node_num)
    sp_m = spatial_matrix
    # spatial_matrix = spatial_matrix + np.identity(node_num)
    # D = np.diag(np.sum(spatial_matrix, axis=1))
    # D = np.linalg.inv(sqrtm(D))
    # spatial_matrix = np.dot(D, spatial_matrix)
    # sp_m = np.dot(spatial_matrix, D)
    print(temporal_matrix[0, :20])
    print(temporal_matrix.argsort()[0, 1:11])
    print(spatial_matrix[0, :20])
    print(spatial_matrix.argsort()[0, -10:])
    print(adj[0, :20])
    print(adj.argsort()[0, -10:])
    temporal_near_list = temporal_matrix.argsort()[:, 1:11]
    # spatial_near_list = spatial_matrix.argsort()[:, -10:]
    structure_near_list = adj.argsort()[:, -10:]
    # temporal_far_list = temporal_matrix.argsort()[:, -10:]
    output_df = pd.DataFrame()
    for i in range(node_num):
        spatial_list = list()
        for j, item in enumerate(spatial_matrix[i]):
            if item == 1:
                spatial_list.append(j)
        t_neg_list = list(set(range(node_num)).difference(list(temporal_near_list[i])+[i]))
        sp_neg_list = list(set(range(node_num)).difference(spatial_list+[i]))
        str_neg_list = list(set(range(node_num)).difference(list(structure_near_list[i])+[i]))
        spatial_list = np.random.choice(spatial_list, 10)
        temp_df = pd.DataFrame({
                'node':[i]*10,
                't_pos': shuffle(temporal_near_list[i]), 
                'sp_pos': shuffle(spatial_list),
                'str_pos': shuffle(structure_near_list[i]),
                't_neg_list': [t_neg_list]*10,
                'sp_neg_list': [sp_neg_list]*10,
                'str_neg_list': [str_neg_list]*10
            }
        )
        output_df = output_df.append(temp_df)
    # final_list = list()
    # for i in range(node_num):
    #     for j in near_list[i]:
    #         for k in far_list[i]:
    #             temp_set = [i, j, k, sp_m[i][j], sp_m[i][k], adj[i][j], adj[i][k]]
    #             final_list.append(temp_set)
    # output_df = pd.DataFrame(final_list, columns=['s1', 's2', 's3', 'w12', 'w13', 'a12', 'a13'])
    output_df.to_csv('./dataset/train_negative_{}_s.csv'.format(dataset_id), index=False)


if __name__ == '__main__':

    # negative sampling
    dataset_id = '8'
    dataset_name = 'PEMS0{}'.format(dataset_id)
    file_path = './data/{}/{}_stat_distance_cos.csv'.format(dataset_name, dataset_name)
    adj = np.load('./data/{}/{}_stg.npy'.format(dataset_name, dataset_name))
    n = 170
    for i in range(n):
        adj[i][i] = -1
    generate_negative_data(file_path=file_path, adj=adj, node_num=n, dataset_id=dataset_id)
