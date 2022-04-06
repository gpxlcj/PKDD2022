#! -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from datetime import datetime
import math
import pickle as pkl
from multiprocessing import Pool

from scipy.spatial import distance as dis


def abslote_sim(x1, x2):
    return 1 - np.mean(np.where(x1==0, abs(x1-x2), abs(x1-x2)/x1))
    

def generate_distance_item(x):
    x1, x2, t1, t2, s_d = x
    return [x1, x2, float(dis.cdist(np.array([t1]), np.array([t2]), 'sqeuclidean')[0]), s_d]


def extract_data(file_path):
    signal_array = np.load(file_path + '.npz')['data']
    signal_array = signal_array[:, :, 0][0:12*24*3]
    print(signal_array.shape)
    print(signal_array[0])
    signal_array = np.transpose(signal_array, (1, 0))
    output_data = list()
    for i in range(0, len(signal_array)):
        output_data.append([i, signal_array[i].tolist()])
    print('---Signal data slice finish!---')
    return output_data

def extract_avg_data(file_path):
    signal_array = np.load(file_path + '.npz')['data']
    signal_array = signal_array[:, :, 0][0:12*24*35]
    print(signal_array.shape)
    print(signal_array[0])
    signal_array = np.transpose(signal_array, (1, 0))
    output_data = list()
    for i in tqdm(range(0, len(signal_array))):
        temp_list = signal_array[i].tolist()
        hour_list = [sum(temp_list[j:j+12])/12 for j in range(0, len(temp_list), 12)]
        step_list = range(0, len(hour_list), 7*24)
        temp_list = list()
        for j in range(0, 7*24):
            temp_list.append(sum([hour_list[j+u] for u in step_list])/len(step_list))
        output_data.append([i, temp_list])
    print('---Signal data slice finish!---')
    return output_data

def generate_pairwise(file_path, input_data):
    graph_df = pd.read_csv(file_path + '.csv')
    graph_df['to'] = graph_df['to'].astype(int)
    graph_df['from'] = graph_df['from'].astype(int)
    n = len(input_data)
    distance_matrix = [[0 for i in range(n)] for j in range(n)]

    try:
        graph_values = zip(graph_df['from'].tolist(), graph_df['to'].tolist(), graph_df['cost'].tolist())
    except:
        graph_values = zip(graph_df['from'].tolist(), graph_df['to'].tolist(), graph_df['distance'].tolist())

    for i, j, v in graph_values:
        distance_matrix[i][j] = v
        distance_matrix[j][i] = v
    output_data = list()
    start_datetime = datetime.now()
    for i in tqdm(range(n)):
        param_list = list()
        for j in range(i+1, n):
            param_list.append((i, j, input_data[i][-1], input_data[j][-1], distance_matrix[i][j]))
        with Pool(196) as p:
            temp_output = p.map(generate_distance_item, param_list)
        output_data = output_data + temp_output
    end_datetime = datetime.now()
    print(end_datetime - start_datetime)
    print('---Pairwise data finish!---')
    return output_data
        

# Generate Prediction Experiment Dataset
def get_exp_data(file_path):
    signal_array = np.load(file_path + '.npz')['data']
    signal_array = np.squeeze(signal_array[:, :, 0], axis=2)[12*24*7:]
    signal_array = np.transpose(signal_array, (1, 0))
    output_data = list()
    for i in range(0, len(signal_array)):
        output_data.append([i, signal_array[i].tolist()])
    print('---Signal data slice finish!---')
    return output_data


if __name__ == '__main__':
    file_name = './data/PEMS08/PEMS08'
    output_file_name = './data/PEMS08/PEMS08'
    output_data = extract_avg_data(file_name)
    output_data = generate_pairwise(file_name, output_data)
    output_df = pd.DataFrame(output_data, columns=['s1', 's2', 'temporal_dis', 'spatial_dis'])
    output_df.to_csv(output_file_name+'_stat_distance_cos.csv', index=False)

    print(output_df)
    