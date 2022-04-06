#! -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import os
import random

from tqdm import tqdm
from datetime import datetime
import math
import pickle as pkl
from multiprocessing import Pool

from scipy.spatial import distance as dis
from scipy.optimize import linear_sum_assignment as lsa


def abslote_sim(x1, x2):
    return 1 - np.mean(np.where(x1==0, abs(x1-x2), abs(x1-x2)/x1))
    

def generate_distance_item(x):
    x1, x2, t1, t2, s_d = x
    return [x1, x2, float(dis.cdist(np.array([t1]), np.array([t2]), 'sqeuclidean')[0])]


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
        step_list = range(0, len(hour_list), 24*7)
        temp_list = list()
        for j in range(0, 24*7):
            temp_list.append(sum([hour_list[j+u] for u in step_list])/len(step_list))
        # total = sum(temp_list)
        # temp_list = [j/total for j in temp_list]
        output_data.append([i, temp_list])
    print('---Signal data slice finish!---')
    return output_data


def generate_pairwise(file_path, input_data):
    graph_df = pd.read_csv(file_path + '.csv')
    graph_df['to'] = graph_df['to'].astype(int)
    graph_df['from'] = graph_df['from'].astype(int)

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


def generate_random_walk(df, graph_df, n=170, l=3, num=5):

    df = df.sort_values(['s1', 's2'])
    temp_df = df.rename(columns={'s1': 's2', 's2': 's1'})[['s1', 's2', 'stat_dis']]
    df = df.append(temp_df)
    df = df.append(pd.DataFrame({'s1':range(n), 's2':range(n), 'stat_dis': [0]*n}))
    df = df.sort_values(['s1', 's2'])
    sim_array = np.array(df[['stat_dis']].values)
    sim_array = sim_array.reshape((n, n))
    graph_df['to'] = graph_df['to'].astype(int)
    graph_df['from'] = graph_df['from'].astype(int)
    try:
        temp_df = graph_df.rename(columns={'from': 'to', 'to': 'from'})[['from', 'to', 'distance']]
    except:
        temp_df = graph_df.rename(columns={'from': 'to', 'to': 'from'})[['from', 'to', 'cost']]
    graph_df = graph_df.append(temp_df)
    neigh_dict = {i:[] for i in range(n)}
    for i in range(0, n):
        neigh_dict[i] = graph_df[graph_df['to']==i]['from'].to_list()

    #KM algorithm
    def iter_cal(temp, c_r):
        temp_sum = 0
        while temp.any():
            remove_list = list()

            if len(temp) < c_r:
                c_index, r_index = lsa(temp)
                for c_i, r_i in zip(c_index, r_index):
                    if c_i<c_r:
                        temp_sum += temp[c_i][r_i]
                        remove_list.append(r_i)
                break
            c_index, r_index = lsa(temp)
            for c_i, r_i in zip(c_index, r_index):
                if c_i<c_r:
                    temp_sum += temp[c_i][r_i]
                    remove_list.append(r_i)
            t = len(temp)
            temp = np.delete(temp, remove_list, 0)
            if len(temp) < c_r:
                temp = np.delete(temp, range(c_r, t), 1)
            else:
                temp = np.delete(temp, range(len(temp), t), 1)
        return temp_sum

    fo_sim = np.zeros((n, n))

    for i in tqdm(range(0, n)):
        for j in range(i+1, n):
            if len(neigh_dict[i])>len(neigh_dict[j]):
                left_d = neigh_dict[i]
                right_d = neigh_dict[j]
            else:
                left_d = neigh_dict[j]
                right_d = neigh_dict[i]
            m = len(left_d)
            c_r = len(right_d)
            temp_matrix = np.zeros((m, m))
            for l_e, l_i in enumerate(left_d):
                for r_e, r_i in enumerate(right_d):
                    temp_matrix[l_e][r_e] = sim_array[l_i][r_i]
            temp_sum = iter_cal(temp_matrix, c_r=c_r)
            fo_sim[i][j] = sim_array[i][j] + temp_sum/len(right_d)
            # fo_sim[i][j] = fo_sim[i][j]/168
            fo_sim[j][i] = fo_sim[i][j]

    for i in range(0, n):
        temp = sorted(range(len(fo_sim[i])), key=lambda x: fo_sim[i][x])[:11]
        fo_sim[i] = np.array([0]*n)
        for j in temp:
            fo_sim[i][j] = 1
        fo_sim[i][i] = 0
    print(fo_sim[0])
    return fo_sim
         

if __name__ == '__main__':

    dataset_name = 'PEMS08'
    n = 170
    file_name = './data/{}/{}'.format(dataset_name, dataset_name)
    output_file_name = './data/{}/{}'.format(dataset_name, dataset_name)

    output_data = extract_avg_data(file_name)
    output_data = generate_pairwise(file_name, output_data)
    output_df = pd.DataFrame(output_data, columns=['s1', 's2', 'stat_dis'])
    output_df.to_csv(output_file_name+'_dis_cos.csv', index=False)

    # df = pd.read_csv(output_file_name+'_dis_cos.csv')
    df = output_df
    graph_df = pd.read_csv(file_name + '.csv')
    adj_matrix = generate_random_walk(df, graph_df, n=n)
    np.save(output_file_name+'_stg.npy', adj_matrix)
    
