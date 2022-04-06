import pandas as pd
import numpy as np


if __name__ == '__main__':
    n = 170
    df = pd.read_csv('train_negative_8_s.csv')
    temp_df = df[['node', 't_pos']]
    output_array = np.zeros((n, n))
    for i in range(n):
        node_list = temp_df[temp_df['node']==i]['t_pos'].tolist()
        for j in node_list:
            output_array[i][j] = 1
        output_array[i][i] = 1
    # np.save('../data/PEMS08/PEMS08_aj_0_0_s.npy', output_array)
    cal_array = np.load('../data/PEMS08/PEMS08_aj_0_00_s.npy')
    print(sum(output_array[0]))
    print(output_array)
    for i in range(n):
        for j in range(n):
            if cal_array[i][j]!=0:
                cal_array[i][j] = 1
            else:
                cal_array[i][j] = 0
    print(sum(cal_array[0]))
    print(cal_array)
    print(sum(sum(abs(cal_array-output_array)/2))/(10*n))
    

