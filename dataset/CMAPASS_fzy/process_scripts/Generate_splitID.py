import random

import numpy as np

arr_outcomes = np.load('../processed_data/fd002/result.npy', allow_pickle=True)
n_2= len(arr_outcomes)
n=n_2
u=2
# split randomization over folds
"""Use 9:1:1 split"""
p_train = 0.80
p_val = 0.10
p_test = 0.10



# for u in range(1, 5):
#     if u == 1:
#         n = 200
#     elif u == 2:
#         n = n_2 # FD002
#     elif u == 3:
#         n = 200
#     elif u == 4:
#         n = 497

    # n = 11988  # original 12000 patients, remove 12 outliers
n_train = round(n * p_train)
n_val = round(n * p_val)
n_test = n - (n_train + n_val)
print(n_train, n_val, n_test)
Nsplits = 5
for j in range(Nsplits):
    p = np.random.permutation(n)
    # for yui in range(9):
    #     p= np.concatenate((p, np.random.permutation(n)), axis=0)
    idx_train = p[:n_train]
    idx_val = p[n_train:n_train + n_val]
    idx_test = p[n_train + n_val:]
    np.save('../splits/FD00'+str(u)+'/FD00'+str(u)+'_split' + str(j + 1) + '.npy', (idx_train, idx_val, idx_test))

    # np.save('../splits/phy12_split_subset'+str(j+1)+'.npy', (idx_train, idx_val, idx_test))
print('split IDs saved')

"""
# # check first split
# idx_train,idx_val,idx_test = np.load('../splits/phy12_split1.npy', allow_pickle=True)
# print(len(idx_train), len(idx_val), len(idx_test))
# np.random.seed(10)
# random.seed(10)  # 确保Python内置的random模块也使用相同的种子
choose_num=2
u=2
Pdict_list = np.load(f'../processed_data/fd00' + str(choose_num) + '/data.npy', allow_pickle=True)
arr_outcomes = np.load(f'../processed_data/fd00' + str(choose_num) + '/result.npy', allow_pickle=True)
ts_params = np.load(f'../processed_data/ts_params.npy', allow_pickle=True)
p=[]
n=len(arr_outcomes)
for i in range(len(arr_outcomes)):
    p.append(int(arr_outcomes[i]['id']))

p=np.array(p)
# n_train = id_array[0:n * p_train]
# n_val = id_array[n * p_train:n * p_train+n * p_val]
# n_test = id_array[n * p_train+n * p_val:n * p_train+n * p_val+n * p_test]
n_train = round(n * p_train)
n_val = round(n * p_val)
n_test = n - (n_train + n_val)
print(n_train, n_val, n_test)
Nsplits = 5
for j in range(Nsplits):
    random.shuffle(p)
    idx_train = p[:n_train]
    idx_val = p[n_train:n_train + n_val]
    idx_test = p[n_train + n_val:]
    np.save('../splits/FD002/FD00' + str(u) + '_split' + str(j + 1) + '.npy', (idx_train, idx_val, idx_test))

"""