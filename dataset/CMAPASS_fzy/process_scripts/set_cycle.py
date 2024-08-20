import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P_list = np.load("../processed_data/P_list_2.npy", allow_pickle=True)
arr_outcomes = np.load("../processed_data/arr_outcomes_2" + ".npy", allow_pickle=True)
ts_params = np.load("../processed_data/ts_params.npy", allow_pickle=True)
static_params = np.load("../processed_data/static_params.npy", allow_pickle=True)

# 打开JSON文件
# with open('../rawdata/FD002_condition.json', 'r', encoding='utf-8') as file:
#     # 读取JSON数据
#     condition_data = json.load(file)
print('number of samples: ', len(P_list))
print(len(ts_params), ts_params)
print(len(static_params), static_params)

PTdict_list = []
DIS = []  # 工况随着周期的分布 例子：dis[0][0]=6 表示id=1的发动机的第一个周期是第六种工况
result=[]
n = int(len(P_list) / 6)
max_t = 100
for i in range(n):
    p = P_list[i * 6:(i + 1) * 6]
    L = p[0]['ts'] + p[1]['ts'] + p[2]['ts'] + p[3]['ts'] + p[4]['ts'] + p[5]['ts']  # 把同一个id的六个工况的ts数据整合到一起
    L.sort()
    len_L= len(L)
    if (len(L) < max_t):  # 整合到一起后长度小于200的话肯定就不满足
        continue
    distribution = []  # 找分布 看周期是从哪一个工况过来的
    for j in range(max_t):
        f = float(j + 1)

        for k in range(6):
            for sample in p[k]['ts']:
                if f == sample[0]:
                    distribution.append(k + 1)
    np.array(distribution)
    DIS.append(distribution)
    res={}
    for u in range(6):
        ID = P_list[i*6+u]['id']
        static = P_list[i*6+u]['static']  # 六种工况不知道取哪一个作为静态变量
        condition = P_list[i*6+u]['condition']
        ts = P_list[i*6+u]['ts']
        # new_ts=[]
        # for y in P_list[i * 6 + u]['ts']:
        #     if y[0] <= max_t:
        #         new_ts.append(y)
        extended_static = [static[0], static[1], static[2], condition]
        length = len([t[0] for t in ts])
        my_dict = {'id': ID, 'static': static, 'extended_static': extended_static,
                   'arr': np.array([list(t[1:]) for t in ts]),
                   'time': [t[0] for t in ts], 'length': length, 'condition': condition}
        PTdict_list.append(my_dict)
    res["id"] = my_dict["id"]
    res["remain_life"]=arr_outcomes[int(res["id"]-1)][0]
    res["is_full_cycle"]=1-arr_outcomes[int(res["id"]-1)][1]
    res["remain_life"]=res["remain_life"]+len_L-max_t
    result.append(res)


PTdict_list=np.array(PTdict_list)
DIS=np.array(DIS)
result=np.array(result)
# print(PTdict_list)
# print(DIS)
# print(result)
np.save("../processed_data/fd002/data" + ".npy", PTdict_list)
np.save("../processed_data/fd002/DIS" + ".npy", DIS)
np.save("../processed_data/fd002/result" + ".npy", result)






