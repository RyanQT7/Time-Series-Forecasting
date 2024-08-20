import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import json
import os


def save_dict_as_json(dict_data, filepath):
    # 确保文件的目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 将字典保存为 JSON 文件
    with open(filepath, 'w') as file:
        json.dump(dict_data, file, indent=4)


def load_dict_from_json(filepath):
    # 从 JSON 文件加载字典
    with open(filepath, 'r') as file:
        return json.load(file)


# todo:写一个函数，将txt数据集中第3，4，5列作为工况，
# def set_condition(result_df, condition_number):
#     # 假设result_df是一个NumPy数组，并且你已经定义了要划分的工况数量
#     num_clusters = condition_number  # 给定数量的工况
#     if num_clusters == 1:
#         # 如果工况数量为1，则直接将工况编码设为0
#         cluster_labels = np.zeros((result_df.shape[0], 1))
#         encoding = np.array([np.where(cluster_labels == label)[0] + 1 for label in cluster_labels])
#     else:
#         # 提取第3、4、5列
#         data_to_cluster = result_df[:, [2, 3, 4]]
#
#         # 特征缩放，K-Means对数据的尺度敏感
#         scaler = MinMaxScaler()
#         data_scaled = scaler.fit_transform(data_to_cluster)
#
#         # 使用K-Means算法进行聚类
#         kmeans = KMeans(n_clusters=num_clusters, random_state=0)
#         clusters = kmeans.fit_predict(data_scaled)
#
#         # 将聚类结果添加到result_df的最后一列
#         result_df = np.append(result_df, clusters.reshape(-1, 1), axis=1)
#
#         # 将工况编码为n-1
#         cluster_labels = np.unique(clusters)
#         encoding = np.array([np.where(cluster_labels == label)[0] + 1 for label in clusters])
#
#     # 将编码结果添加到result_df的最后一列
#     result_df = np.append(result_df, encoding.reshape(-1, 1), axis=1)
#
#     # 打印结果查看
#     return result_df

def set_condition(result_df, condition_number):
    num_clusters = condition_number
    if num_clusters == 1:
        # 如果工况数量为1，直接将工况编码设为0
        clusters = np.zeros((result_df.shape[0], 1))
        cluster_labels = clusters + 1
        data_to_cluster = result_df[:, 2:5]
        cluster_centers_original = np.mean(data_to_cluster, axis=0).reshape(1, -1)
    else:
        # 提取第3、4、5列
        data_to_cluster = result_df[:, 2:5]

        # 特征缩放
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_to_cluster)

        # 使用K-Means算法进行聚类
        kmeans = KMeans(n_clusters=num_clusters, n_init=30, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)

        # 将工况编码为1开始的数字，KMeans的标签从0开始，所以加1
        cluster_labels = clusters + 1

        # 获取质心
        cluster_centers_scaled = kmeans.cluster_centers_
        cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

    # 将工况编码添加到result_df的最后一列
    result_df = np.hstack([result_df, cluster_labels.reshape(-1, 1)])

    return result_df, cluster_centers_original


def group_data_by_id(data, json_path, index):
    """
    将数据集按照第一列的id分组，并从JSON文件中获取相应的static值。

    参数:
    - data: NumPy 数组，包含数据集。
    - json_path: 包含static值的JSON文件路径。
    - index: 1-4，用于区分数据

    返回:
    - grouped_data_list: 数组，数组的元素为字典。
    """
    data = np.array(data)
    grouped_data_list = []
    # grouped_data_list=np.array(grouped_data_list)
    static_data = {}

    # 读取JSON文件中的static值
    with open(json_path, 'r') as json_file:
        static_data = json.load(json_file)

    condition_number = len(static_data)
    print("工况数为" + str(condition_number))

    # 初始化字典来存储每个id的数据
    # grouped_data = {}

    # 遍历数据集
    for row in data:
        num_id = row[0]  # 第一列作为id
        condition = row[-1]  # 最后一列作为工况
        roww = np.append(row[1], row[5:-1])
        ts_data = tuple(roww)  # 排除第一列和最后一列的数据,还有中间三列

        if not any(d['id'] == num_id for d in grouped_data_list):
            for u in range(condition_number):
                grouped_data = {}
                grouped_data["id"] = num_id
                grouped_data["static"] = static_data.get("condition_" + str(u + 1))
                grouped_data["ts"] = []
                grouped_data["condition"] = u + 1
                grouped_data_list.append(grouped_data)
                # print(grouped_data["id"],end=' ')
                # print(grouped_data["condition"])

        a = len(grouped_data_list)
        for p in range(len(grouped_data_list)):
            # q=grouped_data_list[p]
            # print('')
            # # grouped_data_list=np.array(grouped_data_list)
            if grouped_data_list[p]["id"] == row[0] and grouped_data_list[p]["condition"] == row[-1]:
                grouped_data_list[p]["ts"].append(ts_data)

    # 将分组字典的值添加到结果数组中
    # grouped_data_list = [grouped_data[id] for id in sorted(grouped_data)]

    return grouped_data_list


if __name__ == "__main__":

    param_list = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "NF", "NC",
                  "EPR", "PS30", "PHI", "NRF", "NRC", "BPR", "FARB", "HT_BLEED", "NF_DMD", "PCNFR_DMD", "W31", "W32"]

    print("Parameters: ", param_list)
    print("Number of total parameters:", len(param_list))

    # save variable names
    np.save('../processed_data/ts_params.npy', param_list)
    print('ts_params.npy: the names of 21 variables')

    static_param_list = ["H", "Ma", "TRA"]
    np.save('../processed_data/static_params.npy', static_param_list)
    print('save names of static descriptors: static_params.npy')

    
    for i in range(1, 5):
        # 读取CSV文件到DataFrame
        df1 = pd.read_csv("F:/dachang/ViTST-main/dataset/CMAPASS_fzy/rawdata/test_FD00" + str(i) + ".txt", sep=" ",
                          header=None,index_col=False)
        df2 = pd.read_csv("F:/dachang/ViTST-main/dataset/CMAPASS_fzy/rawdata/train_FD00" + str(i) + ".txt", sep=" ",
                          header=None,index_col=False)
        # 删除df1中所有值为空的列
        df1.dropna(axis=1, how='all', inplace=True)
    
        # 删除df2中所有值为空的列
        df2.dropna(axis=1, how='all', inplace=True)
    
        # 增加df2中第一列的所有值100
        df2.iloc[:, 0] += df1.iloc[-1, 0]  # iloc用于基于行和列的位置进行索引
    
        # 拼接df1和修改后的df2
        result_df = pd.concat([df1, df2], axis=0)
    
        result_df = np.array(result_df)
    
        condition_number = -1
        if i == 1 or i == 3:
            condition_number = 1
        elif i == 2 or i == 4:
            condition_number = 6
    
        result_df1, cluster_centers = set_condition(result_df, condition_number)
    
        # 将质心转换为字典
        cluster_centers_dict = {"condition_"+str(j + 1): center.flatten().tolist() for j, center in enumerate(cluster_centers)}
    
        # 指定 JSON 文件的完整路径
        json_filepath = "F:/dachang/ViTST-main/dataset/CMAPASS_fzy/rawdata/FD00"+str(i)+"_condition.json"
    
        # 保存字典到 JSON 文件
        save_dict_as_json(cluster_centers_dict, json_filepath)
    
        np.save("../rawdata/FD00" + str(i) + ".npy", result_df1)
        print("FD00" + str(i) + ".npy saved")

    
    for i in range(4):
        data_1 = np.load("../rawdata/FD00"+str(i+1)+".npy", allow_pickle=True)
        json_path_1 = "../rawdata/FD00"+str(i+1)+"_condition.json"
    
        P_list_1 = group_data_by_id(data_1, json_path_1, i+1)
        P_list_1=np.array(P_list_1)
        np.save("../processed_data/P_list_" + str(i+1) + ".npy", P_list_1)
        # print(P_list_1[-1])
        # print(P_list_1)
    # # 现在result_df包含了拼接后的数据
    # print(result_df)
    

    # df_outcomes_1 = pd.read_csv('../rawdata/RUL_FD001.txt', sep=" ",
    #                             names=["Remaining_service_life"], header=None, index_col=False)
    # # 删除df_outcomes_1中所有值为空的列
    # df_outcomes_1.dropna(axis=1, how='all', inplace=True)
    # df_outcomes_2 = pd.read_csv('../rawdata/RUL_FD002.txt', sep=" ",
    #                             names=["Remaining_service_life"], header=None, index_col=False)
    # df_outcomes_2.dropna(axis=1, how='all', inplace=True)
    # df_outcomes_3 = pd.read_csv('../rawdata/RUL_FD003.txt', sep=" ",
    #                             names=["Remaining_service_life"], header=None, index_col=False)
    # df_outcomes_3.dropna(axis=1, how='all', inplace=True)
    # df_outcomes_4 = pd.read_csv('../rawdata/RUL_FD004.txt', sep=" ",
    #                             names=["Remaining_service_life"], header=None, index_col=False)
    # df_outcomes_4.dropna(axis=1, how='all', inplace=True)
    # print(df_outcomes_1.head(n=5))
    # print(df_outcomes_2.head(n=5))
    # print(df_outcomes_3.head(n=5))
    # print(df_outcomes_4.head(n=5))
    #
    # arr_outcomes_1 = np.array(df_outcomes_1)
    # arr_outcomes_2 = np.array(df_outcomes_2)
    # arr_outcomes_3 = np.array(df_outcomes_3)
    # arr_outcomes_4 = np.array(df_outcomes_4)
    #
    # n_1 = arr_outcomes_1.shape[0]
    # n_2 = arr_outcomes_2.shape[0]
    # n_3 = arr_outcomes_3.shape[0]
    # n_4 = arr_outcomes_4.shape[0]
    # print('n_1 = %d, n_2 = %d, n_3 = %d, n_4 = %d' % (n_1, n_2, n_3, n_4))

    for i in range(1, 5):
        df_outcomes = pd.read_csv("../rawdata/RUL_FD00" + str(i) + ".txt", sep=" ",
                                  names=["Remaining_service_life"], header=None, index_col=False)
        # 删除df_outcomes_1中所有值为空的列
        df_outcomes.dropna(axis=1, how='all', inplace=True)
        arr_outcomes = np.array(df_outcomes)
        n = arr_outcomes.shape[0]
        data_1 = np.load("../rawdata/FD00" + str(i) + ".npy", allow_pickle=True)
        for y in range(int(data_1[-1][0]) - n):
            arr_outcomes = np.append(arr_outcomes, 0)
        np1 = np.ones(n)
        np2 = np.zeros(int(data_1[-1][0]) - n)
        np3 = np.concatenate([np1, np2], axis=0)
        arr_outcomes = np.vstack((arr_outcomes, np3))
        arr_outcomes = np.transpose(arr_outcomes)
        print(arr_outcomes.shape)
        np.save("../processed_data/arr_outcomes_" + str(i) + ".npy", arr_outcomes)


