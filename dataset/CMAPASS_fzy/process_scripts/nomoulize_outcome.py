import numpy as np

FD002_ImageDict_list = np.load('../processed_data/FD002_ImageDict_list.npy', allow_pickle=True)
result = np.load('../processed_data/fd002/result.npy', allow_pickle=True)

remain_life_values=np.array([re['arr_outcome']['remain_life'] for re in FD002_ImageDict_list])
# 最小-最大归一化
min_value = remain_life_values.min()
max_value = remain_life_values.max()
print(f'FD002_ImageDict_list的最小值为{min_value}')
print(f'FD002_ImageDict_list的最大值为{max_value}')
for i in range(len(FD002_ImageDict_list)):
    # 应用归一化公式
    normalized_values = (remain_life_values[i] - min_value) / (max_value - min_value)
    FD002_ImageDict_list[i]['arr_outcome']['remain_life']=normalized_values
FD002_ImageDict_array = np.array(FD002_ImageDict_list)

# 保存数组到.npy文件
np.save('../processed_data/normalized/FD002_ImageDict_list.npy', FD002_ImageDict_array)

remain_life_values2=np.array([re['remain_life'] for re in result])
# 最小-最大归一化
min_value = remain_life_values2.min()
max_value = remain_life_values2.max()
print(f'result的最小值为{min_value}')
print(f'result的最大值为{max_value}')
for i in range(len(result)):
    # 应用归一化公式
    normalized_values = (remain_life_values2[i] - min_value) / (max_value - min_value)
    result[i]['remain_life']=normalized_values
result_array = np.array(result)

# 保存数组到.npy文件
np.save('../processed_data/normalized/result.npy', result_array)