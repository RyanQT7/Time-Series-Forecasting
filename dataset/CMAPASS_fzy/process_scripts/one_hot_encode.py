# Irregular sampling for PhysioNet-2012 dataset
#  & Train/test/val splits
#
# Author: Theodoros Tsiligkaridis
# Last updated: May 4 2021
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:90% !important; }</style>"))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ## Irregular sampling
for i in range(1, 5):
    P_list = []
    arr_outcomes = []
    P_list = np.load("../processed_data/P_list_" + str(i) + ".npy", allow_pickle=True)
    arr_outcomes = np.load("../processed_data/arr_outcomes_" + str(i) + ".npy", allow_pickle=True)

    ts_params = np.load("../processed_data/ts_params.npy", allow_pickle=True)
    static_params = np.load("../processed_data/static_params.npy", allow_pickle=True)

    print('number of samples: ', len(P_list))
    print(len(ts_params), ts_params)
    print(len(static_params), static_params)

    # Estimate max_len across dataset
    n = len(P_list)
    max_tmins = 200
    len_ts = []
    ts = []
    for ind in range(n):  # for each patient
        ts = P_list[ind]['ts']

        if len(ts) == 0:
            len_ts.append(0)
            continue

        len_ts.append(ts[-1][0])
    np_max = np.max(len_ts)

    print('max unique time series length:', np_max)  # np.max(len_ts) = 362.0，FD001时

    # # Histogram of time points
    # _ = plt.hist(np.array(len_ts), bins='auto')
    # plt.xlabel('Number of time points')
    # plt.ylabel('Counts')
    # plt.show()

    extended_static_list = ['flight height', 'mach number', 'Throttle lever Angle', 'Type of working condition']
    np.save("../processed_data/extended_static_params.npy", extended_static_list)

    """Group all patient time series into arrays"""
    n = len(P_list)
    max_len = np_max
    F = len(ts_params)
    PTdict_list = []
    max_hr = 0
    for ind in range(n):
        ID = P_list[ind]['id']
        static = P_list[ind]['static']
        condition_type = P_list[ind]['condition']
        ts = P_list[ind]['ts']
        # if ts[-1][0]<max_tmins:
        #     continue;

        # find unique times
        unq_tmins = []
        for sample in ts:
            current_tmin = sample[0]
            # if (current_tmin not in unq_tmins) and (current_tmin < max_tmins):
            unq_tmins.append(current_tmin)
        #     print('unique times (mins):', unq_tmins)
        #     print('sequence length: ', len(unq_tmins))
        unq_tmins = np.array(unq_tmins)

        # one-hot encoding of categorical static variables
        extended_static = [static[0], static[1], static[2], condition_type]
        # if static[1] == 0:
        #     extended_static[1] = 1
        # elif static[1] == 1:
        #     extended_static[2] = 1
        # if static[3] == 1:
        #     extended_static[4] = 1
        # elif static[3] == 2:
        #     extended_static[5] = 1
        # elif static[3] == 3:
        #     extended_static[6] = 1
        # elif static[3] == 4:
        #     extended_static[7] = 1

        # construct array of maximal size
        # Parr = np.zeros((max_len, F))
        # Tarr = np.zeros((max_len, 1))

        # for each time measurement find index and store
        # for sample in ts:
        #     tmins = sample[2]
        #     param = sample[-2]
        #     value = sample[-1]
        #     if tmins < max_tmins:
        #         time_id = np.where(tmins == unq_tmins)[0][0]
        #         param_id = np.where(ts_params == param)[0][0]
        #         Parr[time_id, param_id] = value
        #         Tarr[time_id, 0] = unq_tmins[time_id]

        length = len([t[0] for t in ts])
        # a=ts[:][0]
        # construct dictionary new_array = [t[1:] for t in original_array]
        my_dict = {'id': ID, 'static': static, 'extended_static': extended_static, 'arr': np.array([list(t[1:]) for t in ts]),
                   'time': [t[0] for t in ts], 'length': length, 'condition': condition_type}

        # add array into list
        PTdict_list.append(my_dict)

    print(len(PTdict_list))
    PTdict_list = np.array(PTdict_list)
    np.save("../processed_data/PTdict_list_" + str(i) + ".npy", PTdict_list)
    print('PTdict_list.npy saved', PTdict_list[0].keys())
