import numpy as np

for j in range(1, 5):
    """Remove 12 patients at blacklist"""
    PTdict_list = np.load("../processed_data/PTdict_list_" + str(j) + ".npy", allow_pickle=True)
    # arr_outcomes = np.load("../processed_data/arr_outcomes_" + str(j) + ".npy", allow_pickle=True)

    # remove blacklist patients
    # blacklist = ['140501', '150649', '140936', '143656', '141264', '145611', '142998', '147514', '142731', '150309', '155655', '156254']

    i = 0
    n = len(PTdict_list)
    while i < n:
        pid = len(PTdict_list[i]['arr'])
        if pid == 0:
            PTdict_list = np.delete(PTdict_list, i)
            print("drop ", i)
            # arr_outcomes = np.delete(arr_outcomes, i, axis=0)
            n -= 1
        i += 1
    print("len of PTdict_list is",len(PTdict_list))

    # np.save("../processed_data/PTdict_list_" + str(j) + ".npy", PTdict_list)
    # np.save("../processed_data/arr_outcomes_" + str(j) + ".npy", arr_outcomes)
