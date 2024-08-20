import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

choose_num = 1
num_of_conditions=1

"""
Dataset configurations 
"""
max_tmins = 362.0
"""
FD001:max unique time series length: 362.0
FD002:max unique time series length: 378.0
FD003:max unique time series length: 525.0
FD004:max unique time series length: 543.0
"""

param_detailed_description = {
    "T2": "Total temperature at fan inlet (°R)",
    "T24": "Total temperature at Low-Pressure Compressor outlet (°R)",
    "T30": "Total temperature at High-Pressure Compressor outlet (°R)",
    "T50": "Total temperature at Low-Pressure Compressor outlet (°R)",
    "P2": "Pressure at fan inlet (psia)",
    "P15": "Total pressure in bypass-duct (psia)",
    "P30": "Total pressure in bypass-duct (psia)",
    "Nf": "Physical fan speed (rpm)",
    "Nc": "Physical core speed (rpm)",
    "epr": "Engine pressure ratio (P50/P2) (--)",
    "Ps30": "Static pressure at HPC outlet (psia)",
    "phi": "Ratio of fuel flow to Ps30 (pps/psi)",
    "NRf": "Corrected fan speed (rpm)",
    "NRc": "Corrected core speed (rpm)",
    "BPR": "Bypass Ratio (--)",
    "farB": "Burner fuel-air ratio (--)",
    "htBleed": "Bleed Enthalpy (--)",
    "Nf_dmd": "Demanded fan speed (rpm)",
    "PCNfR_dmd": "Demanded corrected fan speed (rpm)",
    "W31": "High-Pressure Compressor coolant bleed (lbm/s)",
    "W32": "Low-Pressure Compressor coolant bleed (lbm/s)",
}


color_detailed_description = {
    "aliceblue": "1",
    "antiquewhite": "2",
    "aqua": "3",
    "aquamarine": "4",
    "azure": "5",
    "beige": "6",
    "bisque": "7",
    "black": "8",
    "blanchedalmond": "9",
    "blue": "10",
    "blueviolet": "11",
    "brown": "12",
    "burlywood": "13",
    "cadetblue": "14",
    "chartreuse": "15",
    "chocolate": "16",
    "coral": "17",
    "cornflowerblue": "18",
    "cornsilk": "19",
    "crimson": "20",
    "cyan": "21",
    "darkblue": "22",
    "darkcyan": "23",
    "darkgoldenrod": "24",
    "darkgray": "25",
    "darkgreen": "26",
    "darkgrey": "27",
    "darkkhaki": "28",
    "darkmagenta": "29",
    "darkolivegreen": "30",
    "darkorange": "31",
    "darkorchid": "32",
    "darkred": "33",
    "darksalmon": "34",
    "darkseagreen": "35",
    "darkslateblue": "36",
    "darkslategray": "37",
    "darkslategrey": "38",
    "darkturquoise": "39",
    "darkviolet": "40",
    "deeppink": "41",
    "deepskyblue": "42",
    "dimgray": "43",
    "dimgrey": "44",
    "dodgerblue": "45",
    "firebrick": "46",
    "floralwhite": "47",
    "forestgreen": "48",
    "fuchsia": "49",
    "gainsboro": "50",
    "ghostwhite": "51",
    "gold": "52",
    "goldenrod": "53",
    "gray": "54",
    "green": "55",
    "greenyellow": "56",
    "grey": "57",
    "honeydew": "58",
    "hotpink": "59",
    "indianred": "60",
    "indigo": "61",
    "ivory": "62",
    "khaki": "63",
    "lavender": "64",
    "lavenderblush": "65",
    "lawngreen": "66",
    "lemonchiffon": "67",
    "lightblue": "68",
    "lightcoral": "69",
    "lightcyan": "70",
    "lightgoldenrodyellow": "71",
    "lightgray": "72",
    "lightgreen": "73",
    "lightgrey": "74",
    "lightpink": "75",
    "lightsalmon": "76",
    "lightseagreen": "77",
    "lightskyblue": "78",
    "lightslategray": "79",
    "lightslategrey": "80",
    "lightsteelblue": "81",
    "lightyellow": "82",
    "lime": "83",
    "limegreen": "84",
    "linen": "85",
    "magenta": "86",
    "maroon": "87",
    "mediumaquamarine": "88",
    "mediumblue": "89",
    "mediumorchid": "90",
    "mediumpurple": "91",
    "mediumseagreen": "92",
    "mediumslateblue": "93",
    "mediumspringgreen": "94",
    "mediumturquoise": "95",
    "mediumvioletred": "96",
    "midnightblue": "97",
    "mintcream": "98",
    "mistyrose": "99",
    "moccasin": "100",
    "navajowhite": "101",
    "navy": "102",
    "oldlace": "103",
    "olive": "104",
    "olivedrab": "105",
    "orange": "106",
    "orangered": "107",
    "orchid": "108",
    "palegoldenrod": "109",
    "palegreen": "110",
    "paleturquoise": "111",
    "palevioletred": "112",
    "papayawhip": "113",
    "peachpuff": "114",
    "peru": "115",
    "pink": "116",
    "plum": "117",
    "powderblue": "118",
    "purple": "119",
    "red": "120",
    "rosybrown": "121",
    "royalblue": "122",
    "saddlebrown": "123",
    "salmon": "124",
    "sandybrown": "125",
    "seagreen": "126"
}

"""
Code
"""


def construct_demogr_description(static_demogr):
    desc = []
    # age
    if static_demogr[0]:
        desc.append(f"flight altitude is {static_demogr[0]} thousand feet")

    # if int(static_demogr[1]) == 0:
    #     desc.append("female")
    # elif int(static_demogr[1]) == 1:
    #     desc.append("male")
    if static_demogr[1]:
        desc.append(f"the flying speed is {static_demogr[1]} mach")

    if static_demogr[2]:
        desc.append(f"throttle lever Angle is {static_demogr[2]} degrees")
    #
    # if static_demogr[3]:
    #     desc.append(f"engine working environment is type {static_demogr[3]}")

    # # icu type
    # if static_demogr[3] > 0:
    #     if int(static_demogr[3]) == 1:
    #         icu = "coronary care unit"
    #         desc.append(f"stayed in {icu}")
    #     elif int(static_demogr[3]) == 2:
    #         icu = "cardiac surgery recovery unit"
    #         desc.append(f"stayed in {icu}")
    #     elif int(static_demogr[3]) == 3:
    #         icu = "medical ICU"
    #         desc.append(f"stayed in {icu}")
    #     elif int(static_demogr[3]) == 4:
    #         icu = "surgical ICU"
    #         desc.append(f"stayed in {icu}")

    if desc:
        desc = (f"The engine operates in a total of {num_of_conditions} "
                f"operating environments,the working environment of the engine is as follows:") + ", ".join(desc) + "."
    else:
        desc = ""

    return desc


def draw_image(pid, split_idx, ts_orders, ts_values, ts_times, ts_params, ts_scales,
               override, differ, outlier, interpolation, order,
               image_size, grid_layout,
               linestyle, linewidth, marker, markersize,
               ts_color_mapping, ts_idx_mapping):
    # set matplotlib param
    grid_height = grid_layout[0]
    grid_width = grid_layout[1]
    if image_size is None:
        cell_height = 64
        cell_width = 64
        img_height = grid_height * cell_height
        img_width = grid_width * cell_width
    else:
        img_height = image_size[0]
        img_width = image_size[1]

    dpi = 100
    plt.rcParams['savefig.dpi'] = dpi  # default=100
    plt.rcParams['figure.figsize'] = (img_width / dpi, img_height / dpi)
    plt.rcParams['figure.frameon'] = False

    # save path
    base_path = f"{linestyle}x{linewidth}_{marker}x{markersize}_{grid_height}x{grid_width}_{img_height}x{img_width}_split{split_idx}_images"
    base_path2 = f"{linestyle}x{linewidth}_{marker}x{markersize}_{grid_height}x{grid_width}_{img_height}x{img_width}_split{split_idx}_images"

    if interpolation:
        base_path = "interpolation_" + base_path
    if differ:
        base_path = "differ_" + base_path
    if order:
        base_path = f"order_" + base_path
    if outlier:
        base_path = f"{outlier}_" + base_path
    base_path = 'FD00' + str(choose_num) + '_' + base_path
    base_path = "../processed_data/" + base_path

    if not os.path.exists(base_path): os.mkdir(base_path)
    img_path = os.path.join(base_path, f"{pid}.png")
    # if os.path.exists(img_path):
    #     if not override:
    #         return []


    if interpolation:
        base_path2 = "interpolation_" + base_path2
    if differ:
        base_path2 = "differ_" + base_path2
    if order:
        base_path2 = f"order_" + base_path2
    if outlier:
        base_path2 = f"{outlier}_" + base_path2
    base_path2 = 'ConID_FD00' + str(choose_num) + '_' + base_path2
    base_path2 = "../processed_data/" + base_path2

    if not os.path.exists(base_path2): os.mkdir(base_path2)
    img_path = os.path.join(base_path2, f"{pid}.png")
    # if os.path.exists(img_path):
        # if not override:
        #     return []

    drawed_params = []

    # find the information across all the patients
    max_hours, num_params = ts_values.shape[0], ts_values.shape[1]
    for idx, param_idx in enumerate(ts_orders):  # ts_desc: (215, 36)

        param = ts_params[param_idx]
        ts_value = ts_values[:, param_idx]

        # the scale of x, y axis
        param_scale_x = [0, max_tmins]
        param_scale_y = ts_scales[param_idx]
        # only one value, expand the y axis
        if param_scale_y[0] == param_scale_y[1]:
            param_scale_y = [param_scale_y[0] - 0.5, param_scale_y[0] + 0.5]

        ts_time = np.array(ts_times).reshape(-1, 1)
        ts_value = np.array(ts_value).reshape(-1, 1)
        # handling missing value and extreme values
        kept_index = (ts_value != 0)
        removed_index = (ts_value == 0)
        if interpolation:
            ts_time = ts_time[kept_index]
            ts_value = ts_value[kept_index]
        else:
            ts_time[removed_index] = np.nan
            ts_value[removed_index] = np.nan
        # handling extreme values
        min_index = (ts_value < param_scale_y[0])
        ts_value[min_index] = param_scale_y[0]
        # handling extreme values
        max_index = (ts_value > param_scale_y[1])
        ts_value[max_index] = param_scale_y[1]

        ##### draw the plot for each parameter
        # param_marker = ts_marker_mapping[param]
        param_color = ts_color_mapping[param]
        param_idx = ts_idx_mapping[param]

        # plt.subplot(grid_height, grid_width, param_idx+1) # 6*6
        plt.subplot(grid_height, grid_width, idx + 1)  # 6*6
        if differ:  # using different colors and markers
            plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                     color=param_color)
            # print()
        else:
            plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth, marker=marker,
                     markersize=markersize, )

        # set the scale for x, y axis
        plt.xlim(param_scale_x)
        plt.ylim(param_scale_y)
        plt.xticks([])
        plt.yticks([])

        drawed_params.append(param)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # 显示图表
    # plt.show()
    plt.savefig(img_path, pad_inches=0)
    plt.clf()

    return drawed_params


def construct_image(
        linestyle="-", linewidth=1, marker="x", markersize=2,
        override=False,
        differ=False,
        outlier=None,
        interpolation=True,
        order=False,
        grid_layout=(4, 6),
        image_size=None
):
    # load data
    Pdict_list = np.load(f'../processed_data/PTdict_list_' + str(choose_num) + '.npy', allow_pickle=True)
    arr_outcomes = np.load(f'../processed_data/arr_outcomes_' + str(choose_num) + '.npy', allow_pickle=True)
    ts_params = np.load(f'../processed_data/ts_params.npy', allow_pickle=True)

    num_samples = len(Pdict_list)
    print(f"{num_samples} patients in total!")
    print(list(Pdict_list[0].keys()))  # ['id', 'static', 'extended_static', 'arr', 'time', 'length']

    num_ts_params = len(ts_params)  # 36
    print(f"{num_ts_params} parameters!")

    plt_colors = list(color_detailed_description.keys())
    # np.random.seed(10)
    random.shuffle(plt_colors)
    num_colors = len(plt_colors)
    print(f"{num_colors} colors!")

    """
    for random param color exp
    """
    if num_ts_params > num_colors:
        plt_colors = []
        rs = list(np.linspace(0.0, 1.0, num_ts_params))
        random.shuffle(rs)  # from 0 to 1
        gs = list(np.linspace(0.0, 1.0, num_ts_params))
        random.shuffle(gs)  # from 0 to 1
        bs = list(np.linspace(0.0, 1.0, num_ts_params))
        random.shuffle(bs)  # from 0 to 1
        for idx in range(num_ts_params):
            color = (rs[idx], gs[idx], bs[idx])
            plt_colors.append(color)

    # construct the mapping from param to marker, color, and idx
    ts_idx_mapping = {}
    ts_color_mapping = {}
    for idx, param in enumerate(ts_params):
        ts_color_mapping[param] = plt_colors[idx]
        ts_idx_mapping[param] = idx

    with open('../processed_data/param_idx_mapping_' + str(choose_num) + '.json', 'w') as f:
        json.dump(ts_idx_mapping, f)
    with open('../processed_data/param_color_mapping_' + str(choose_num) + '.json', 'w') as f:
        json.dump(ts_color_mapping, f)

    for split_idx in range(5):
        # start constructing the data list
        ImageDict_list = []
        demogr_lengths = []

        base_path = '../'
        split_path = '/splits/FD00' + str(choose_num) + '_split' + str(split_idx + 1) + '.npy'
        idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)
        # extract train/val/test examples
        Ptrain = Pdict_list[idx_train]
        Pval = Pdict_list[idx_val]
        Ptest = Pdict_list[idx_test]

        # first round, find the mean and std for each param on training set
        train_ts_values = [[] for _ in range(num_ts_params)]
        all_ts_values = [[] for _ in range(num_ts_params)]
        stat_ts_values = np.ones(shape=(num_ts_params, 12))  # mean, std, y_min, y_max
        for idx, p in tqdm(enumerate(Ptrain)):
            ts_values = p['arr']  # (267, 21)
            for param_idx in range(num_ts_params):
                # 使用列表推导式将每个元组转换为列表
                # list_ts_values = [list(t) for t in ts_values]
                # print(len(list_ts_values))
                ts_value = ts_values[:, param_idx]
                ts_value = np.array(ts_value).reshape(-1, 1)
                # handling missing value
                ts_value = ts_value[ts_value != 0]
                train_ts_values[param_idx].extend(list(ts_value))

        for idx, p in tqdm(enumerate(Pdict_list)):
            ts_values = p['arr']  # (60, 34)
            for param_idx in range(num_ts_params):  # ts_desc: (60, 34)
                ts_value = ts_values[:, param_idx]
                ts_value = np.array(ts_value).reshape(-1, 1)
                # handling missing value
                ts_value = ts_value[ts_value != 0]
                all_ts_values[param_idx].extend(list(ts_value))

        # sort the params based on missing ratios
        if order:
            ts_value_nums = [len(_) for _ in train_ts_values]
            ts_orders = np.argsort(ts_value_nums)[::-1]
        else:
            ts_orders = list(range(num_ts_params))

        # change from list to array
        for param_idx in range(num_ts_params):
            train_ts_values[param_idx] = np.array(train_ts_values[param_idx])

        for param_idx in range(num_ts_params):  # ts_desc: (60, 34)
            param_ts_value = np.array(train_ts_values[param_idx])

            stat_ts_values[param_idx, 0] = param_ts_value.mean()
            stat_ts_values[param_idx, 1] = param_ts_value.std()
            stat_ts_values[param_idx, 2] = param_ts_value.min()
            stat_ts_values[param_idx, 3] = param_ts_value.max()

            """
            option 1. remove outliers with boxplot
            """
            q1 = np.percentile(param_ts_value, 25)
            q3 = np.percentile(param_ts_value, 75)
            med = np.median(param_ts_value)
            iqr = q3 - q1
            upper_bound = q3 + (1.5 * iqr)
            lower_bound = q1 - (1.5 * iqr)
            stat_ts_values[param_idx, 4] = lower_bound
            stat_ts_values[param_idx, 5] = upper_bound
            param_ts_value1 = param_ts_value[(lower_bound < param_ts_value) & (upper_bound > param_ts_value)]
            outlier_ratio = 1 - (len(param_ts_value1) / len(param_ts_value))
            # print(f"{param_idx}, {outlier_ratio}")

            """
            option 2. remove outliers with standard deviation
            """
            mean = np.mean(param_ts_value)
            std = np.std(param_ts_value)
            upper_bound = mean + (4 * std)
            lower_bound = mean - (4 * std)
            stat_ts_values[param_idx, 6] = lower_bound
            stat_ts_values[param_idx, 7] = upper_bound
            param_ts_value2 = param_ts_value[(lower_bound < param_ts_value) & (upper_bound > param_ts_value)]
            outlier_ratio = 1 - (len(param_ts_value2) / len(param_ts_value))

            """
            option 3. remove outliers with modified z-score
            """
            med = np.median(param_ts_value)
            deviation_from_med = param_ts_value - med
            mad = np.median(np.abs(deviation_from_med))
            # modified_z_score = (deviation_from_med / mad)*0.6745
            lower_bound = (-3.5 / 0.6745) * mad + med
            upper_bound = (3.5 / 0.6745) * mad + med
            stat_ts_values[param_idx, 8] = lower_bound
            stat_ts_values[param_idx, 9] = upper_bound
            param_ts_value3 = param_ts_value[(lower_bound < param_ts_value) & (upper_bound > param_ts_value)]
            outlier_ratio = 1 - (len(param_ts_value3) / len(param_ts_value))
            # print(f"{param_idx}, {outlier_ratio}")

            """
            option 4. quartile
            """
            sorted_param_ts_value = np.sort(param_ts_value)
            value_len = sorted_param_ts_value.shape[0]
            max_position = min(value_len - 1, round(value_len * 0.99995))
            min_position = max(0, round(value_len * 0.00005))
            upper_bound = sorted_param_ts_value[max_position]
            lower_bound = sorted_param_ts_value[min_position]
            stat_ts_values[param_idx, 10] = lower_bound
            stat_ts_values[param_idx, 11] = upper_bound

        # second round, draw the image for each patient
        for idx, p in tqdm(enumerate(Pdict_list)):

            pid = int(p['id'])
            ts_values = p['arr']  # (215, 36)
            ts_times = p['time']

            # textual label
            arr_outcome = arr_outcomes[idx]
            label = int(arr_outcome[-1])
            label_name = "Total life cycle" if label == 1 else "Partial life cycle"

            # static feature
            static_demogr = p['extended_static']
            demogr_desc = construct_demogr_description(static_demogr)

            # deal with outliers
            if not outlier:
                ts_scales = stat_ts_values[:, 2:4]  # no removal
            elif outlier == "iqr":
                ts_scales = stat_ts_values[:, 4:6]  # iqr
            elif outlier == "std":
                ts_scales = stat_ts_values[:, 6:8]  # std
            elif outlier == "mzs":
                ts_scales = stat_ts_values[:, 8:10]  # mzs
            elif outlier == "qt":
                ts_scales = stat_ts_values[:, 10:12]  # quartile

            # draw the image for each p
            drawed_params = draw_image(pid, split_idx, ts_orders, ts_values, ts_times, ts_params, ts_scales, override,
                                       differ, outlier, interpolation, order,
                                       image_size, grid_layout,
                                       linestyle, linewidth, marker, markersize,
                                       ts_color_mapping, ts_idx_mapping)

            ImageDict = {
                "id": pid,
                "text": demogr_desc,
                "label": label,
                "label_name": label_name,
                "arr_outcome": arr_outcome
            }
            ImageDict_list.append(ImageDict)

    print(len(ImageDict_list))
    np.save(f'../processed_data/FD00' + str(choose_num) + '_ImageDict_list.npy', ImageDict_list)
    print(f"Save data in ImageDict_list.npy")


if __name__ == "__main__":
    np.random.seed(10)
    random.seed(10)  # 确保Python内置的random模块也使用相同的种子
    construct_image(
        linestyle="-", linewidth=1, marker="x", markersize=2,
        override=False,
        differ=True,
        outlier=None,
        interpolation=True,
        order=True,
        grid_layout=(4, 6),
        image_size=None
    )
