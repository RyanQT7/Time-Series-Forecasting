from email.mime import image
import numpy as np
import pandas as pd
import argparse
import random
from itertools import chain
import os
from datasets import load_dataset
from datasets import load_metric
from datasets import Dataset, Image


def load_image(Pdict_list, y, base_path, split_idx, dataset_prefix, dataset_condition, missing_ratio, max_tmins):
    images_path = []
    images_path2 = []
    labels = []
    texts = []
    for idx, d in enumerate(Pdict_list):
        pid = d['id']
        text = d['text']
        assert d['arr_outcome']['remain_life'] == y[idx]
        label = y[idx]
        labels.append(label)
        texts.append(text)

        if missing_ratio == 0:
            image_path = base_path + f'/processed_data/{dataset_prefix}split{split_idx - 1}_cut_by{max_tmins}/{pid}.png'
        elif missing_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            image_path = base_path + f'/processed_data/{dataset_prefix}ms{missing_ratio}_cut_by{max_tmins}/{pid}.png'
        else:
            raise Exception(f"No dataset for this missing ratio {missing_ratio}")
        images_path.append(image_path)

        image_path2 = base_path + f'/processed_data/{dataset_condition}split{split_idx - 1}_cut_by{max_tmins}/{pid}.png'
        images_path2.append(image_path2)

    datadict = {"image": images_path, "ID_image": images_path2, "text": texts, "label": labels}
    # 创建数据集实例
    dataset = Dataset.from_dict(datadict)
    # 将"image"列转换为Image类型
    dataset = dataset.cast_column("image", Image())
    # 将"ID_image"列也转换为Image类型
    dataset = dataset.cast_column("ID_image", Image())
    # print(dataset[0])
    return dataset, datadict


def get_data_split(base_path, split_path, split_idx, dataset='FD001', prefix='', condition='', upsample=False,
                   missing_ratio=0., max_tmins=100.0):
    # load data
    if dataset == 'FD001':
        Pdict_list = np.load(base_path + f'/processed_data/FD001_ImageDict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/fd001/result.npy', allow_pickle=True)
        task = "regression"
        num_labels = 2
    elif dataset == 'FD002':
        Pdict_list = np.load(base_path + f'/processed_data/normalized/FD002_ImageDict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/normalized/result.npy', allow_pickle=True)
        task = "regression"
        num_labels = 2
    elif dataset == 'FD003':
        Pdict_list = np.load(base_path + f'/processed_data/FD003_ImageDict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/fd003/result.npy', allow_pickle=True)
        task = "regression"
        num_labels = 8
    elif dataset == 'FD004':
        Pdict_list = np.load(base_path + f'/processed_data/FD004_ImageDict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/fd004/result.npy', allow_pickle=True)
        task = "regression"
        num_labels = 8
    elif "Classification" in base_path:
        Pdict_list = np.load(base_path + f'/processed_data/ImageDict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        task = "classification"
    elif "Regression" in base_path:
        Pdict_list = np.load(base_path + f'/processed_data/ImageDict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        task = "regression"

    y = [item['remain_life'] for item in arr_outcomes if 'is_full_cycle' in item]
    y = np.array(y)
    y = y.reshape((-1, 1))
    # y = arr_outcomes[:]["is_full_cycle"].reshape((-1, 1))
    if task == "classification":
        y = y.astype(np.int32)
    elif task == "regression":
        y = y.astype(np.float32)
    # q = np.load(base_path + split_path, allow_pickle=True)
    idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)
    # extract train/val/test examples
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    # upsampling the training dataset
    if upsample:
        ytrain = y[idx_train]
        idx_0 = np.where(ytrain == 0)[0]
        idx_1 = np.where(ytrain == 1)[0]
        n0, n1 = len(idx_0), len(idx_1)
        print(n0, n1)
        if n0 > n1:
            idx_1 = random.choices(idx_1, k=n0)
        else:
            idx_0 = random.choices(idx_0, k=n1)
        # make sure positive and negative samples are placed next to each other
        random.shuffle(idx_0)
        random.shuffle(idx_1)
        upsampled_train_idx = list(chain.from_iterable(zip(idx_0, idx_1)))
        Ptrain = Ptrain[upsampled_train_idx]
        ytrain = ytrain[upsampled_train_idx]

    # only remove part of params in val, test set
    train_dataset, train_datadict = load_image(Ptrain, ytrain, base_path, split_idx, prefix,
                                               condition, missing_ratio=0., max_tmins=max_tmins)
    val_dataset, val_datadict = load_image(Pval, yval, base_path, split_idx, prefix, condition,
                                           missing_ratio, max_tmins=max_tmins)
    test_dataset, test_datadict = load_image(Ptest, ytest, base_path, split_idx, prefix, condition,
                                             missing_ratio, max_tmins=max_tmins)

    return train_dataset, val_dataset, test_dataset, ytrain, yval, ytest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD002', choices=['FD001', 'FD002', 'FD003', 'FD004'])  #
    parser.add_argument('--withmissingratio', default=False,
                        help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0')  #
    parser.add_argument('--feature_removal_level', type=str, default='no_removal',
                        choices=['no_removal', 'set', 'sample'],
                        help='use this only when splittype==random; otherwise, set as no_removal')  #
    args = parser.parse_args()

    dataset = args.dataset
    print('Dataset used: ', dataset)

    if dataset == 'FD001':
        base_path = '../../dataset/CMAPASS_fzy'
    elif dataset == 'FD002':
        base_path = '../../dataset/CMAPASS_fzy'
    elif dataset == 'FD003':
        base_path = '../../dataset/CMAPASS_fzy'
    elif dataset == 'FD004':
        base_path = '../../dataset/CMAPASS_fzy'

    feature_removal_level = args.feature_removal_level  # 'set' for fixed, 'sample' for random sample

    """While missing_ratio >0, feature_removal_level is automatically used"""
    if args.withmissingratio == True:
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        missing_ratios = [0]
    print('missing ratio list', missing_ratios)

    n_splits = 5
    subset = False
    for k in range(n_splits):
        split_idx = k + 1
        print('Split id: %d' % split_idx)
        if dataset == 'FD001':
            if subset == True:
                split_path = '/splits/FD001/FD001_split_subset' + str(split_idx) + '.npy'
            else:
                split_path = '/splits/FD001/FD001_split' + str(split_idx) + '.npy'
        elif dataset == 'FD002':
            split_path = '/splits/FD002/FD002_split' + str(split_idx) + '.npy'
        elif dataset == 'FD003':
            split_path = '/splits/FD003/FD003_split' + str(split_idx) + '.npy'
        elif dataset == 'FD004':
            split_path = '/splits/FD004/FD004_split' + str(split_idx) + '.npy'

        # prepare the data:
        Ptrain, Pval, Ptest, label2id, id2label, ytrain, yval, ytest = get_data_split(base_path, split_path, split_idx)
        print(len(Ptrain), len(Pval), len(Ptest))
