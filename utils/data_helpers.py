import os
import random


def auto_gen_2d_classification_data(n=100, min_num=0.0, max_num=1.0):
    data = []
    avg_num = (min_num + max_num) / 2.0
    for id_ in range(n):
        ft1 = random.uniform(min_num, max_num)
        ft2 = random.uniform(min_num, max_num)
        if (ft1 < avg_num) & (ft2 < avg_num):
            label = 0
            data.append([ft1, ft2, label])
        elif (ft1 > avg_num) & (ft2 > avg_num):
            label = 1
            data.append([ft1, ft2, label])
    return data


def auto_gen_and_save_classification_data(n=100, file_path=''):
    if (n > 0) & (len(file_path) > 0):
        data = auto_gen_2d_classification_data(n=n)
        f = open(file_path, 'w')
        for sample_ in data:
            sample_str = map(str, sample_)
            f.write('\t'.join(sample_str))
            f.write('\n')
        f.close()


def read_data(file_path):
    data = []
    if os.path.isfile(file_path):
        f = open(file_path, 'r')
        samples = list(f.readlines())
        f.close()
        samples = [sample.strip() for sample in samples if len(sample.strip()) > 0]
        for sample in samples:
            sample_arr = sample.split('\t')
            feature_arr = map(float, sample_arr[:-1])
            label = int(sample_arr[-1])
            data.append((feature_arr, label))
    return data


def plot_data(file_path):
    data = read_data(file_path)
    if len(data) > 0:
        xs = []
        ys = []
        labels = []
        for sample in data:
            feature_arr = sample[0]
            label = sample[1]
            xs.append(feature_arr[0])
            ys.append(feature_arr[1])
            labels.append(label)

