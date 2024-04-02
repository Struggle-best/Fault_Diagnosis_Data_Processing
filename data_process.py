import os
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.io as scio
from scipy.io import loadmat


# ---------------------公用---------------------
def split_data(data, split_rate, shuffle=None):
    # 拆分数据集为 train, valid, test
    length = len(data)
    num1 = int(length * split_rate[0])
    num2 = int(length * split_rate[1])
    if shuffle is None or shuffle is True:
        index1 = random.sample(range(num1), num1)
        index2 = random.sample(range(num2), num2)
    else:
        index1 = range(num1)
        index2 = range(num2)

    train = data[index1]
    data = np.delete(data, index1, axis=0)
    valid = data[index2]
    test = np.delete(data, index2, axis=0)
    return train, valid, test

# ---------------------公用---------------------
def data_label_onehot(data, window_size):
    onehot_encoder = preprocessing.OneHotEncoder(sparse_output=False)
    Data = data[:, 0:window_size]
    Label = data[:, window_size]
    Label = Label.reshape(len(Label), 1)
    Label = onehot_encoder.fit_transform(Label)
    return Data, Label

# ---------------------case---------------------
def case_open_data(bath_path, key_num):
    path = bath_path + str(key_num) + ".mat"
    str1 = "X" + "%03d" % key_num + "_DE_time"  # 将数字 key_num 格式化成 字符串
    data = scio.loadmat(path)
    data = data[str1]
    return data

# ---------------------case---------------------
def case_deal_data(data, length, label):
    data = np.reshape(data, (-1))
    num = len(data) // length
    data = data[0:num * length]
    data = np.reshape(data, (num, length))

    min_max_scaler = preprocessing.MinMaxScaler()

    data = min_max_scaler.fit_transform(np.transpose(data, [1, 0]))
    data = np.transpose(data, [1, 0])
    label = np.ones((num, 1)) * label
    return np.column_stack((data, label))

# --------------------- XJUT ---------------------
def xjut_data_load(filename, label, window_size):
    fl = pd.read_csv(filename)
    fl = fl["Horizontal_vibration_signals"]
    fl = fl.values
    fl = fl.reshape(-1)
    data1 = []
    for i in range(0, len(fl) - (len(fl) % window_size), window_size):
        da = fl[i:i + window_size]
        # print(label)
        da = np.append(da, label)
        data1.append(da)
    return data1

# --------------------- pu ---------------------
def pu_data_load(filename, name, label, window_size):
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  # Take out the data
    fl = fl.reshape(-1)
    data1 = []
    for i in range(0, len(fl) - (len(fl) % window_size), window_size):
        da = fl[i:i + window_size]
        da = np.append(da, label)
        data1.append(da)

    return data1

dataset_path = {
    'case10_1': '../datasets/case_dataset/Normal_Baseline_Data/',  # path of Normal Baseline Data
    'case10_2': '../datasets/case_dataset/12k_DriveEnd_Bearing_Fault_Data/',  # path of 12k Drive End Bearing Fault Data
    'xjtu15': '../datasets/XJTU_Bearing_Datasets',
    'mftp15': '../datasets/MFTP_dataset',
    'pu13': '../datasets/pu_Bearing_dataset',
    'seu10': '../datasets/SEU_dataset/gearbox/bearingset'
}
# dataset_path[dataset_name]


def load_dataset(window_size, split_rate, dataset_name, shuffle=None):
    if dataset_name == 'case10':
        fault_diameter = [0.007, 0.014, 0.021]
        hp = [0, 1, 2, 3]
        num = 100

        bath_path1 = dataset_path[dataset_name + '_1']
        bath_path2 = dataset_path[dataset_name + '_2']
        data_list = []
        file_list = np.array([[105, 118, 130, 106, 119, 131, 107, 120, 132, 108, 121, 133],  # 0.007
                              [169, 185, 197, 170, 186, 198, 171, 187, 199, 172, 188, 200],  # 0.014
                              [209, 222, 234, 210, 223, 235, 211, 224, 236, 212, 225, 237]])  # 0.021
        label = 0

        for i in hp:
            normal_data = case_open_data(bath_path1, 97 + i)
            data = case_deal_data(normal_data, window_size, label=label)
            data_list.append(data)

        # hp = [0, 1, 2, 3]
        for i in fault_diameter:
            for j in hp:
                inner_num = file_list[int(i / 0.007 - 1), 3 * j]  # 内圈
                ball_num = file_list[int(i / 0.007 - 1), 3 * j + 1]  # 滚动体
                outer_num = file_list[int(i / 0.007 - 1), 3 * j + 2]  # 外圈

                inner_data = case_open_data(bath_path2, inner_num)
                inner_data = case_deal_data(inner_data, window_size, label + 1)
                data_list.append(inner_data)

                ball_data = case_open_data(bath_path2, ball_num)
                ball_data = case_deal_data(ball_data, window_size, label + 4)
                data_list.append(ball_data)

                outer_data = case_open_data(bath_path2, outer_num)
                outer_data = case_deal_data(outer_data, window_size, label + 7)
                data_list.append(outer_data)

            label = label + 1

        # 保持每个类的数据数量相同,遍历data_list 找出其最小长度
        num_list = []
        for i in data_list:
            num_list.append(len(i))
        min_num = min(num_list)

        if num > min_num:
            print("The number of each class overflow, the maximum number is：%d" % min_num)

        min_num = min(num, min_num)
        train = []
        valid = []
        test = []

        for data in data_list:
            data = data[0:min_num, :]
            a, b, c = split_data(data, split_rate, shuffle=shuffle)
            train.append(a)
            valid.append(b)
            test.append(c)

        train = np.reshape(train, (-1, window_size + 1))
        valid = np.reshape(valid, (-1, window_size + 1))
        test = np.reshape(test, (-1, window_size + 1))
        if shuffle is None or shuffle is True:
            train = train[random.sample(range(len(train)), len(train))]
            valid = valid[random.sample(range(len(valid)), len(valid))]
            test = test[random.sample(range(len(test)), len(test))]

    elif dataset_name == 'xjtu15':
        label1 = [i for i in range(0, 5)]
        label2 = [i for i in range(5, 10)]
        label3 = [i for i in range(10, 15)]
        root = dataset_path[dataset_name]
        train = []
        val = []
        test = []
        WC = os.listdir(root)  # Three working conditions WC0:35Hz12kN WC1:37.5Hz11kN WC2:40Hz10kN

        datasetname1 = os.listdir(os.path.join(root, WC[0]))
        datasetname2 = os.listdir(os.path.join(root, WC[1]))
        datasetname3 = os.listdir(os.path.join(root, WC[2]))
        for i in range(len(datasetname1)):
            files = os.listdir(os.path.join(root, WC[0], datasetname1[i]))
            data = []
            for ii in [-4, -3, -2, -1]:  # Take the data of the last three CSV files
                path1 = os.path.join(root, WC[0], datasetname1[i], files[ii])
                data1 = xjut_data_load(path1, label=label1[i], window_size=window_size)
                # print(data1.shape)
                data += data1
            result_array1 = np.array(data)
            train_1, valid_1, test_1 = split_data(result_array1, split_rate, shuffle=None)
            train.append(train_1)
            val.append(valid_1)
            test.append(test_1)
        for j in range(len(datasetname2)):
            files = os.listdir(os.path.join(root, WC[1], datasetname2[j]))
            data = []
            for jj in [-4, -3, -2, -1]:
                path2 = os.path.join(root, WC[1], datasetname2[j], files[jj])
                data2 = xjut_data_load(path2, label=label2[j], window_size=window_size)
                data += data2
            result_array2 = np.array(data)
            train_2, valid_2, test_2 = split_data(result_array2, split_rate, shuffle=None)
            train.append(train_2)
            val.append(valid_2)
            test.append(test_2)
        for k in range(len(datasetname3)):
            files = os.listdir(os.path.join(root, WC[2], datasetname3[k]))
            data = []
            for kk in [-4, -3, -2, -1]:
                path3 = os.path.join(root, WC[2], datasetname3[k], files[kk])
                data3 = xjut_data_load(path3, label=label3[k], window_size=window_size)
                data += data3
            result_array3 = np.array(data)
            train_3, valid_3, test_3 = split_data(result_array3, split_rate, shuffle=None)
            train.append(train_3)
            val.append(valid_3)
            test.append(test_3)
        train = np.vstack(train)
        valid = np.vstack(val)
        test = np.vstack(test)

    elif dataset_name == 'mftp15':
        path = dataset_path[dataset_name]
        csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.mat')]
        train = []
        val = []
        test = []
        # 读取每个csv文件的第二列数据
        for idx, file in enumerate(csv_files):
            data = []
            df = loadmat(file)
            data_column = df['bearing']['gs'][0][0].reshape(-1)
            data_column = data_column[:146484]

            for i in range(0, len(data_column) - (len(data_column) % window_size), window_size):
                da = data_column[i:i + window_size]
                # print(idx)
                da = np.append(da, idx)
                data.append(da)
            result_array = np.array(data)
            train_, valid_, test_ = split_data(result_array, split_rate, shuffle=None)
            train.append(train_)
            val.append(valid_)
            test.append(test_)
        # print(train)
        train = np.array(train).transpose((1, 0, 2)).reshape((-1, window_size + 1))
        valid = np.array(val).transpose((1, 0, 2)).reshape((-1, window_size + 1))
        test = np.array(test).transpose((1, 0, 2)).reshape((-1, window_size + 1))

    elif dataset_name == 'pu10':
        root = dataset_path[dataset_name]
        RDBdata = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI14', 'KI16', 'KI17', 'KI18',
                   'KI21']
        label3 = [i for i in range(13)]

        # working condition
        WC = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
        state = WC[0]  # WC[0] can be changed to different working states
        train = []
        val = []
        test = []
        for k in range(len(RDBdata)):
            data = []
            name = state + "_" + RDBdata[k] + "_1"
            path = os.path.join(root, RDBdata[k], RDBdata[k], name + ".mat")
            data1 = pu_data_load(path, label=label3[k], name=name, window_size=window_size)
            data += data1
            result_array = np.array(data)
            train_, valid_, test_ = split_data(result_array, split_rate, shuffle=None)
            train.append(train_)
            val.append(valid_)
            test.append(test_)

        train = np.vstack(train)
        valid = np.vstack(val)
        test = np.vstack(test)

    elif dataset_name == 'seu10':
        path = dataset_path[dataset_name]
        csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]
        train = []
        val = []
        test = []
        # 读取每个csv文件的第二列数据
        for idx, file in enumerate(csv_files):
            data = []
            df = pd.read_csv(file, skiprows=15, usecols=[1], sep='\t')
            data_column = df.iloc[:, 0].values  # numpy.ndarray
            # print(data_column.shape)
            for i in range(0, len(data_column) - (len(data_column) % window_size), window_size):
                da = data_column[i:i + window_size]
                da = np.append(da, idx)
                data.append(da)
            result_array = np.array(data)
            train_, valid_, test_ = split_data(result_array, split_rate, shuffle=None)
            train.append(train_)
            val.append(valid_)
            test.append(test_)
        train = np.array(train).transpose((1, 0, 2)).reshape((-1, window_size + 1))
        valid = np.array(val).transpose((1, 0, 2)).reshape((-1, window_size + 1))
        test = np.array(test).transpose((1, 0, 2)).reshape((-1, window_size + 1))
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    train_data, train_label = data_label_onehot(train, window_size)
    valid_data, valid_label = data_label_onehot(valid, window_size)
    test_data, test_label = data_label_onehot(test, window_size)
    return train_data, train_label, valid_data, valid_label, test_data, test_label


if __name__ == '__main__':
    train_dataset, train_label, val_dataset, val_label, test_dataset, test_label = load_dataset(window_size=1024,
                                                                                                split_rate=[0.7, 0.1,
                                                                                                            0.2],
                                                                                                dataset_name='seu10')

    print(train_dataset.shape)
    print(train_label.shape)
    print(val_dataset.shape)
    print(val_label.shape)
    print(test_dataset.shape)
    print(test_label.shape)
