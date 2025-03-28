import os.path
from Global_Variable import *
from Definition import DataPoint, Trajectory
import pandas as pd
import numpy as np
import math
import re
import pywt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def Mercator(longitude, latitude):
    # 墨卡托投影,经纬度转成x,y轴坐标
    longitude = float(longitude)
    latitude = float(latitude)
    x = longitude * 20037508.342789 / 180
    y = math.log(math.tan((90 + latitude) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34789 / 180
    return x, y


def Reverse_Mercator(x, y):
    # x,y轴坐标转为经纬度
    x = float(x)
    y = float(y)
    longitude = x * 180 / 20037508.342789
    y = y * 180 / 20037508.34789
    latitude = np.arctan(np.exp((math.pi / 180) * y)) * 360 / math.pi - 90
    return longitude, latitude


def wavelet_transform(x):
    l = len(x)
    cA, cD = pywt.dwt(x, 'sym2')
    x = pywt.idwt(cA, None, 'sym2')
    return x[0:l]


def Singular_Spectrum_Analysis(x):
    # step1 嵌入
    windowLen = 24  # 嵌入窗口长度
    seriesLen = len(x)  # 序列长度
    K = seriesLen - windowLen + 1
    X = np.zeros((windowLen, K))
    for i in range(K):
        X[:, i] = x[i:i + windowLen]

    # step2: svd分解， U和sigma已经按升序排序
    U, sigma, VT = np.linalg.svd(X, full_matrices=False)

    for i in range(VT.shape[0]):
        VT[i, :] *= sigma[i]
    A = VT

    # 重组
    rec = np.zeros((windowLen, seriesLen))
    for i in range(windowLen):
        for j in range(windowLen - 1):
            for m in range(j + 1):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (j + 1)
        for j in range(windowLen - 1, seriesLen - windowLen + 1):
            for m in range(windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= windowLen
        for j in range(seriesLen - windowLen + 1, seriesLen):
            for m in range(j - seriesLen + windowLen, windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (seriesLen - j)

    # rrr = np.sum(rec[0:5, :], axis=0)  # 选择重构的部分，这里选了全部

    plt.figure()
    for i in range(10):
        ax = plt.subplot(5, 2, i + 1)
        ax.plot(rec[i, :])
    plt.show()
    return rec[0, :]


def format_conversion(data, file_format=None, year_range=None, latitude_range=None, longitude_range=None,
                      Data_processing=True, PCA_precess=False, ouput_file=None):
    '''
    :param data: 输入的数据
    :param file_format: 筛选后转换符合要求的数据的格式
    :param year_range:年限[start_year, end_year]
    :param latitude_range:纬度范围[start_latitude, end_latitude]
    :param longitude_range:经度范围[start_longitude, end_longitude]
    :param Data_processing:是否需要处理数据，默认需要处理，如果不需要，则false，如果是true需要处理数据则需要把数据进行存储方便下次直接读取
    :param PCA_precess(废弃):是否做数据降维处理，主成分分析(根据经验分析振幅，与速度，影响半径有强相关性，看作一个整体，因此我们进行局部降维)
    :param ouput_file:结果输出文件
    :return:轨迹字典，key是轨迹id,value是对应轨迹
    '''
    tr_dic = {}  # 用于存储数据csv格式
    if file_format == 'NC':
        for i in range(data['time'].shape[0]):
            if Data_processing:
                if latitude_range is not None:
                    if float(data['latitude'][i].data) < latitude_range[0] or \
                            float(data['latitude'][i].data) > latitude_range[1] or \
                            float(data['longitude'][i].data) < longitude_range[0] or \
                            float(data['longitude'][i].data) > longitude_range[1]:
                        continue
            dp = DataPoint(id=i, TrackId=int(data['track'][i]), time=int(data['time'][i]),
                           amplitude=float(data['amplitude'][i]), latitude=float(data['latitude'][i].data),
                           longitude=float(data['longitude'][i].data), speed_average=float(data['speed_average'][i]),
                           effective_radius=float(data['effective_radius'][i]))
            if year_range is not None:
                if dp.year < year_range[0] or dp.year > year_range[1]:
                    continue
            if dp.TrackId not in tr_dic.keys():
                tr_dic[dp.TrackId] = Trajectory(dp.TrackId)
            tr_dic[dp.TrackId].add_DataPoint(dp, Data_processing)
    elif file_format == 'TXT':
        # 文件格式：[0latitude,1longitude,2amplitude,3cyclonic_type,4observation_flag,5observation_number,6speed_average,7speed_average,8time,9track]
        for i in range(1, len(data)):  # 第一排是表头
            data_list = data[i].split(',')
            if Data_processing:
                latitude = float(data_list[0])
                longitude = float(data_list[1])
                if latitude < latitude_range[0] or latitude > latitude_range[1] or longitude < longitude_range[0] \
                        or longitude > longitude_range[1] or int(data_list[3]) == 1:
                    continue
            dp = DataPoint(id=i, TrackId=int(data_list[9][:-1]), time=float(data_list[8]),
                           amplitude=float(data_list[2]), latitude=float(data_list[0]), longitude=float(data_list[1]),
                           speed_average=float(data_list[6]), effective_radius=float(data_list[7]))
            if dp.TrackId not in tr_dic.keys():
                tr_dic[dp.TrackId] = Trajectory(dp.TrackId)
            tr_dic[dp.TrackId].add_DataPoint(dp, Data_processing)
    elif file_format == 'CSV':
        if PCA_precess:
            pca_list = []
            for i in data.columns:
                if 'Components_' in i:
                    pca_list.append(i)
                elif 'Components_ASE' in i:
                    pca_list.append(i)
        if not Data_processing:
            s = []
            b = []
            time_scope = []
            for column in data.columns:  # 遍历每一列名
                t = re.compile(r'[\d+ \d+]').search(column)
                if t is not None:
                    time_scope.append(column)
            for i in range(2, 9):
                temp = data.iloc[:, i]
                # s.append(np.nanmin(temp.replace(0, np.nan)))  # v末尾点取0
                s.append(min(temp))
                b.append(max(temp))
            # 经纬度需要转成xy
            s[1], s[2] = Mercator(longitude=s[1], latitude=s[2])
            b[1], b[2] = Mercator(longitude=b[1], latitude=b[2])
        for i in range(data.shape[0]):
            if Data_processing:
                if latitude_range is not None:
                    if data.at[i, 'latitude'] < latitude_range[0] or data.at[i, 'latitude'] > latitude_range[1] or \
                      data.at[i, 'longitude'] < longitude_range[0] or data.at[i, 'longitude'] > longitude_range[1]:
                        continue
                dp = DataPoint(id=i, TrackId=int(data.at[i, 'track']), time=int(data.at[i, 'time']),
                               amplitude=float(data.at[i, 'amplitude']), latitude=float(data.at[i, 'latitude']),
                               longitude=float(data.at[i, 'longitude']),
                               speed_average=float(data.at[i, 'speed_average']),
                               effective_radius=float(data.at[i, 'effective_radius']))
            else:
                dp = DataPoint(id=i, TrackId=int(data.at[i, 'track']), time=int(data.at[i, 'time']),
                               amplitude=float(data.at[i, 'amplitude']),
                               latitude=float(data.at[i, 'latitude']), longitude=float(data.at[i, 'longitude']),
                               speed_average=float(data.at[i, 'speed_average']),
                               effective_radius=float(data.at[i, 'effective_radius']),
                               velocity=float(data.at[i, 'velocity']),
                               angle=float(data.at[i, 'angle']))
                # [0TrackId, 1timestamp, 2amplitude, 3x_coordinate, 4y_coordinate, 5speed_average,
                # 6effective_radius, 7velocity, 8angle]
                dp.array = [dp.TrackId, dp.timestamp, (dp.amplitude - s[0]) / (b[0] - s[0]),
                             (dp.x_coordinate - s[1]) / (b[1] - s[1]), (dp.y_coordinate - s[2]) / (b[2] - s[2]),
                             (dp.speed_average - s[3]) / (b[3] - s[3]), (dp.effective_radius - s[4]) / (b[4] - s[4]),
                             (dp.velocity - s[5]) / (b[5] - s[5]) if dp.velocity - s[5] >= 0 else 0,
                             (dp.angle - s[6]) / (b[6] - s[6]) if dp.angle - s[6] >= 0 else 0]
                if not Data_processing and 'pca_list' in locals().keys():
                    pca_data = []
                    for j in pca_list:
                        pca_data.append(data.at[i, j])
                    dp.add_PCA(pca_data)
            if Data_processing and year_range is not None:
                if dp.year < year_range[0] or dp.year > year_range[1]:
                    continue
            if dp.TrackId not in tr_dic.keys():
                tr_dic[dp.TrackId] = Trajectory(dp.TrackId)
                if len(time_scope) != 0:
                    tr_dic[dp.TrackId].time_membership = data.at[i, time_scope[0]]
            tr_dic[dp.TrackId].add_DataPoint(dp, Data_processing)
    if Data_processing:
        # 删除轨迹点数目小于25的轨迹
        del_list = []
        for i in tr_dic.keys():
            if tr_dic[i].destroy:  # 轨迹存在缺失段
                tr_dic[i].missing_value = np.array(tr_dic[i].missing_value, dtype=int)
                if np.max(list((tr_dic[i].missing_value[:, 1] - tr_dic[i].missing_value[:, 0]).flat)) < 6:
                    Interpolation(tr_dic[i])
                else:  # 缺失值过多丢弃
                    del_list.append(i)
                    print("缺失值过多%d轨迹数据舍弃" % tr_dic[i].TrackId)
            if tr_dic[i].NumPoint < 25:  # 轨迹段个数少
                del_list.append(i)
        for i in set(del_list):
            del tr_dic[i]
        p_array = []
        # 对每个轨迹利用小波变换进行降噪
        # 计算每个轨迹的速度和加速度
        # [self.TrackId, self.timestamp, self.amplitude, self.longitude, self.latitude, self.speed_average,
        #                 self.effective_radius, self.velocity, self.angle]
        for i in tr_dic.keys():
            for j in tr_dic[i].point_list:
                p_array.append(j.origin_array())
        p_array = np.array(p_array, dtype=np.float64)
        for i in tr_dic.keys():
            for j in range(2, p_array.shape[1]-2):
                p_array[p_array[:, 0] == i, j] = wavelet_transform(p_array[p_array[:, 0] == i, j])  # 小波变换
            point_matrix = p_array[p_array[:, 0] == i, :]
            # 对每个轨迹计算va
            for point_index in range(point_matrix.shape[0]):
                if point_index == point_matrix.shape[0]-1:
                    # 最后一个点不需要计算
                    continue
                x1, y1 = Mercator(point_matrix[point_index, 3], point_matrix[point_index, 4])
                x2, y2 = Mercator(point_matrix[point_index+1, 3], point_matrix[point_index+1, 4])
                point_matrix[point_index, 7] = math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2)) / (
                        point_matrix[point_index+1, 1] - point_matrix[point_index, 1])
                if x1 == x2:
                    angle = math.pi / 2 if (y2 - y1) > 0 else math.pi + math.pi / 2
                else:
                    if (y2 - y1) >= 0 and (x2 - x1) > 0:
                        angle = np.arctan((y2 - y1) / (x2 - x1))
                    elif (y2 - y1) >= 0 and (x2 - x1) < 0:
                        angle = math.pi + np.arctan((y2 - y1) / (x2 - x1))
                    elif (y2 - y1) < 0 and (x2 - x1) < 0:
                        angle = math.pi + np.arctan((y2 - y1) / (x2 - x1))
                    elif (y2 - y1) < 0 and (x2 - x1) > 0:
                        angle = np.arctan((y2 - y1) / (x2 - x1)) + math.pi * 2
                point_matrix[point_index, 8] = angle * 180 / math.pi
            p_array[p_array[:, 0] == i, :] = point_matrix
        df = pd.DataFrame(data=p_array, columns=['track', 'time', 'amplitude', 'longitude', 'latitude',
                                                 'speed_average', 'effective_radius', 'velocity', 'angle'])
        if PCA_precess:
            # 添加主成分分析后的数据列
            # pca = PCA(data[['amplitude', 'longitude', 'latitude', 'speed_average', 'effective_radius', 'velocity',
            #                 'angle']])
            # principal_col = pca.principal_comps()
            # characteristic_value = pca.characteristic_value
            # for i in range(principal_col.shape[1]):
            #     df.insert(9 + i, 'Components_' + str(i) + ':' + str(characteristic_value[i]), principal_col[:, i])
            from sklearn.decomposition import PCA
            # PCA降维前需要先做数据标准化
            scaler = StandardScaler()
            normed = scaler.fit_transform(df[['amplitude', 'speed_average', 'effective_radius']])
            pca = PCA(n_components=1).fit_transform(normed)
            df.insert(9, 'Components_ASE:', pca)
        # 首先对数据进行归一化， 对7为数据进行降维
        # mms = MinMaxScaler()
        # normed = mms.fit_transform(df[['amplitude', 'longitude', 'latitude', 'speed_average', 'effective_radius',
        #                                'velocity', 'angle']])
        # import umap
        # normed = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.1).fit_transform(normed)
        # df.insert(df.shape[1], 'UMAP0:', normed[:, 0])
        # df.insert(df.shape[1], 'UMAP1:', normed[:, 1])
        if ouput_file is None:
            ouput_file = os.path.join(DATA_PATH, 'Eddy_trajectory_nrt_3.2exp_cyclonic_20180101_20220210.csv')
        df.to_csv(ouput_file, index=False)
        print('{} file saved！'.format(ouput_file))
    # else:
    #     # 对数据特征进行归一化
    #     p_array = []
    #     for i in tr_dic.keys():
    #         for j in tr_dic[i].point_list:
    #             p_array.append([j.TrackId, j.timestamp, j.amplitude, j.x_coordinate, j.y_coordinate, j.speed_average,
    #                             j.effective_radius, j.velocity, j.angle])
    #
    print('Data read over.')
    return tr_dic


def Interpolation(track):
    # 对数据进行线性插值
    if len(track.missing_value) == 0:
        return None
    print("interpolation of {} track ".format(track.TrackId))
    point_matrix = []
    for p in track.point_list:
        point_matrix.append(p.origin_array()[0:7])
    point_matrix = np.array(point_matrix, dtype=np.float64)
    from scipy.interpolate import UnivariateSpline
    amplitude_func = UnivariateSpline(point_matrix[:, 1], point_matrix[:, 2])  # 线性拟合
    longitude_func = UnivariateSpline(point_matrix[:, 1], point_matrix[:, 3])
    latitude_func = UnivariateSpline(point_matrix[:, 1], point_matrix[:, 4])
    speed_average_func = UnivariateSpline(point_matrix[:, 1], point_matrix[:, 5])
    effective_radius_func = UnivariateSpline(point_matrix[:, 1], point_matrix[:, 6])
    for Missing_range in track.missing_value:  # 遍历缺失段
        for miss_point in range(Missing_range[0]+1, Missing_range[1]):
            amplitude = float(amplitude_func(miss_point))
            longitude = float(longitude_func(miss_point))
            latitude = float(latitude_func(miss_point))
            speed_average = float(speed_average_func(miss_point))
            effective_radius = float(effective_radius_func(miss_point))
            temp_point = DataPoint(0, track.TrackId, miss_point, amplitude, latitude, longitude, speed_average,
                                   effective_radius)
            track.point_list.insert(np.where(point_matrix[:, 1] == Missing_range[0])[0][0]+1, temp_point)
            track.NumPoint += 1


class ProcessingData:  # 数据模型预处理
    def __init__(self, file_format, data_processing=True, year_range=None, longitude_range=None, latitude_range=None,
                 output_file=None):
        self.file_format = file_format  # 文件格式"NC","TXT","CSV"
        # 预处理：时空的删选，计算va，删除数量少和缺失过多的轨迹，缺失值较少的线性插值，小波变换
        self.data_processing = data_processing  # 数据是否需要预处理
        self.year_range = year_range  # 年限[start_year, end_year]
        self.latitude_range = latitude_range  # 纬度范围[start_latitude, end_latitude]
        self.longitude_range = longitude_range  # 经度范围[start_longitude, end_longitude]
        self.output_file = output_file  # 输出文件保存路径
        self.minimum_list = []  # 最小值列表——针对中尺度涡旋7维数据
        self.maximum_list = []  # 最大值列表

    def fit(self, data):  # 数据输入
        print("Start data processing.")
        tr_dic = {}  # 用于返回轨迹字典
        if self.file_format == 'NC':
            self.read_nc(data, tr_dic)
        elif self.file_format == 'TXT':
            self.read_txt(data, tr_dic)
        elif self.file_format == 'CSV':
            self.read_csv(data, tr_dic)
        if self.data_processing:
            # 删除轨迹点数目小于25的轨迹
            del_list = []
            for i in tr_dic.keys():
                if tr_dic[i].destroy:  # 轨迹存在缺失段
                    tr_dic[i].missing_value = np.array(tr_dic[i].missing_value, dtype=int)
                    if np.max(list((tr_dic[i].missing_value[:, 1] - tr_dic[i].missing_value[:, 0]).flat)) < 6:
                        Interpolation(tr_dic[i])
                    else:  # 缺失值过多丢弃
                        del_list.append(i)
                        print("缺失值过多%d轨迹数据舍弃" % tr_dic[i].TrackId)
                if tr_dic[i].NumPoint < 25:  # 轨迹段个数少
                    del_list.append(i)
            for i in set(del_list):
                del tr_dic[i]
            p_array = []
            # 对每个轨迹利用小波变换进行降噪
            # 计算每个轨迹的速度和加速度
            # [self.TrackId, self.timestamp, self.amplitude, self.longitude, self.latitude, self.speed_average,
            #                 self.effective_radius, self.velocity, self.angle]
            for i in tr_dic.keys():
                for j in tr_dic[i].point_list:
                    p_array.append(j.origin_array())
            p_array = np.array(p_array, dtype=np.float64)
            for i in tr_dic.keys():
                for j in range(2, p_array.shape[1] - 2):
                    p_array[p_array[:, 0] == i, j] = wavelet_transform(p_array[p_array[:, 0] == i, j])  # 小波变换
                point_matrix = p_array[p_array[:, 0] == i, :]
                # 对每个轨迹计算va
                for point_index in range(point_matrix.shape[0]):
                    if point_index == point_matrix.shape[0] - 1:
                        # 最后一个点不需要计算
                        continue
                    x1, y1 = Mercator(point_matrix[point_index, 3], point_matrix[point_index, 4])
                    x2, y2 = Mercator(point_matrix[point_index + 1, 3], point_matrix[point_index + 1, 4])
                    point_matrix[point_index, 7] = math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2)) / (
                            point_matrix[point_index + 1, 1] - point_matrix[point_index, 1])
                    if x1 == x2:
                        angle = math.pi / 2 if (y2 - y1) > 0 else math.pi + math.pi / 2
                    else:
                        if (y2 - y1) >= 0 and (x2 - x1) > 0:
                            angle = np.arctan((y2 - y1) / (x2 - x1))
                        elif (y2 - y1) >= 0 and (x2 - x1) < 0:
                            angle = math.pi + np.arctan((y2 - y1) / (x2 - x1))
                        elif (y2 - y1) < 0 and (x2 - x1) < 0:
                            angle = math.pi + np.arctan((y2 - y1) / (x2 - x1))
                        elif (y2 - y1) < 0 and (x2 - x1) > 0:
                            angle = np.arctan((y2 - y1) / (x2 - x1)) + math.pi * 2
                    point_matrix[point_index, 8] = angle * 180 / math.pi
                p_array[p_array[:, 0] == i, :] = point_matrix
            df = pd.DataFrame(data=p_array, columns=['track', 'time', 'amplitude', 'longitude', 'latitude',
                                                     'speed_average', 'effective_radius', 'velocity', 'angle'])
            if self.ouput_file is None:
                ouput_file = os.path.join(DATA_PATH, 'Eddy_trajectory_nrt_3.2exp_cyclonic_20180101_20220210.csv')
            df.to_csv(ouput_file, index=False)
            print('{} file saved！'.format(ouput_file))
        print("Data read over.")
        return tr_dic

    def read_nc(self, nc_data, tr_dic):  # 读取nc文件
        for i in range(nc_data['time'].shape[0]):
            if self.data_processing:
                if self.latitude_range is not None:
                    if float(nc_data['latitude'][i].data) < self.latitude_range[0] or \
                            float(nc_data['latitude'][i].data) > self.latitude_range[1] or \
                            float(nc_data['longitude'][i].data) < self.longitude_range[0] or \
                            float(nc_data['longitude'][i].data) > self.longitude_range[1]:
                        continue
            dp = DataPoint(id=i, TrackId=int(nc_data['track'][i]), time=int(nc_data['time'][i]),
                           amplitude=float(nc_data['amplitude'][i]), latitude=float(nc_data['latitude'][i].data),
                           longitude=float(nc_data['longitude'][i].data),
                           speed_average=float(nc_data['speed_average'][i]),
                           effective_radius=float(nc_data['effective_radius'][i]))
            if self.year_range is not None:
                if dp.year < self.year_range[0] or dp.year > self.year_range[1]:
                    continue
            if dp.TrackId not in tr_dic.keys():
                tr_dic[dp.TrackId] = Trajectory(dp.TrackId)
            tr_dic[dp.TrackId].add_DataPoint(dp, self.data_processing)

    def read_txt(self, txt_data, tr_dic):  # 读txt文件
        # 文件要求格式：[0latitude,1longitude,2amplitude,3cyclonic_type,4observation_flag,5observation_number,6speed_average,7speed_average,8time,9track]
        for i in range(1, len(txt_data)):  # 第一排是表头
            data_list = txt_data[i].split(',')
            if self.data_processing:
                latitude = float(data_list[0])
                longitude = float(data_list[1])
                if latitude < self.latitude_range[0] or latitude > self.latitude_range[1] or \
                        longitude < self.longitude_range[0] or longitude > self.longitude_range[1] or \
                        int(data_list[3]) == 1:
                    continue
            dp = DataPoint(id=i, TrackId=int(data_list[9][:-1]), time=float(data_list[8]),
                           amplitude=float(data_list[2]), latitude=float(data_list[0]),
                           longitude=float(data_list[1]),
                           speed_average=float(data_list[6]), effective_radius=float(data_list[7]))
            if dp.TrackId not in tr_dic.keys():
                tr_dic[dp.TrackId] = Trajectory(dp.TrackId)
            tr_dic[dp.TrackId].add_DataPoint(dp, self.data_processing)

    def read_csv(self, csv_data, tr_dic):  # 读csv文件（主要就是这种文件的读取）
        if not self.data_processing:
            s = []
            b = []
            time_scope = []  # 确定时间段
            for column in csv_data.columns:  # 遍历每一列名
                t = re.compile(r'[\d+ \d+]').search(column)
                if t is not None:
                    time_scope.append(column)
            for i in range(2, 9):
                temp = csv_data.iloc[:, i]
                s.append(min(temp))
                b.append(max(temp))
            # 经纬度需要转成xy
            s[1], s[2] = Mercator(longitude=s[1], latitude=s[2])
            b[1], b[2] = Mercator(longitude=b[1], latitude=b[2])
            self.minimum_list = s
            self.maximum_list = b
        for i in range(csv_data.shape[0]):
            if self.data_processing:
                if self.latitude_range is not None:
                    if csv_data.at[i, 'latitude'] < self.latitude_range[0] or \
                            csv_data.at[i, 'latitude'] > self.latitude_range[1] or \
                            csv_data.at[i, 'longitude'] < self.longitude_range[0] or \
                            csv_data.at[i, 'longitude'] > self.longitude_range[1]:
                        continue
                dp = DataPoint(id=i, TrackId=int(csv_data.at[i, 'track']), time=int(csv_data.at[i, 'time']),
                               amplitude=float(csv_data.at[i, 'amplitude']), latitude=float(csv_data.at[i, 'latitude']),
                               longitude=float(csv_data.at[i, 'longitude']),
                               speed_average=float(csv_data.at[i, 'speed_average']),
                               effective_radius=float(csv_data.at[i, 'effective_radius']))
            else:
                dp = DataPoint(id=i, TrackId=int(csv_data.at[i, 'track']), time=int(csv_data.at[i, 'time']),
                               amplitude=float(csv_data.at[i, 'amplitude']),
                               latitude=float(csv_data.at[i, 'latitude']), longitude=float(csv_data.at[i, 'longitude']),
                               speed_average=float(csv_data.at[i, 'speed_average']),
                               effective_radius=float(csv_data.at[i, 'effective_radius']),
                               velocity=float(csv_data.at[i, 'velocity']),
                               angle=float(csv_data.at[i, 'angle']))
                # [0TrackId, 1timestamp, 2amplitude, 3x_coordinate, 4y_coordinate, 5speed_average,
                # 6effective_radius, 7velocity, 8angle]
                dp.array = [dp.TrackId, dp.timestamp, (dp.amplitude - s[0]) / (b[0] - s[0]),
                            (dp.x_coordinate - s[1]) / (b[1] - s[1]), (dp.y_coordinate - s[2]) / (b[2] - s[2]),
                            (dp.speed_average - s[3]) / (b[3] - s[3]), (dp.effective_radius - s[4]) / (b[4] - s[4]),
                            (dp.velocity - s[5]) / (b[5] - s[5]) if dp.velocity - s[5] >= 0 else 0,
                            (dp.angle - s[6]) / (b[6] - s[6]) if dp.angle - s[6] >= 0 else 0]
            if self.data_processing and self.year_range is not None:
                if dp.year < self.year_range[0] or dp.year > self.year_range[1]:
                    continue
            if dp.TrackId not in tr_dic.keys():
                tr_dic[dp.TrackId] = Trajectory(dp.TrackId)
                if len(time_scope) != 0:
                    tr_dic[dp.TrackId].time_membership = csv_data.at[i, time_scope[0]]
            tr_dic[dp.TrackId].add_DataPoint(dp, self.data_processing)

    def normalization(self, multidata):  # 数据归一化
        if len(multidata) != 7:
            print("{} is not 7 dimensional data,so fail normalization.".format(multidata))
            return multidata
        multidata[2], multidata[3] = Mercator(multidata[2], multidata[3])
        return (multidata - self.maximum_list) / (self.maximum_list - self.minimum_list)

    def restore(self, multidata):  # 还原数据
        if len(multidata) != 7:
            print("{} is not 7 dimensional data,so fail normalization.".format(multidata))
            return multidata
        output = multidata * (np.array(self.maximum_list, np.float64) - np.array(self.minimum_list, np.float64)
                              ) + np.array(self.minimum_list, np.float64)
        # 墨卡托投影转换回经纬度
        output[1], output[2] = Reverse_Mercator(output[1], output[2])
        return output


if __name__ == '__main__':
    file_name = os.path.join(r'D:\Trajectory_analysis\Data', 'Eddy_trajectory_nrt_3.2exp_cyclonic_20180101_20220210.csv')
    data = pd.read_csv(file_name)
    tra_dic = format_conversion(data, file_format='CSV', Data_processing=True)
