import time
import math
import numpy as np
import pandas as pd


class DataPoint:
    def __init__(self, id, TrackId, time, amplitude, latitude, longitude, speed_average, effective_radius, velocity=0,
                 angle=0):
        self.id = id  # 数据点的id
        self.TrackId = int(TrackId)  # 该数据点所属轨迹id
        self.timestamp = int(time)  # 数据点的时间戳
        self.year = None
        self.month = None
        self.mday = None
        self.amplitude = amplitude  # 振幅
        self.latitude = latitude  # 纬度
        self.longitude = longitude  # 经度
        self.x_coordinate = 0  # longitude转换为坐标系
        self.y_coordinate = 0  # latitude转换为坐标系
        self.speed_average = speed_average  # 平均速度
        self.effective_radius = effective_radius  # 影响半径
        self.velocity = velocity  # 轨迹数据点的移动速度
        self.angle = angle  # 轨迹数据点的转角
        self.cluster = 0  # 聚类标签
        self.timestamp_conversion()
        self.get_Coordinates(self.longitude, self.latitude)
        self.PCA = []  # PCA降维后的（标准化）数据
        self.array = []

    def timestamp_conversion(self):
        # 输入的time是时间戳的时间，需要进行进行时间戳转换时间
        # time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=14, tm_min=53,
        # tm_sec=57, tm_wday=3, tm_yday=1, tm_isdst=0)
        timeArray = pd.to_datetime('1950-01-01 00:00:00') + pd.Timedelta(str(self.timestamp)+'D')
        self.year = timeArray.year
        self.month = timeArray.month
        self.mday = timeArray.day

    def get_Coordinates(self, longitude, latitude):
        # if 'N' in latitude or 'S' in latitude:
        #     # 输入的是纬度
        #     latitude = float(latitude[:-1]) if 'N' in latitude else float('-' + latitude[:-1])
        # if 'E' in longitude or 'W' in longitude:
        #     # 输入的是经度
        #     longitude = float(longitude[:-1]) if 'E' in longitude else float('-' + longitude[:-1])
        # r = a * math.cos()/math.sqrt(1-(math.exp(2)*pow(math.sin(), 2)))
        # q = math.log(math.tan(math.pi / 4 + latitude / 2) * pow(1 - math.e * math.sin(latitude) / (
        # 1 + math.e * math.sin(latitude)), math.e / 2), math.e)
        # self.x_coordinate = r * longitude
        # self.y_coordinate = r * q
        # self.x_coordinate = longitude * 20037508.342789 / 180
        # self.y_coordinate = math.log(math.tan((90 + latitude) * math.pi / 360)) / (math.pi / 180)
        # self.y_coordinate = self.y_coordinate * 20037508.34789 / 180
        from Data_Preprocessing import Mercator
        self.x_coordinate, self.y_coordinate = Mercator(longitude=longitude, latitude=latitude)

    def record_array(self, array):
        # 把轨迹数据转为归一化后的n纬数据
        # [0TrackId, 1timestamp, 2amplitude, 3x_coordinate, 4y_coordinate, 5speed_average,
        # 6effective_radius, 7velocity, 8angle]
        # [0TrackId, 1timestamp, 2amplitude, 3x_coordinate, 4y_coordinate, 5speed_average,
        # 6effective_radius, 7velocity, 8angle]
        self.array = array
        # if len(self.PCA) == 1:
        #     # 则说明对ASE进行了降维合并：
        #     # [0TrackId, 1timestamp, 2ASE, 3x_coordinate, 4y_coordinate,5velocity, 6angle]
        #     return [self.TrackId, self.timestamp, self.PCA[0], self.x_coordinate, self.y_coordinate,
        #             self.velocity, self.angle]
        # # [0TrackId, 1timestamp, 2amplitude, 3x_coordinate, 4y_coordinate, 5speed_average,
        # # 6effective_radius, 7velocity, 8angle]
        # return [self.TrackId, self.timestamp, self.amplitude, self.x_coordinate, self.y_coordinate,
        # self.speed_average, self.effective_radius, self.velocity, self.angle]

    def origin_array(self):
        # 所有轨迹信息存储方便下次直接读取
        # [0TrackId, 1timestamp, 2amplitude, 3longitude, 4latitude, 5speed_average, 6effective_radius,
        # 7velocity, 8angle]
        return [self.TrackId, self.timestamp, self.amplitude, self.longitude, self.latitude, self.speed_average,
                self.effective_radius, self.velocity, self.angle]

    def add_PCA(self, principal_data):
        self.PCA = principal_data


class Trajectory:
    def __init__(self, id):
        self.TrackId = id  # 轨迹id
        self.StartPoint = None  # 该轨迹开始点的id
        self.EndPoint = None  # 该轨迹结束点的id
        self.NumPoint = 0  # 数据点数量
        self.TimeSpan = [0, 0]  # 时间跨度
        self.ClusterId = -1  # 聚类标签
        self.point_list = []
        self.line_list = []
        self.visited = False
        self.destroy = False  # 当轨迹段缺失值过多则舍弃
        self.noise = False  # 该轨迹是否为噪声点
        self.cluster_repressive = False  # 该轨迹是否为代表轨迹
        self.missing_value = []  # 轨迹缺失段（确实数据）
        self.time_membership = 0  # 密集时间段的隶属度
        self.space_membership = 0  # 密集热点区域的隶属度

    def add_DataPoint(self, point, PointProcessing=True):
        if len(self.point_list) == 0:
            # 这个轨迹聚类添加第一个数据点
            self.StartPoint = point.id
            self.TimeSpan[0] = point.timestamp
            self.NumPoint += 1
        else:
            # 这个轨迹聚类添加最后一个数据点
            self.EndPoint = point.id
            self.TimeSpan[1] = point.timestamp
            self.NumPoint += 1
            if PointProcessing:
                ex_point = self.point_list[-1]
                # 判断时间戳是否连续
                if point.timestamp - ex_point.timestamp != 1:
                    print(self.TrackId, " 轨迹时间戳不连续: ", ex_point.timestamp, "  --->  ", point.timestamp)
                    # if point.timestamp - ex_point.timestamp == 2:
                    #     # 如果时间戳缺少的数量少（2），用线性插值法
                    #     amplitude = (float(ex_point.amplitude) + float(point.amplitude)) / 2
                    #     latitude = (float(ex_point.latitude) + float(point.latitude)) / 2
                    #     longitude = (float(ex_point.longitude) + float(point.longitude)) / 2
                    #     speed_average = (float(ex_point.speed_average) + float(point.speed_average)) / 2
                    #     effective_radius = (float(ex_point.effective_radius) + float(point.effective_radius)) / 2
                    #     temp_point = DataPoint(0, self.TrackId, int(ex_point.timestamp)+1, amplitude, latitude,
                    #                            longitude, speed_average, effective_radius)
                    #     # # 计算上一个数据点速度与转角的属性
                    #     # x1 = ex_point.x_coordinate
                    #     # y1 = ex_point.y_coordinate
                    #     # x2 = temp_point.x_coordinate
                    #     # y2 = temp_point.y_coordinate
                    #     # ex_point.velocity = math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2)) / (
                    #     #             temp_point.timestamp - ex_point.timestamp)
                    #     # if x1 == x2:
                    #     #     ex_point.angle = 0 if x1 > 0 else math.pi
                    #     # else:
                    #     #     ex_point.angle = np.arctan((y2 - y1) / (x2 - x1)) if (y2 - y1) > 0 else np.arctan(
                    #     #         (y2 - y1) / (x2 - x1)) + math.pi
                    #     self.point_list.append(temp_point)
                    #     self.NumPoint += 1
                        # ex_point = self.point_list[-1]
                    # else:
                        # 缺失值过多该轨迹数据舍弃
                        # print("缺失值过多%d轨迹数据舍弃" % self.TrackId)
                    self.destroy = True
                    self.missing_value.append([ex_point.timestamp, point.timestamp])
        self.point_list.append(point)

    def add_Line(self, line):
        self.line_list.append(line)

    def clear_Line(self):
        # 清空line列表
        self.line_list = []

    def cal_va(self):
        # 遍历所有轨迹点，计算v和a
        for i, point in enumerate(self.point_list):
            if i == self.NumPoint-1:
                # 最后一个点不需要计算
                continue
            after_point = self.point_list[i+1]
            x1 = point.x_coordinate
            y1 = point.y_coordinate
            x2 = after_point.x_coordinate
            y2 = after_point.y_coordinate
            point.velocity = math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2)) / (
                    after_point.timestamp - point.timestamp)
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
            point.angle = angle * 180 / math.pi


class Line:
    def __init__(self, id, TrackId, StartPoint, EndPoint):
        self.LineId = id  # 线段id
        self.TrackId = TrackId  # 所属轨迹id
        self.StartPoint = StartPoint.id  # 线段出发点id
        self.EndPoint = EndPoint.id  # 线段结束点id
        self.point_list = [StartPoint, EndPoint]  # 包含在该线段中的数据点
        self.time_span = [StartPoint.timestamp, EndPoint.timestamp]  # 线段的时间跨度
        self.Amplitude_Variations = 0
        self.ASE_Variations = 0
        self.X_Variations = 0
        self.Y_Variations = 0
        self.Speed_Variations = 0
        self.Effective_radius_Variations = 0
        self.Velocity_Variations = 0
        self.Angle_Variations = 0
        self.PCA_Variations = []
        self.feature = None
        self.weight = 0
        # self.calculate_variations()

    def calculate_variations(self):
        # 计算变化特征
        self.Amplitude_Variations = self.point_list[0].amplitude - self.point_list[1].amplitude
        self.X_Variations = float(self.point_list[0].x_coordinate) - float(self.point_list[1].x_coordinate)
        self.Y_Variations = float(self.point_list[0].y_coordinate) - float(self.point_list[1].y_coordinate)
        self.Speed_Variations = self.point_list[0].speed_average - self.point_list[1].speed_average
        self.Effective_radius_Variations = self.point_list[0].effective_radius - self.point_list[1].effective_radius
        self.Velocity_Variations = self.point_list[0].velocity - self.point_list[1].velocity
        self.Angle_Variations = self.point_list[0].angle - self.point_list[1].angle
        self.feature = [self.Amplitude_Variations, self.X_Variations, self.Y_Variations,
                        self.Speed_Variations, self.Effective_radius_Variations, self.Velocity_Variations,
                        self.Angle_Variations]

    def add_variations(self,  X_Variations, Y_Variations, Velocity_Variations, Angle_Variations,
                       Amplitude_Variations=0, Speed_Variations=0,
                       Effective_radius_Variations=0, ASE_Variations=0):
        # 添加变化特征
        self.Amplitude_Variations = float(Amplitude_Variations)
        self.X_Variations = float(X_Variations)
        self.Y_Variations = float(Y_Variations)
        self.Speed_Variations = float(Speed_Variations)
        self.Effective_radius_Variations = float(Effective_radius_Variations)
        self.Velocity_Variations = float(Velocity_Variations)
        self.Angle_Variations = float(Angle_Variations)
        if ASE_Variations != 0:
            self.ASE_Variations = float(ASE_Variations)
            self.feature = [self.ASE_Variations, self.X_Variations, self.Y_Variations,
                            self.Velocity_Variations, self.Angle_Variations]
        else:
            self.feature = [self.Amplitude_Variations, self.X_Variations, self.Y_Variations, self.Speed_Variations,
                        self.Effective_radius_Variations, self.Velocity_Variations, self.Angle_Variations]

    # def add_variations(self, Amplitude_Variations, X_Variations, Y_Variations, Speed_Variations,
    #                    Effective_radius_Variations, Velocity_Variations):
    #     # 添加变化特征
    #     self.Amplitude_Variations = float(Amplitude_Variations)
    #     self.X_Variations = float(X_Variations)
    #     self.Y_Variations = float(Y_Variations)
    #     self.Speed_Variations = float(Speed_Variations)
    #     self.Effective_radius_Variations = float(Effective_radius_Variations)
    #     self.Velocity_Variations = float(Velocity_Variations)
    #     self.feature = [self.Amplitude_Variations, self.X_Variations, self.Y_Variations, self.Speed_Variations,
    #                     self.Effective_radius_Variations, self.Velocity_Variations]

    def add_PCA_Variations(self, PCA_Variations_list):
        # 添加PCA的变化值
        self.PCA_Variations = PCA_Variations_list

    def add_weight(self, weight):
        # 添加权重=linespan/traspan
        self.weight = weight

