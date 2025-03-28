import netCDF4
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # 预处理
    file = "D:\\Dataset\\META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc"
    content = netCDF4.Dataset(file)
    data = content.variables
    print(data)
    # result = []  # [track,time,amplitude,longitude,latitude,speed_average,effective_radius]
    # del_track = []  # 当存在只有一半轨迹在范围中的时候，把该轨迹删除
    # pre_id = -1
    # scope = [0, 0]
    # for i in range(data['time'].shape[0]):  # 遍历内容
    #     if pre_id != int(data['track'][i]):
    #         if sum(scope) == 2:
    #             del_track.append(pre_id)
    #         pre_id = int(data['track'][i])
    #         scope = [0, 0]  # 超出范围，在范围内
    #     if float(data['longitude'][i].data) < 109 or float(data['longitude'][i].data) > 121:
    #         scope[0] = 1
    #         continue
    #     if float(data['latitude'][i].data) < 3 or float(data['latitude'][i].data) > 23:
    #         scope[0] = 1
    #         continue
    #     scope[1] = 1
    #     result.append([int(data['track'][i]), int(data['time'][i]), float(data['amplitude'][i]),
    #                    float(data['longitude'][i].data), float(data['latitude'][i].data),
    #                    float(data['speed_average'][i]), float(data['effective_radius'][i])])
    # df = pd.DataFrame(data=result, columns=['track', 'time', 'amplitude', 'longitude', 'latitude', 'speed_average',
    #                                         'effective_radius'])
    # for del_t in del_track:
    #     df = df[df["track"] != del_t]
    # print("track num:", len(df["track"].unique()))
    # df.to_csv(r"D:\数据集\Dataset2\Datset2.csv", index=False)
    # 检查缺值轨迹, 检查过短轨迹
    # df1 = pd.read_csv(r"D:\数据集\Dataset2\Datset2_short.csv")
    # df2 = pd.read_csv(r"D:\数据集\Dataset2\Datset2_long.csv")
    # df = []  # 全部
    # del_track = []
    # for track, tr_df in df1.groupby("track"):
    #     if tr_df.shape[0] > 10 and tr_df.shape[0] == int(tr_df["time"].max() - tr_df["time"].min() + 1):
    #         df.extend(list(np.array(tr_df, dtype=np.float64)))
    # for track, tr_df in df2.groupby("track"):
    #     if tr_df.shape[0] > 10 and tr_df.shape[0] == int(tr_df["time"].max() - tr_df["time"].min() + 1):
    #         df.extend(np.array(tr_df, dtype=np.float64).tolist())
    # df = pd.DataFrame(data=df, columns=['track', 'time', 'amplitude', 'longitude', 'latitude', 'speed_average', 'effective_radius'])
    # print("track num:", len(df["track"].unique()))
    # print("track point:", df.shape[0])
    # df.to_csv(r"D:\数据集\Dataset2\Dataset2.csv", index=False)