import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import math

if __name__ == '__main__':
    file = r"D:\数据集\Black Vultures and Turkey Vultures Southeastern USA.csv"
    df = pd.read_csv(file)
    df = df[["location-long", "location-lat", "individual-local-identifier"]]
    Line = {}
    for track, track_df in df.groupby("individual-local-identifier"):
        # 分割轨迹
        point = np.array(track_df[["location-long", "location-lat"]], dtype=np.float64)
        point = point[np.where(~np.isnan(point))[0]]
        point = point[np.arange(0, len(point), 30)]
        print("before:", point.shape[0])
        Line[track] = [point[0]]
        for i in range(point.shape[0]):
            if i == 0 or i == point.shape[0]-1:
                continue
            pre_p = point[i-1]
            p = point[i]
            aft_p = point[i+1]
            pre_angle = math.atan2(pre_p[1]-p[1], pre_p[0]-p[0])
            pre_angle = pre_angle * 180 / math.pi
            aft_angle = math.atan2(aft_p[1]-p[1], aft_p[0]-p[0])
            aft_angle = aft_angle * 180 / math.pi
            if min(abs(aft_angle-pre_angle), abs(aft_angle-360-pre_angle), abs(aft_angle+360-pre_angle)) < 90:
                Line[track].append(p)
        Line[track].append(point[-1])
        print("after:", len(Line[track]))
    line = []
    # temp_line = []
    for track in Line.keys():
        for i, p in enumerate(Line[track]):
            if i == len(Line[track])-1:
                continue
            line.append([p, Line[track][i+1]])
            # .append([p[0], p[-1], Line[track][i+1][0], Line[track][i+1][-1]])
    print(len(line))
    from sklearn.cluster import OPTICS
    line = np.array(line, np.float16)
    temp_line = (line[:, 0]+line[:, 1])/2
    print(temp_line.shape)
    from sklearn.manifold import TSNE
    temp_line = np.array(temp_line, dtype=np.float16)
    temp_line = TSNE().fit_transform(temp_line)
    result = OPTICS(min_samples=80).fit_predict(temp_line)
    print("不含噪声轨迹段数量", len(result[result != -1]))
    print("聚类数量(含有噪声)：", len(set(result)))
    plt.figure()
    # colors = ["#FFFF88", "#FFFF88", "#FF6E6E", "#FF6E6E", "#FF6E6E", "#FF6E6E",
    #           "#FF6E6E", "#FF6E6E", "#FF6E6E", "#FF6E6E", "#FF6E6E", "#FF6E6E",
    #           "#FF6E6E", "#FF6E6E", "#FF6E6E", "#59F4FF", "#A6EAFF", "#A6EAFF"]  # #FFFF88,#FF6E6E,#59F4FF
    m = Basemap(resolution='i', llcrnrlon=-86.5, llcrnrlat=24.5, urcrnrlon=-75.5, urcrnrlat=35.5)
    for i, l in enumerate(line):
        l = np.array(l)
        if result[i] == -1:
            m.plot(l[:, 0], l[:, 1], "-", markersize=0.9, color="#d1ccc0", linewidth=0.4)
        else:
            print(result[i])
            m.plot(l[:, 0], l[:, 1], "-", markersize=0.9, color="#FFFF88", linewidth=0.4)
    parallels = np.arange(20., 40, 5)
    m.drawparallels(parallels, labels=[False, True, True, False], color="#d1ccc0")
    meridians = np.arange(-90., -70., 5)
    m.drawmeridians(meridians, labels=[True, False, False, True], color="#d1ccc0")
    m.arcgisimage(service='World_Imagery', xpixels=1500, verbose=True)
    plt.savefig(r"F:\Review\segment.png", dpi=800)
    plt.show()




    # 段聚类
    # point = np.array(df[["location-long", "location-lat"]], dtype=np.float64)
    # point = point[np.where(~np.isnan(point))[0]]
    # point = point[np.arange(0, len(point), 900)]
    # from sklearn.cluster import DBSCAN
    # result = DBSCAN(eps=0.2, min_samples=30).fit_predict(point)
    # # for track, track_df in df.groupby("individual-local-identifier"):
    # #     m.plot(track_df["location-long"], track_df["location-lat"], ".", markersize=0.05, linewidth=0.4)
    # #     x += 1
    # colors = ["#FFFF88", "#FF6E6E", "#59F4FF", "#6e6eff", "#ffb76e", "#ffff6e", "#d1ccc0"]
    # labels = list(set(result))
    # print(len(labels))
    # for label in labels:
    #     print(len(point[result == label]))
    #     m.plot(point[result == label][:, 0], point[result == label][:, 1], ".", markersize=0.9, color=colors[int(label)])
    # parallels = np.arange(20., 40, 5)
    # m.drawparallels(parallels, labels=[False, True, True, False], color="#d1ccc0")
    # meridians = np.arange(-90., -70., 5)
    # m.drawmeridians(meridians, labels=[True, False, False, True], color="#d1ccc0")
    # m.arcgisimage(service='World_Imagery', xpixels=1500, verbose=True)
    # # plt.savefig(r"F:\Review\point.png", dpi=800)
    # plt.show()
