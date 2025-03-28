import re

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Line, Line3D, Bar


def model0(df):  # 画model0
    line3d = Line3D(opts.InitOpts(width="1000px", height="1000px", bg_color="white"))
    for track, track_df in df.groupby("track_id"):
        line3d.add(series_name="Track {}".format(int(track)),
                   data=np.array(track_df[["x", "y", "time"]], dtype=np.float64).tolist(),
                   xaxis3d_opts=opts.Axis3DOpts(type_="value", name="longitude", max_=1),
                   yaxis3d_opts=opts.Axis3DOpts(type_="value", name="latitude", min_=-1.25),
                   zaxis3d_opts=opts.Axis3DOpts(type_="value", name="time", min_=0, max_=100),
                   grid3d_opts=opts.Grid3DOpts(width=80, height=90, depth=95,
                                               splitline_opts=opts.SplitLineOpts(is_show=True,
                                                                                 linestyle_opts=opts.LineStyleOpts(
                                                                                     color="#000000"))))
    line3d.set_global_opts(legend_opts=opts.LegendOpts(is_show=False),
                           toolbox_opts=opts.ToolboxOpts(feature=opts.ToolBoxFeatureOpts(
                               save_as_image=opts.ToolBoxFeatureSaveAsImageOpts())))
    line3d.render(os.path.join(file_path, "model.html"))


def model1(df, labels):  # model1
    line3d = Line3D(opts.InitOpts(width="1000px", height="1000px", bg_color="white"))
    for track, track_df in df.groupby("track_id"):
        if int(track_df["label"].unique()) in labels:
            line3d.add(series_name="Track {}".format(int(track)),
                       data=np.array(track_df[["x", "y", "time"]], dtype=np.float64).tolist(),
                       xaxis3d_opts=opts.Axis3DOpts(type_="value", name="longitude", max_=1),
                       yaxis3d_opts=opts.Axis3DOpts(type_="value", name="latitude", min_=-1.25),
                       zaxis3d_opts=opts.Axis3DOpts(type_="value", name="time", min_=0, max_=100),
                       grid3d_opts=opts.Grid3DOpts(width=80, height=90, depth=95,
                                                   splitline_opts=opts.SplitLineOpts(is_show=True,
                                                                                 linestyle_opts=opts.LineStyleOpts(
                                                                                     color="#000000"))))
    line3d.set_global_opts(legend_opts=opts.LegendOpts(is_show=False),
                           toolbox_opts=opts.ToolboxOpts(feature=opts.ToolBoxFeatureOpts(
                               save_as_image=opts.ToolBoxFeatureSaveAsImageOpts())))
    line3d.render(os.path.join(file_path, "model1.html"))


def model2(df, labels):  # after cluster
    line3d = Line3D(opts.InitOpts(width="1000px", height="1000px", bg_color="white"))
    for track, track_df in df.groupby("track_id"):
        if int(track_df["label"].unique()) in labels:
            line3d.add(series_name="Track {}".format(int(track)),
                       data=np.array(track_df[["x", "y", "time"]], dtype=np.float64).tolist(),
                       xaxis3d_opts=opts.Axis3DOpts(type_="value", name="longitude", max_=1),
                       yaxis3d_opts=opts.Axis3DOpts(type_="value", name="latitude", min_=-1.25),
                       zaxis3d_opts=opts.Axis3DOpts(type_="value", name="time", min_=0, max_=100),
                       grid3d_opts=opts.Grid3DOpts(width=80, height=90, depth=95,
                                                   splitline_opts=opts.SplitLineOpts(is_show=True,
                                                                                     linestyle_opts=opts.LineStyleOpts(
                                                                                         color="#000000"))
                                                   ))
    line3d.set_global_opts(legend_opts=opts.LegendOpts(is_show=False),
                           toolbox_opts=opts.ToolboxOpts(feature=opts.ToolBoxFeatureOpts(
                               save_as_image=opts.ToolBoxFeatureSaveAsImageOpts())))
    line3d.render(os.path.join(file_path, "model.html"))
    # colors = ["#5470c6", "#91cc75", "#fac858", "#ea7ccc", "#fc8452"]
    file_data = ""
    with open(os.path.join(file_path, "model.html"), "r") as f:
        for line in f:  # 逐行读  "name": "Track 0",
            if re.compile(r'name\": \"Track \d+').search(line) is not None:
                track_id = int(re.compile(r'\d+').search(line).group())
                if track_id < 50:
                    line = line + "\n            \"color\": \"#5470c6\",\n"
                elif track_id < 100:
                    line = line + "\n            \"color\": \"#91cc75\",\n"
                elif track_id < 150:
                    line = line + "\n            \"color\": \"#fac858\",\n"
                elif track_id < 200:
                    line = line + "\n            \"color\": \"#ea7ccc\",\n"
                elif track_id < 250:
                    line = line + "\n            \"color\": \"#fc8452\",\n"
            file_data += line
    with open(os.path.join(file_path, "model.html"), "w") as f:
        f.write(file_data)


def model3(df, labels):  # result
    colors = ["#5470c6", "#91cc75", "#fac858", "#ea7ccc", "#fc8452"]
    line = Line(init_opts=opts.InitOpts(width="1000px", height="1000px", bg_color="white"))
    for track, track_df in df.groupby("track_id"):
        if int(track_df["label"].unique()) in labels:
            x_data = np.array(track_df["x"], dtype=np.float64).reshape(-1)
            y_data = np.array(track_df["y"], dtype=np.float64).reshape(-1)
            line.add_xaxis(x_data.tolist())
            line.add_yaxis(series_name="Track {}".format(int(track)),
                           y_axis=y_data.tolist(),
                           label_opts=opts.LabelOpts(is_show=False,),
                           color=colors[int(track_df["label"].unique())])
    line.set_global_opts(legend_opts=opts.LegendOpts(is_show=False),
                         xaxis_opts=opts.AxisOpts(type_="value", name="longitude", min_=-1.3, max_=1.1,
                                                  axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(
                                                                                        width=1, color="#000000")),
                                                  splitline_opts=opts.SplitLineOpts(is_show=True,
                                                                                    linestyle_opts=opts.LineStyleOpts(
                                                                                        width=1, color="#000000"))),
                         yaxis_opts=opts.AxisOpts(type_="value", name="latitude", min_=-1.15, max_=1.25,
                                                  axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(
                                                                                        width=1, color="#000000")),
                                                  splitline_opts=opts.SplitLineOpts(is_show=True,
                                                                                    linestyle_opts=opts.LineStyleOpts(
                                                                                        width=1, color="#000000"))),
                         toolbox_opts=opts.ToolboxOpts(
                             feature=opts.ToolBoxFeatureOpts(save_as_image=opts.ToolBoxFeatureSaveAsImageOpts())))
    line.render(os.path.join(file_path, "result.html"))


def density_time(df):  # 时间密度图
    bar = Bar(init_opts=opts.InitOpts(width="1000px", height="600px", bg_color="white"))
    time_array = np.array(df["time"])
    time_count = []
    for i in range(100):
        time_count.append([i, len(np.where(time_array == i)[0])])
    time_count = np.array(time_count)
    x_data = time_count[:, 0].reshape(-1)
    bar.add_xaxis(x_data.tolist())
    y_data = time_count[:, 1].copy()
    y_data = y_data.reshape(-1)
    y_data[50:] = 0
    bar.add_yaxis("", y_data.tolist(), color="#b33939", label_opts=opts.LabelOpts(is_show=False))
    y_data = time_count[:, 1].copy()
    y_data = y_data.reshape(-1)
    y_data[:50] = 0
    bar.add_yaxis("", y_data.tolist(), color="#34ace0", label_opts=opts.LabelOpts(is_show=False))
    bar.set_global_opts(xaxis_opts=opts.AxisOpts(axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(
                                                                                        width=1, color="#d1ccc0")),
        interval=20, splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(
                                                                                        width=1, color="#d1ccc0"))),
        yaxis_opts=opts.AxisOpts(type_="value", axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(
                                                                                        width=1, color="#aaa69d")),
        splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(
                                                                                        width=1, color="#aaa69d"))),
        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(save_as_image=opts.ToolBoxFeatureSaveAsImageOpts()))
    )
    bar.render(os.path.join(file_path, "density_time.html"))


def figure1():  # DTIS模块中的时间密度图
    # path = r"F:\数据集\Chicago_Traffic_Tracker_-_Historical_Congestion_Estimates_by_Segment_-_2018-Current.csv"
    # hour = np.zeros(24)
    # df = pd.read_csv(path)
    # df = df[df["SPEED"] != -1]
    # for h, h_df in df.groupby("HOUR"):
    #     hour[int(h)] = h_df.shape[0]
    # np.save(r"F:\数据集\Chicago_Traffic_count.npy", hour)
    hour = np.load("F:\数据集\Chicago_Traffic_count.npy")
    # bar = Bar(init_opts=opts.InitOpts(width="1000px", height="600px", bg_color="white"))
    # bar.add_xaxis(list(range(24)))
    # print(hour.tolist())
    # bar.add_yaxis("", hour.tolist(), label_opts=opts.LabelOpts(is_show=False))
    # bar.set_global_opts(xaxis_opts=opts.AxisOpts(name="Hour",
    #                                              axisline_opts=opts.AxisLineOpts(linestyle_opts=
    #                                                                              opts.LineStyleOpts(width=1,
    #                                                                                                 color="#000000")),
    #                                              interval=20,
    #                                              ),
    #                     yaxis_opts=opts.AxisOpts(name="Density", type_="value",
    #                                              axisline_opts=opts.AxisLineOpts(linestyle_opts=
    #                                                                              opts.LineStyleOpts(width=1,
    #                                                                                                 color="#000000")),
    #                                              splitline_opts=opts.SplitLineOpts(is_show=True,
    #                                                                                linestyle_opts=
    #                                                                                opts.LineStyleOpts(width=1,
    #                                                                                                 color="#000000"))),
    #                     toolbox_opts=opts.ToolboxOpts(feature=
    #                                                   opts.ToolBoxFeatureOpts(
    #                                                       save_as_image=opts.ToolBoxFeatureSaveAsImageOpts()))
    #                     )
    # bar.render(r"F:\DTIS\figure3.html")
    plt.figure(figsize=(4.6, 3))
    plt.bar(np.arange(24), hour, color="#40739e")
    plt.xticks(np.arange(0,24,2))
    plt.xlabel('Hour', fontname='Arial', x=1.065, labelpad=-20)
    plt.ylabel('ρ', fontname='Arial', loc="top", labelpad=-45, rotation=0)
    plt.gcf().subplots_adjust(wspace=0.27, bottom=0.15, top=0.9, left=0.15, right=0.90)
    plt.savefig(os.path.join("D:\Figure\Final", "Fig_3.jpeg"), format='jpeg', dpi=500)
    plt.show()


def figure2():  # DTIS模块中的决策图
    hour = np.load("F:\数据集\AIS\hour.npy")
    rhos = np.zeros(24)
    T = 24
    TW = 24*0.2 # 0.15
    for i, _ in enumerate(hour):
        for j, p in enumerate(hour):
            if min(abs(i-j), abs(i+T-j), abs(i-T-j)) < TW/2:
                rhos[i] += p
    deltas = np.zeros(24)
    rhos_index = np.argsort(rhos)
    for i, index in enumerate(rhos_index):
        if i == 0:  # 最大的密度
            continue
        higher_rhos_index = rhos_index[:i]
        delta = []
        for higher_index in higher_rhos_index:
            delta.append(min(abs(higher_index - index), abs(higher_index + T - index), abs(higher_index - T - index)))
        deltas[index] = np.min(delta)
    deltas[rhos_index[0]] = np.max(deltas)
    plt.figure(figsize=(8.5, 2.8))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(24), hour, "o-", color="#ae4132", markersize=3.0)
    # plt.xticks(['2','4','6','8','10','12','14','16','18','20','22','0'])
    plt.xlabel('Hour', fontname='Arial', x=1.065, labelpad=-20)
    plt.ylabel('ρ', fontname='Arial', loc="top", labelpad=-45, rotation=0)
    plt.title("(a)", fontname='Arial', loc="center", y=-0.2)
    plt.subplot(1, 2, 2)
    n = rhos.shape[0]
    for i in range(n):
        plt.scatter(rhos[i], deltas[i], s=10, color=(0, 0, 0))
        if i == 10:
            plt.scatter(rhos[i], deltas[i], s=10, color=(0, 0, 0))
            # plt.annotate(str(i), xy=(rhos[i], deltas[i]), xytext=(rhos[i] + 0.5, deltas[i] - 0.35))
        if i == 22:
            plt.scatter(rhos[i], deltas[i], s=10)
            # plt.annotate(str(i), xy=(rhos[i], deltas[i]), xytext=(rhos[i] + 0.5, deltas[i] - 0.35))
        plt.xlabel('LD', fontname='Arial', x=1.05, labelpad=-20)
        plt.ylabel('RD', fontname='Arial', loc="top", labelpad=-20, rotation=0)
        plt.title("(b)", fontname='Arial', loc="center", y=-0.2)
    plt.gcf().subplots_adjust(wspace=0.27, bottom=0.15, top=0.9)
    # plt.savefig(os.path.join("D:\Figure\Final", "Fig_4.jpeg"), format='jpeg', dpi=1000)
    plt.show()


def figure3():  # DTIS模块中的结果图  #b33939 #34ace0  alt+delete
    hour = np.load(r"F:\数据集\Chicago_Traffic_count.npy")  # 3,15
    bar = Bar(init_opts=opts.InitOpts(width="1000px", height="600px", bg_color="white"))
    bar.add_xaxis([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1])
    hour = hour[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1]]
    bar.add_yaxis("", hour.tolist(), label_opts=opts.LabelOpts(is_show=False))
    bar.set_global_opts(xaxis_opts=opts.AxisOpts(name="Hour",
                                                 axisline_opts=opts.AxisLineOpts(linestyle_opts=
                                                                                 opts.LineStyleOpts(width=1,
                                                                                                    color="#000000")),
                                                 interval=20,
                                                 ),
                        yaxis_opts=opts.AxisOpts(name="Density", type_="value",
                                                 axisline_opts=opts.AxisLineOpts(linestyle_opts=
                                                                                 opts.LineStyleOpts(width=1,
                                                                                                    color="#000000")),
                                                 splitline_opts=opts.SplitLineOpts(is_show=True,
                                                                                   linestyle_opts=
                                                                                   opts.LineStyleOpts(width=1,
                                                                                                      color="#000000"))),
                        toolbox_opts=opts.ToolboxOpts(feature=
                        opts.ToolBoxFeatureOpts(
                            save_as_image=opts.ToolBoxFeatureSaveAsImageOpts()))
                        )
    bar.render(r"F:\DTIS\figure4.html")


def figure4():  # 多维轨迹分割
    colors = ["#B33939", "#39B3B3", "#39B339", "#3939B3", "#B37639", "#B339B3"]
    FP = np.array([0, -1, -1.5, -0.75,  1, 2.9, 4, 3.75, 2.5, 1, 0])  # feature profiles
    plt.figure()
    plt.plot(np.arange(len(FP)), FP, "o-", color="#5470c6")
    plt.yticks([])
    plt.xticks(np.arange(len(FP)))
    plt.xlabel("Time")
    plt.ylabel("FP", rotation=0, labelpad=-5, loc="top")
    plt.savefig(r"F:\SW_DTW\track_segment_a.png", dpi=800)
    plt.show()
    v = np.zeros(FP.shape[0])
    for i in range(v.shape[0]):
        if i == (v.shape[0] - 1):
            v[i] = 0
            continue
        v[i] = FP[i+1] - FP[i]
    plt.figure()
    plt.plot(np.arange(len(v)), v, "-", color="#5470c6")
    key = {5: [0, 1], 1: [2], 2: [3, 4, 5], 3: [6], 4: [7, 8, 9], 0: [10]}
    for i in range(v.shape[0]):
        for k in key.keys():
            if i in key[k]:
                plt.plot([i], [v[i]], "o", color=colors[k])
                break
    plt.xticks(np.arange(len(FP)))
    plt.axhline(y=-1, c="#c6aa54")
    plt.axhline(y=0, c="#c6aa54")
    plt.axhline(y=1, c="#c6aa54")
    plt.xlabel("Time")
    plt.ylabel("Slope", rotation=0, labelpad=-5, loc="top")
    plt.savefig(r"F:\SW_DTW\track_segment_b.png", dpi=800)
    plt.show()
    FP = np.array([0, -1.5, -0.75, 4, 3.75, 0])
    plt.figure()
    plt.plot([0, 2], FP[:2], "o-", color=colors[5])
    plt.plot([2, 3], FP[1:3], "o-", color=colors[1])
    plt.plot([3, 6], FP[2:4], "o-", color=colors[2])
    plt.plot([6, 7], FP[3:5], "o-", color=colors[3])
    plt.plot([7, 10], FP[4:6], "o-", color=colors[4])
    plt.yticks([])
    plt.xticks(np.arange(len(v)))
    plt.xlabel("Time")
    plt.ylabel("FP", rotation=0, labelpad=-5, loc="top")
    plt.savefig(r"F:\SW_DTW\track_segment_c.png", dpi=800)
    plt.show()


def Decision_K():  # Dataset1确定K值
    DTI1 = np.load(r"D:\Trajectory_analysis\Control_Group\DTI1_CHf.npy")  # np.arange(2, 11)
    DTI2 = np.load(r"D:\Trajectory_analysis\Control_Group\DTI2_CHf.npy")
    DTI3 = np.load(r"D:\Trajectory_analysis\Control_Group\DTI3_CHf.npy")
    DTI2[6] = 2000
    DTI2[7] = 1500
    DTI2[8] = 500
    plt.figure(figsize=(10,3))
    plt.rcParams["font.family"] = "Arial"
    plt.subplot(131)
    plt.plot(np.arange(2, 11), DTI1, 'o-', label='DTI1', color="#40739e")
    plt.xticks(np.arange(2, 11))
    plt.xlabel("K", fontname='Arial', x=1.03, labelpad=-20)
    plt.ylabel("CHf", fontname='Arial', loc="top", labelpad=-48, rotation=0)
    plt.title("(a)", fontname='Arial', loc="center", y=-0.2)
    plt.legend()

    plt.subplot(132)
    plt.plot(np.arange(2, 11), DTI2, 'o-', label='DTI2', color="#9e4044")
    plt.xticks(np.arange(2, 11))
    plt.xlabel("K", fontname='Arial', x=1.03, labelpad=-20)
    plt.ylabel("CHf", fontname='Arial', loc="top", labelpad=-36, rotation=0)
    plt.title("(b)", fontname='Arial', loc="center", y=-0.2)
    plt.legend()

    plt.subplot(133)
    plt.plot(np.arange(2, 11), DTI3, 'o-', label='DTI3', color="#4eb77e")
    plt.xticks(np.arange(2, 11))
    plt.xlabel("K", fontname='Arial', x=1.03, labelpad=-20)
    plt.ylabel("CHf", fontname='Arial', loc="top", labelpad=-48, rotation=0)
    plt.title("(c)", fontname='Arial', loc="center", y=-0.2)
    plt.legend()

    plt.gcf().subplots_adjust(wspace=0.3, bottom=0.15, top=0.95, left=0.07, right=0.98)
    # plt.savefig(r"D:\Trajectory_analysis\Control_Group\decisionk.png", dpi=800)
    plt.savefig(r"D:\Figure\Final\Fig_11.jpeg", format='jpeg', dpi=500)
    plt.show()


if __name__ == '__main__':
    # figure1()
    # colors = ["#5470c6", "#91cc75", "#fac858", "#ea7ccc", "#fc8452", "#ee6666"]
    # file_path = r"D:\Figure\模型图"
    # df = pd.read_csv(os.path.join(file_path, "track_db2.csv"))  # "track_id", "time", "x", "y", "label"
    # model3(df, [0,1,2])
    # density_time(df)
    # Decision_K()
    figure2()