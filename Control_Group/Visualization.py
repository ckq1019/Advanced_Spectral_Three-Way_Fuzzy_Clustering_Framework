import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

if __name__ == '__main__':
    colors = ["#C31223", "#D994AD", "#3840CF", "#23AA97", "#22A1F0", "#FFCC33", "#80C018", "#DAD81C", "#808080"]
    # Dataset1数据集的可视化
    file_path = os.path.join(os.getcwd(), "data")
    df = pd.read_csv(os.path.join(file_path, "track_db_notime.csv"))  # 读取数据
    track_ids = list(df["track_id"].unique())  # 轨迹id
    count_cluster = {}  # 统计轨迹簇数量
    plt.figure()
    for track_id in track_ids:
        label = int(df[df["track_id"] == track_id]["label"].unique())
        if label == -1:
            plt.plot(df[df["track_id"] == track_id]["x"], df[df["track_id"] == track_id]["y"], "-", color=colors[label],
                     alpha=0.8, label="Noise")
        else:
            plt.plot(df[df["track_id"] == track_id]["x"], df[df["track_id"] == track_id]["y"], "-", color=colors[label],
                     alpha=0.8, label="Cluster " + str(label))
        if label not in count_cluster.keys():
            count_cluster[label] = 0
        count_cluster[label] = count_cluster[label] + 1
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    print("Count cluster: ", count_cluster)
    # Dataset2数据集的可视化
    df = pd.read_csv(os.path.join(file_path, "track_db.csv"))  # 读取数据
    track_ids = list(df["track_id"].unique())  # 轨迹id
    plt.figure()
    count_cluster = {}  # 统计轨迹簇数量
    for track_id in track_ids:
        label = int(df[df["track_id"] == track_id]["label"].unique())
        if label == -1:
            plt.plot(df[df["track_id"] == track_id]["x"], df[df["track_id"] == track_id]["y"], "-", color=colors[label],
                     alpha=0.8, label="Noise")
        else:
            plt.plot(df[df["track_id"] == track_id]["x"], df[df["track_id"] == track_id]["y"], "-", color=colors[label],
                     alpha=0.8, label="Cluster " + str(label))
        if label not in count_cluster.keys():
            count_cluster[label] = 0
        count_cluster[label] = count_cluster[label] + 1
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    print("Count cluster: ", count_cluster)
