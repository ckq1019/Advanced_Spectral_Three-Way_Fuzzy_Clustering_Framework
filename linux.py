import os
import time

from Global_Variable import DATA_PATH
from Similarity.Similarity_Measurement_Methodology import SimilarityMatrix
from Data_Preprocessing import format_conversion
from KNN.R_TREE import RTree
from main import read_data
from KNN.KDtree import KNNTree
from Intensive_Space_Extraction.Sapce_Extraction import Intensive_space_interval
from Global_Variable import *

if __name__ == '__main__':
    # Trajectory preprocessing
    file_path = r"cyclonic_[353  80].csv"
    origin_data, file_format = read_data(os.path.join(DATA_PATH, file_path))
    # 数据预处理
    data = format_conversion(origin_data, file_format=file_format, year_range=[2018, 2022], latitude_range=[0, 25.0],
                             longitude_range=[100.0, 122.0], Data_processing=False)
    track_list = []
    for i in data.keys():
        track_list.append(data[i])

    # membership_mat = Intensive_space_interval(track_list, Show=False, Save=False, reshuffle=False)
    start_time = time.time()
    K = 0
    sm = SimilarityMatrix(isCalculate=True, TR_List=track_list, K=K)
    k = KNNTree(track_list, None)
    # 检索KNN轨迹
    cal_type = 'P'  # 'P' or 'L'
    for track_index in range(len(track_list)):
        sm.calculation(track_index, k.knn(track_index, K), cal_type, 'HD', Compression=None)
    sm.savefile(path=os.path.join(SIMILAR_MATRIX, 'HD_Similarity_matrix.npy'))
    end_time = time.time()
    print("耗时：", end_time - start_time)
