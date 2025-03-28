import os
import numpy as np
import random


def roulette_wheel_selection(probability_list):
    # 轮盘选择策略，得到固定概率的值
    probability_list = np.array(probability_list, dtype=np.float64)
    if len(np.where(probability_list < 0)[0]) > 0:
        # 出现负数的情况，先进行归一化
        if max(probability_list) - min(probability_list) != 0:  # 不应该出现0
            probability_list = (probability_list - min(probability_list)) / (max(probability_list) -
                                                                             min(probability_list))
    # 计算选择概率（相加等于1）
    probability_list = probability_list / sum(probability_list)
    probability_list = np.cumsum(probability_list)
    r = np.random.rand()  # 随机生成0-1的数字
    return np.argwhere(r <= probability_list)[0][0]


def linear_interpolation(output):
    def f(start, end, x):  # 求线性中间值
        if np.isnan(start) or np.isnan(end):
            print("{} or {} is null, please check!".format(start, end))
        return (end - start) / (x - start) * (output[end] - output[start]) + output[start]
    # 获得关键点的下标
    key_index = np.where(~np.isnan(output))[0]
    for idx in range(len(output)):
        if np.isnan(output[idx]):
            output[idx] = f(key_index[idx > key_index][-1], key_index[idx < key_index][0], idx)


def elastic(time_series, length):  # 时间序列拉伸
    if len(time_series) == length:
        return time_series
    # 计算拉伸率
    elongation = (length-1) / (len(time_series)-1)
    output = np.full(length, np.nan)
    # 识别关键点
    from Trajectory_Segmentation.Trajectory_Segmentation_Methodology import myMethodology
    key = myMethodology(time_series)  # 时间的得到的关键点[[0, x[0]], [index, x[index]], ...]
    if len(key) != length:  # 关键点小于需要需要拉伸或者缩小的长度
        # 确定关键点再新时间序列中的位置
        for k_pos, k in enumerate(key):
            if np.isnan(output[round(elongation * k[0])]):
                output[round(elongation * k[0])] = k[1]
            else:  # 进行比较
                ex_key = key[k_pos-1]  # 上一个关键词
                if k_pos-1 == 0:  # 初始值
                    continue
                elif k_pos == len(key) - 1:  # 最后一个关键词
                    output[round(elongation * k[0])] = k[1]
                else:
                    if (abs(ex_key[1] - key[k_pos-2][1]) + abs(ex_key[1] - k[1])) > (abs(k[1] - ex_key[1]) +
                                                                                     abs(k[1] - key[k_pos+1][1])):
                        continue
                    else:
                        output[round(elongation * k[0])] = k[1]
        # 线性填充
        linear_interpolation(output)

    elif len(key) == length:
        output = key[:, 1]
    if len(np.where(np.isnan(output))[0]) != 0:
        print(output)
    return output


class Individual:  # 每个个体(轨迹)
    def __init__(self, DNA_num, weight):
        self.DNA_num = DNA_num  # 轨迹长度,，每个个体的长度可能长度不一样
        self.weight = weight  # 三向隶属度(于这个集群的隶属度)
        self.feature = np.zeros((DNA_num, 8))
        self.fitness = 1  # 该个体的适应度（越小越好）

    def record_feature(self, track_df):  # 根据轨迹点得到个体特征（归一后）
        self.feature = np.array(track_df[["x", "y", "v", "a"]], np.float64)

    def Crossover(self, mother):  # 与其他个体交配
        child = Individual(round((self.fitness * self.DNA_num + mother.fitness * mother.DNA_num)/(self.fitness +
                                                                                                  mother.fitness)),
                           (self.fitness*self.weight + mother.fitness*mother.weight) / (self.fitness + mother.fitness))
        # 子类的DNA长度
        child_length = round((self.fitness*self.DNA_num + mother.fitness*mother.DNA_num)/(self.fitness +
                                                                                          mother.fitness))
        child.fitness = (self.fitness + mother.fitness) / 2
        # 选择父类和母类的概率（根据适应度（越小概率应该越大）判断）
        probability_list = [(self.fitness + mother.fitness) / self.fitness,
                            (self.fitness + mother.fitness) / mother.fitness]
        # 确认子类的8个特征
        for col in range(8):
            parent = roulette_wheel_selection([probability_list])
            if col == 0:  # 时间的变化方式不同
                if parent == 0:
                    child.feature[0, col] = self.feature[0, col]  # 获取父类的时间
                else:
                    child.feature[0, col] = mother.feature[0, col]  # 获取父类的时间
                child.feature[:, col] = np.arange(child.feature[0, col], child_length + child.feature[0, col])
            else:
                # 确认是从父类继承还是母类继承DNA
                if parent == 0:  # 父类
                    child.feature[:, col] = elastic(self.feature[:, col], child_length)  # 进行时间序列拉伸
                else:  # 母类
                    child.feature[:, col] = elastic(mother.feature[:, col], child_length)
        return [child]

    def Mutation(self, refer_gene):
        # 发生基因突变,向适应度好的个体变异
        mutation_div = self.copy()  # 拷贝
        mutant_feature = random.sample(range(0, mutation_div.feature.shape[-1]),
                                       random.randint(1, mutation_div.feature.shape[-1]-1))  # 变异feature下标
        alpha = float(refer_gene.fitness / (refer_gene.fitness + self.fitness))  # 基因参考率
        for m in mutant_feature:
            if m == 0:  # 时间特征变异处理不同
                mutation_div.feature[0, m] = round(alpha * refer_gene.feature[0, m] + (1-alpha) * self.feature[0, m])
                mutation_div.feature[:, m] = np.arange(mutation_div.feature[0, m], mutation_div.feature[0, m] +
                                                       mutation_div.DNA_num)
            else:
                if refer_gene.DNA_num == self.DNA_num:
                    mutation_div.feature[:, m] = alpha * refer_gene.feature[:, m] + (1-alpha) * self.feature[:, m]
                else:
                    mutation_div.feature[:, m] = alpha * elastic(refer_gene.feature[:, m], self.DNA_num) + \
                                                 (1-alpha) * self.feature[:, m]

        return mutation_div

    def copy(self):  # 拷贝一个个体
        copy_div = Individual(self.DNA_num, self.weight)
        copy_div.fitness = self.fitness
        copy_div.feature = self.feature.copy()
        return copy_div

    def distance(self, other_feature):  # 与别的个体进行比较，主要用再计算适应度计算
        dis = 0
        # 时间相似度
        Change_Trend_Similarity(self.feature, other_feature)
        # 空间相似度
         Position_Similarity(self.feature, other_feature)
        # 活动相似度
        Timestamp_Similarity(self.feature, other_feature)
        return dis/3

    def compare(self, other_individual):  # 与其他个体比较，主要用在最后的计算算法是否稳定
        # if other_individual.feature.shape[0] == self.feature.shape[0]:  # 如果长度相同
        #     return np.sum(abs(other_individual.feature.flatten() - self.feature.faltten()))
        # else:
        #     return self.distance(other_individual.feature)
        return abs(self.fitness - other_individual.fitness)


class Population:  # 利用遗传算法得到每个簇的代表轨迹
    def __init__(self, population_num, track_df, track_dictionary, max_iter=300, Crossover_percentage=0.6,
                 Mutation_percentage=0.4, max_selection=10):
        print("Creative a population, number is {}.".format(population_num))
        self.div_num = population_num  # 种群中个体的个数(可以理解为参与演变的轨迹个数）
        self.Track_database = track_df  # 轨迹数据库，用来计算计算出来的的代表轨迹到簇中各个轨迹之间的距离
        self.track_dictionary = track_dictionary  # 轨迹字典：index-->轨迹id, 0:1; 1:1 ; 2:1....
        self.Track_num = len(set(track_df["track_id"]))  # 轨迹数量就是所有轨迹的数量
        self.max_iter = max_iter  # Maximum iteration of genetic algorithm
        self.max_selection = max_selection  # 当簇内的数量到底一点数量的时候只计算前几个
        self.Crossover_percentage = Crossover_percentage  # 交叉百分比（一般，交叉百分比+变异百分比=1）
        self.Mutation_percentage = Mutation_percentage  # 变异百分比（现有个体乘以现有百分比就是变异的个数）
        self.King = None  # 最后的优胜者

    def fit(self, membership):  # 训练模型，遗传算法
        # 初始化
        population = []  # Existing groups
        print("Initialize Population.")  # 创建个体，初始化适应度，并且适应度由高到低进行排列
        if len(np.where(membership != 0)[0]) > self.max_selection:  # 当轨迹数量超过一点数量的话，需要进行删选
            for track_idx in np.argsort(-membership)[:self.max_selection]:
                track_id = self.track_dictionary[track_idx]
                track_df = self.Track_database[self.Track_database["track_id"] == track_id]
                person = Individual(DNA_num=track_df.shape[0], weight=membership[track_idx])
                person.record_feature(track_df)  # 根据轨迹点记录特征
                population.append(person)
        else:
            for track_idx in np.argsort(-membership):
                # 把轨迹格式转换成个体
                track_id = self.track_dictionary[track_idx]
                track_df = self.Track_database[self.Track_database["track_id"] == track_id]
                person = Individual(DNA_num=track_df.shape[0], weight=membership[track_idx])
                person.record_feature(track_df)  # 根据轨迹点记录特征
                population.append(person)
        print("Initialize Population Over!\nStart population evolution...")
        crossover_num = round(self.div_num * self.Crossover_percentage)  # 交叉生成的个体数量
        mutant_num = self.div_num - crossover_num  # 变异生成的个数
        for t in range(self.max_iter):
            print("This is the {}th iteration.".format(t))
            # 注意：经过：10->交叉(+6)->变异(+4)->选择(20->10)之后的个体个数不变！
            # crossover
            crossover_individual = self.crossover(population, crossover_num)  # 把现有群体进行交配，得到（子类和父类)
            population.extend(crossover_individual)  # 10+6
            # mutation
            mutant_individual = self.mutation(population[:self.div_num], mutant_num)  # 对现有群体随机进行变异
            population.extend(mutant_individual)  # 10+6+4
            # selection
            new_population = self.select(population, self.div_num)  # 选择适应度较高的个体进行保留(与原个数相同) 10
            # 比较两个种群，当新产生的种群和老的种群差异收敛，则为结果
            diff = 0
            for div_index, div in enumerate(new_population):
                diff += div.compare(population[div_index])
            print("{}th loss: {}".format(t, diff))
            if diff < 0.0001:  # 函数收敛, 聚类结果
                break
            population = new_population

        # 得到结果隶属度， 选出fit_div中的最高适应度(就是最小值)的个体
        self.King = population[0]  # 返回目前群体中适应度最高的个体
        return self.King.feature

    def select(self, population, select_num):  # 选择适应度较高的个体，并且更新个体适应度
        result = []
        fit = []
        for div in population:  # 对现有群体进行遍历
            # calculate the fitness
            fit.append(self.fitness(div))  # 适应度函数（越小越好）
        fit = np.array(fit, np.float64)
        print("Initial population fitness:", fit[:select_num])
        print("New population fitness:", fit[select_num:])
        print("The value of population fitness which elected is :", fit[np.argsort(fit)[:select_num]])
        # Select individuals with higher fitness for retention
        for div_index in np.argsort(fit)[:select_num]:
            result.append(population[div_index])
        return result

    def fitness(self, div):  # 适应度函数,越大越好（各个特征的SC，总共的SC）
        distance = 0
        w = 0
        for idx in range(self.Track_num):  # 遍历每个轨迹
            # 计算该个体与其他轨迹个体的距离（相似度）越小越好
            other_track = self.Track_database[self.Track_database[:, 0] == idx, 1:]  # 第一列为轨迹下标
            if other_track.shape[0] == div.DNA_num and sum(abs(other_track.flatten() - div.feature.flatten())) == 0:
                continue
            distance += div.distance(other_track) * div.weight * self.Track_membership[idx]
            w += div.weight * self.Track_membership[idx]
        div.fitness = distance / w
        return distance / w

    def crossover(self, population, crossover_num):  # 交叉（后代）
        new_crossover = []  # 存放交叉后代
        fit = []
        for div in population:
            fit.append(1/div.fitness)
        for t in range(int(crossover_num)):
            father_index = roulette_wheel_selection(fit)  # 确定父
            mother_index = roulette_wheel_selection(fit)  # 确定母
            while father_index == mother_index:  # 不可以存在父==母
                father_index = roulette_wheel_selection(fit)
                mother_index = roulette_wheel_selection(fit)
            father = population[father_index]
            mother = population[mother_index]
            new_crossover.extend(father.Crossover(mother))
        print("There are {} off-spring are successfully produced by crossover in the population.".format(crossover_num))
        return new_crossover

    def mutation(self, population, mutant_num):  # 变异
        new_mutation = []  # 变异个体
        fit_div = []
        for div in population:
            fit_div.append(div.fitness)  # 适应度越小（越小）的被选中的可能越小
        for t in range(mutant_num):
            random_div = roulette_wheel_selection(fit_div)
            # 基因库中随机挑选一个基因作为参考
            refer_gene = random.randint(0, self.Track_num-1)  # 基因库中随机抽取一个轨迹
            while refer_gene == random_div:  # 不可以和需要变异的轨迹相同
                refer_gene = random.randint(0, self.Track_num - 1)
            new_mutation.append(population[random_div].Mutation(population[refer_gene]))
        print("There are {} individuals that have mutated in the population.".format(mutant_num))
        return new_mutation


class GenerateRTR:  # 生成代表轨迹——GA,密度,DPC
    def __init__(self, track_df, membership):
        self.track_df = track_df  # 轨迹数据
        self.membership = membership  # 每个簇的隶属度
        self.representative_trajectories = []  # 代表轨迹
        self.min_val = []
        self.max_val = []
        self.normalization()

    def normalization(self):  # 归一化数据
        for col in ["x", "y", "v", "a"]:
            self.min_val.append(self.track_df[col].min())
            self.max_val.append(self.track_df[col].max())
            # 对数据框的每列应用归一化函数
            self.track_df[col] = (self.track_df[col] - self.track_df[col].min()) / (self.track_df[col].max() -
                                                                               self.track_df[col].min())

    def restore(self):  # 还原数据
        for i, col in enumerate(["x", "y", "v", "a"]):
            self.track_df[col] = self.track_df[col] * (self.max_val[i] - self.min_val[i]) + self.min_val[i]

    def GenerateGA(self):  # 用遗传算法生成代表轨迹
        for cluster_id in range(self.membership.shape[1]):  # 遍历每个簇
            track_idx = np.where(self.membership[:, cluster_id] != 0)[0]  # 该簇的所有轨迹
            p = Population(population_num=len(track_idx), track_df=self.track_df)  # 创建种群
            self.representative_trajectories.extend(p.fit(membership=self.membership[:, cluster_id]))
            print("The representative trajectory of cluster {} was successfully generated".format(cluster_id))

    def Save_results(self, filepath):  # 结果保存
        np.save(os.path.join(filepath, "Representative_track.npy"), self.representative_trajectories)
        print("Representative track saved successfully.")
