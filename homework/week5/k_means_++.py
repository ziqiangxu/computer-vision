import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.centers: np.ndarray = None
        self.k = 0

    @staticmethod
    def get_mock_data(centers: List, radius: List, point_num: int) -> np.ndarray:
        """
        generate mock data
        :param centers:
        :param radius:
        :param point_num:
        :return:
        """
        k = len(centers)
        points = np.zeros((k * point_num, 2))
        global_i = 0
        for i in range(k):
            center = centers[i]
            r = radius[i]
            # points[global_i] = np.array(center)
            # global_i += 1
            for j in range(point_num):
                distance = np.random.randint(-r, r) + np.random.random()
                angle = np.random.random() * 3.14
                offset_x = np.cos(angle) * distance
                offset_y = np.sin(angle) * distance
                random_p = center[0] + offset_x, center[1] + offset_y
                points[global_i] = np.array(random_p)
                global_i += 1
        return np.insert(points, 2, values=0, axis=1)

    @staticmethod
    def compute_distance(a, b):
        """
        compute distance of two points
        :param a:
        :param b:
        :return:
        """
        # delta_x = a[0] - b[0]
        # delta_y = a[1] - b[0]
        # return delta_x**2 + delta_y**2
        delta = a - b
        return (delta ** 2).sum()

    def start(self, k: int, plus_algorithm: bool = False, max_iterator: int = 1000):
        """
        start k means
        :param k:
        :param plus_algorithm:
        :param max_iterator:
        :return:
        """
        data_length = len(self.data)
        self.k = k
        # 普通k-means和k-means++的区别，初始化中心点的方式
        if plus_algorithm:
            self.centers = self.init_plus_centers()
        else:
            ids = np.random.choice(data_length, k, False)
            self.centers = self.data[ids][:]

        num = 0
        for i in range(max_iterator):
            num += 1
            new_centers = self.compute_new_centers()
            if self.is_center_keep(self.centers[:, :-1], new_centers[:, :-1]):
                # 中心点不发生变化，结束计算
                break
            # 更新中心点
            self.centers = new_centers
        print(f'{num} times computed')

    @staticmethod
    def is_center_keep(old_centers: np.ndarray, new_centers: np.ndarray, decimals: int = 6):
        """
        :param old_centers:
        :param new_centers:
        :param decimals:
        :return:
        """
        old = np.round(old_centers, decimals)
        new = np.round(new_centers, decimals)
        return (old == new).all()

    def init_plus_centers(self):
        """
        初始化k-means++的中心点
        :return:
        """
        data = self.data
        index = np.random.randint(0, len(data))
        data_size = len(self.data)
        init_centers = np.zeros((self.k, 3))
        init_centers[0] = data[index]

        # 计算余下的中心点
        for i in range(1, self.k):
            chosen_cluster_center = self.compute_cluster_center(
                np.array(init_centers)
            )
            sample_size = min(1000, data_size)
            ids = np.random.choice(data_size, sample_size)
            sample_points = data[ids]
            possibility_arr = np.zeros((sample_size, 3))
            # 计算其它点到已确定中心点的距离，距离越大，占比越大，被选中的概率越大 这个地方可以并行计算
            # 但是为什么还要这个随机性呢？
            for j in range(sample_size):
                point = sample_points[j]
                distance = self.compute_distance(chosen_cluster_center, point)
                possibility_arr[j][0] = distance
            possibility_arr[:, 1] = possibility_arr[:, 0] / possibility_arr[:, 0].sum()

            # 计算该点所在区间 https://blog.csdn.net/dpengwang/article/details/86574999
            possibility_arr[0, 2] = possibility_arr[0, 1]
            for k in range(1, sample_size):
                possibility_arr[k, 2] = possibility_arr[k - 1, 2] + possibility_arr[k, 1]
            # 对这些点进行挑选
            where = np.random.random()
            lower_points = possibility_arr[possibility_arr[:, 2] < where]
            chosen_point = sample_points[len(lower_points)]
            # 填入已选的中心点
            init_centers[i] = chosen_point
        return init_centers

    @staticmethod
    def compute_cluster_center(points: np.ndarray):
        """
        计算一堆点的质心
        :param points:
        :return:
        """
        return points.mean(axis=0)

    def compute_new_centers(self):
        """
        计算中心点
        :return:
        """
        centers = self.centers
        data = self.data
        k = self.k
        new_centers = np.zeros(centers.shape)
        for point in data:
            distance = np.zeros(k)
            for i in range(k):
                distance[i] = self.compute_distance(point[:-1], centers[i, :-1])
                # 计算出距离最小的中心点，判断所属的类别，并保存到data
                min_index = np.argmin(distance)
                point[-1] = min_index

        # 计算各簇的中心点，更新为新的中心点
        for i in range(k):
            # 选出每个类别的点进行计算
            cluster_points = data[data[:, -1] == i]
            new_centers[i] = self.compute_cluster_center(cluster_points)
        return new_centers

    def save_result(self, show_img: bool, img_name: str):
        plt.clf()
        data = self.data
        for i in range(self.k):
            points = data[data[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1], cmap=i)
        plt.savefig(img_name)
        if show_img:
            plt.show()


if __name__ == '__main__':
    mock_data = KMeans.get_mock_data([[10, 10], [20, 25], [20, 13]],
                                         [5, 8, 5], 300)
    # mock_data = KMeans.get_mock_data([[10, 10], [20, 30]],
    #                                  [5, 8], 100)
    # print(mock_data)
    # plt.scatter(mock_data[:, 0], mock_data[:, 1])

    k_means = KMeans(mock_data)
    k_means.start(3)

    start_time = time.time()
    k_means.start(2)
    end_time = time.time()
    print(f'normal k-means, time cost: {end_time - start_time}')
    k_means.save_result(True, 'log/normal.png')

    start_time = time.time()
    k_means.start(2, True)
    end_time = time.time()
    print(f'plus k-means, time cost: {end_time - start_time}')
    k_means.save_result(True, 'log/plus.png')

    """
    通过实验发现，k-means和k-means++都会出现分类效果不理想的情况
    """
