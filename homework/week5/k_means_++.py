from typing import List

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.centers = []
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
        if plus_algorithm:
            # todo ++
            pass
        else:
            ids = np.random.choice(data_length, k, False)
            self.centers = self.data[ids][:]

        num = 0
        for i in range(max_iterator):
            num += 1
            new_centers = self.compute_centers()
            if self.is_center_keep(self.centers[:, :-1], new_centers[:, :-1]):
                # 中心点不发生变化，结束计算
                break
            # 更新中心点
            self.centers = new_centers
        print(f'{num} times computed')

    @staticmethod
    def is_center_keep(old_centers: np.ndarray, new_centers: np.ndarray, decimals: int = 3):
        """
        :param old_centers:
        :param new_centers:
        :param decimals:
        :return:
        """
        old = np.round(old_centers, decimals)
        new = np.round(new_centers, decimals)
        return (old == new).all()

    @staticmethod
    def compute_cluster_center(points: np.ndarray):
        """
        计算一堆点的质心
        :param points:
        :return:
        """
        return points.mean(axis=0)

    def compute_centers(self):
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

    def show_2d(self):
        data = self.data
        for i in range(self.k):
            points = data[data[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1], cmap=i)
        plt.show()


if __name__ == '__main__':
    mock_data = KMeans.get_mock_data([[10, 10], [20, 30], [20, 13]],
                                         [5, 8, 5], 100)
    # mock_data = KMeans.get_mock_data([[10, 10], [20, 30]],
    #                                  [5, 8], 100)
    # print(mock_data)
    # plt.scatter(mock_data[:, 0], mock_data[:, 1])

    k_means = KMeans(mock_data)
    k_means.start(3)
    # print(mock_data)
    k_means.show_2d()
    # plt.show()
