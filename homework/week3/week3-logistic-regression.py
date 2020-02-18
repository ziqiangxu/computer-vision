#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import random


def gen_sample_data():
    center_a = np.random.randint(1, 10, 2) + np.random.random(2)
    angle = random.random() * np.pi
    # a, b中心点距离为10
    center_b = np.array([center_a[0] + np.cos(angle) * 10,
                         center_a[1] + np.sin(angle) * 10])
    # center_b = np.random.randint(15, 25, 2) + np.random.random(2)
    # center_a = np.array([5, 15])
    # center_b = np.array([15, 3])
    print('center point of a: {}, b:{}'.format(center_a, center_b))
        
    points_a = np.array([np.random.randint(-5, 5, 30) + np.random.random(30), 
                         np.random.randint(-5, 5, 30) + np.random.random(30)])
    points_a += center_a.reshape((2, 1))
    const = np.zeros((1, 30), dtype=np.int8)
    points_a = np.vstack((points_a, const))
    type_a = np.zeros((1, 30), dtype=np.int8)
    points_a = np.vstack((points_a, type_a))
    
    points_b = np.array([np.random.randint(-5, 5, 30) + np.random.random(30), 
                         np.random.randint(-5, 5, 30) + np.random.random(30)])
    points_b += center_b.reshape((2, 1))
    points_b = np.vstack((points_b, const))
    type_b = np.ones((1, 30), dtype=np.int8)
    points_b = np.vstack((points_b, type_b))
    return points_a, points_b


# ## Generate sample data


a, b = gen_sample_data()
points = np.concatenate((a, b), axis=1)
# plt.scatter(points[0], points[1])
# plt.show()


# ## 模型


def inference(x: np.ndarray, w: np.ndarray):
#     print(x.shape, w.shape)
    z = w.T.dot(x)
    return 1 / (1 + np.math.e**(-z))

# inference(np.array([[1,2], [3,4]]), np.array([3,4]))


# ## 损失函数


def loss_fun(index, gt):
    """
    :param index: 指标，值在(0， 1)区间
    :param gt: 真实值0或者1
    :return:
    """
    # # 如果index逼近0，则为0类型，那么其损失为0，---即取1的对数
    min_f = 10E-10
    if index == 0.0:
        index = min_f
    elif index == 1.0:
        index = 1 - min_f
    # if gt == 0:
    #     try:
    #         return -np.math.log(1 - index, np.math.e)
    #     except:
    #         print('error0: ', index, gt)
    # # 如果index逼近1，则为1类型，那么其损失为1，---即取1的对数
    # elif gt == 1:
    #     try:
    #         return -np.math.log(index, np.math.e)
    #     except:
    #         print('error1:', index, gt)
    return (gt - 1) * np.log(1 - index) - gt * np.log(index)


def cal_step_gradient(x: np.ndarray, gt: np.ndarray,
                      w, lr):
    """
    :param x:
    :param gt:
    :param w:
    :param lr:
    :return:
    """
    indexes = inference(x, w)

    # 对损失函数求导的计算方式
    e = np.math.e
    d_index: np.ndarray = -(indexes**2) * (e**(-w.dot(x))) * x
    dw_total: np.ndarray = (gt - 1) * d_index / (1 - indexes) - gt * d_index / indexes
    return w - dw_total.mean(axis=1) * lr

    # 优秀作业的梯度计算方式（有疑问）
    # res = x.dot(indexes - gt) / len(x[0])
    # return w - res * lr


def train(data: np.ndarray, ground_truth:np.ndarray,
          batch_size: int, learning_rate=0.1, max_iterations=30):
    w = np.zeros(len(data))
    # w = np.random.random(len(data))
    for i in range(max_iterations):
        ids = np.random.choice(len(data[0]), batch_size)
        # 这里选出的batch_data不对
        batch_data = data[:, ids]
        batch_truth = ground_truth[ids]
        w = cal_step_gradient(batch_data, batch_truth, w, learning_rate)
        # 用w绘制分割边界，此处为2维，则是一条直线
#         x_points = np.linspace(0, 30)
#         y_points = x_points*w[0] + w[1]
        step = max_iterations // 10
        if (i+1) % step == 0:
            indexes = inference(batch_data, w)
            # print(batch_truth)
            loss = np.vectorize(loss_fun)(indexes, batch_truth).mean()
            print('iterator: {}, weights: {}, loss:{}'.format(i, w, loss))
    return w


weights = train(points[:3], points[3], 10, 0.01, 1000)

# 0类别
p0 = a[:, :10]
res0 = inference(p0[:3], weights)
loss_0 = [round(loss_fun(r, 0), 4) for r in res0]
loss_1 = [round(loss_fun(r, 1), 4) for r in res0]
# print("type 0:", p0[:2])
print("res0:", res0)
print("loss0:", loss_0)
print("loss1:", loss_1)

# 1类别
p1 = b[:, :10]
res1 = inference(p1[:3], weights)
loss_0 = [round(loss_fun(r, 0), 4) for r in res1]
loss_1 = [round(loss_fun(r, 1), 4) for r in res1]
# print("type 1:", p1[:2])
print("res1:", res1)
print("loss0:", loss_0)
print("loss1:", loss_1)

plt.scatter(p0[0], p0[1])
plt.scatter(p1[0], p1[1])
plt.show()



