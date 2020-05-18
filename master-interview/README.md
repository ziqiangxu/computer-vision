# 肺实质分割和肺结节检测

## 程序执行环境

- `python 3.6.8`
- 软件依赖包列表 `reqiurements.txt`
- 安装进行肺实质分割的包 `pip install git+https://github.com/JoHof/lungmask`

## 项目概览

```
.
├── data # 存放原始对影像数据和中间结果、最终结果
│   ├── archiving
│   ├── log
│   ├── lung.nii
│   ├── nod.mat
│   └── output
├── nodule_detector.ipynb
├── nodule_measure.ipynb
├── README.md
├── requirements.txt # 项目依赖包清单
├── data_convert.py  # 1. 首先对题目对mat数据进行转换，转为nii文件
├── lung_seg.py # 2. 肺实质分割
├── suspect_nodule_detector.py # 3. 肺结节疑似区域检测
├── nodule_mesure.py # 4. 对肺结节进行测量
├── const.py # 项目使用的部分常量
└── tools.py # 提供一些通用的函数
```

其中 `.ipynb` 文件是在探索问题解决方案时的一些试验性代码， `.py` 的代码由此进而来


## 运行本项目

在搭建好运行环境之后，逐步进行下面的步骤

1. 准备数据 `python data_convert.py`
2. 肺实质分割 `python lung_seg.py`
3. 疑似区域检测 `python suspect_nodule_detector.py`
4. 肺结节指标测量 `python nodule_mesure.py`

> 注: 依据肺结节疑似区域分割肺结节暂未完成

## 解题思路和流程

拿到题目之后，有些慌张，因为我没有太多影像知识的。在此之前只做过医疗图像的可视化，并没有做分割的工作。好在学过一些CV知识和简单使用过OpenCV的一些API，近期也学习了少量的医疗图像知识。

首先，对于肺实质、肺结节这些概念，我是不清楚的。所以，我首先自己要学习一下相关的定义，学会基本的读片，才能正确地利用计算机手段来检测他们。

相关图片在 `img` 目录下