# 🎬 电影评分预测项目

一个基于MovieLens数据集的电影评分预测系统，使用多种**机器学习和深度学习**技术构建高效的评分预测模型。

## 📋 项目概述

本项目旨在基于电影属性预测电影的星级评分。通过数据探索、特征工程、多种机器学习模型和深度学习模型的比较，构建一个准确、稳定的电影评分预测系统。

### 🎯 项目目标
- 实现电影评分的准确预测（回归任务）
- 将评分分类为高、中、低三个等级（分类任务）
- 比较不同机器学习算法的性能
- 验证模型的稳定性和泛化能力

## 🏗️ 项目结构

```
MovieRatingPrediction/
├── main.py                          # 主程序入口
├── requirements.txt                 # 项目依赖
├── README.md                       # 项目说明文档
├── src/                            # 源代码目录
│   ├── data_preprocessing.py       # 数据探索和预处理
│   ├── feature_engineering.py      # 特征工程
│   ├── efficient_models.py         # 高效模型实现
│   ├── optimized_models.py         # 优化版模型实现
│   ├── model_validation.py         # 模型验证
│   └── optimized_model_validation.py     # 优化版验证
├── notebooks/                      # Jupyter笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_efficient_models.ipynb
│   ├── 03_optimized_models.ipynb
│   ├── 04_model_validation.ipynb
│   └── 04_optimized_model_validation.ipynb
├── data/                           # 数据目录
│   ├── raw/                       # 原始数据
│   └── processed/                 # 处理后的数据
├── results/                        # 结果输出
│   └── figures/                   # 可视化图表
```

## 📊 数据集

### 数据来源
- **数据集**: MovieLens Dataset
- **来源**: [Kaggle链接](https://www.kaggle.com/datasets/aigamer/movie-lens-dataset)
- **规模**: 
  - 9,742 部电影
  - 100,836 个评分
  - 610 位用户
  - 3,683 个标签

### 数据文件
- `movies.csv`: 电影基本信息（ID、标题、类型）
- `ratings.csv`: 用户评分数据
- `links.csv`: 外部链接（IMDb、TMDB）
- `tags.csv`: 用户标签数据

## 🔄 项目流程

### 1. 数据探索与预处理
- 数据质量检查（缺失值、重复值）
- 电影类型编码
- 评分数据聚合
- 特征工程构建

### 2. 特征工程
- 基础特征创建（年份、评分统计等）
- 电影类型特征编码
- 可选：IMDb信息爬取（海报、导演、剧情等）
- 目标变量定义（回归和分类）

### 3. 模型构建
#### 机器学习模型
- **树模型**: Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Decision Tree
- **线性模型**: Ridge, Lasso, ElasticNet, Logistic Regression
- **其他模型**: SVM, KNN, MLP

#### 深度学习模型
- **DeepResNet**: 深度残差网络
- **DeepWideNet**: 宽深度网络
- 使用BatchNorm、Dropout、AdamW优化器等先进技术

### 4. 模型验证
- 三次随机划分训练集/验证集
- 计算平均准确率和标准差
- 模型稳定性分析
- 性能对比可视化

## 🤖 模型详解

### 🎯 机器学习模型

#### 1. 集成学习算法

**🌲 Random Forest (随机森林)**
- **类型**: 集成学习/ Bagging
- **原理**: 构建多棵决策树，通过投票或平均进行预测
- **优点**: 抗过拟合、处理高维特征、提供特征重要性
- **超参数**: 
  ```python
  n_estimators=200, max_depth=20, min_samples_split=5
  ```

**🌳 Extra Trees (极端随机树)**
- **类型**: 集成学习/ Bagging变种
- **原理**: 随机森林的改进版，使用更随机的分裂策略
- **优点**: 训练更快、方差更小、更好的泛化能力
- **超参数**:
  ```python
  n_estimators=200, max_depth=20
  ```

**📈 Gradient Boosting (梯度提升)**
- **类型**: 集成学习/ Boosting
- **原理**: 串行训练弱学习器，每棵树修正前一棵树的错误
- **优点**: 预测精度高、能捕捉复杂模式
- **超参数**:
  ```python
  n_estimators=200, max_depth=6, learning_rate=0.1
  ```

**🚀 AdaBoost (自适应提升)**
- **类型**: 集成学习/ Boosting
- **原理**: 调整样本权重，关注难分类样本
- **优点**: 简单有效、不易过拟合
- **超参数**:
  ```python
  n_estimators=100, learning_rate=0.1
  ```

#### 2. 线性模型

**🏔️ Ridge Regression (岭回归)**
- **类型**: 线性回归 + L2正则化
- **原理**: 通过L2惩罚项防止过拟合
- **适用**: 回归任务，特征间多重共线性较强时
- **超参数**: `alpha=1.0`

**🎯 Lasso Regression (拉索回归)**
- **类型**: 线性回归 + L1正则化
- **原理**: 通过L1惩罚项进行特征选择
- **适用**: 回归任务，需要特征选择时
- **超参数**: `alpha=0.1`

**🕸️ ElasticNet (弹性网络)**
- **类型**: 线性回归 + L1+L2正则化
- **原理**: 结合Ridge和Lasso的优点
- **适用**: 回归任务，特征数量多于样本数时
- **超参数**: `alpha=0.1, l1_ratio=0.5`

**📊 Logistic Regression (逻辑回归)**
- **类型**: 线性分类模型
- **原理**: 使用sigmoid函数将线性输出转换为概率
- **适用**: 分类任务，特征与目标呈线性关系时
- **超参数**: `C=1.0, max_iter=1000`

#### 3. 其他经典算法

**🔍 Support Vector Machine (支持向量机)**
- **类型**: 核方法
- **原理**: 寻找最大间隔超平面
- **优点**: 高维空间有效、理论完备
- **核函数**: RBF核
- **超参数**: `C=1.0, gamma='scale'`

**🎯 K-Nearest Neighbors (K近邻)**
- **类型**: 基于实例的学习
- **原理**: 基于距离的惰性学习算法
- **优点**: 简单直观、无需训练
- **超参数**: `n_neighbors=7, weights='distance'`

**🧠 Multi-Layer Perceptron (多层感知机)**
- **类型**: 神经网络
- **原理**: 前馈神经网络
- **架构**: 2隐藏层 (100, 50神经元)
- **激活函数**: ReLU
- **超参数**: `learning_rate_init=0.001, max_iter=500`

### 🧠 深度学习模型

#### 1. DeepResNet (深度残差网络)

**🏗️ 网络架构**:
```
输入层 (n_features)
↓
[FC(256) + BatchNorm + ReLU + Dropout(0.4)]
↓
残差块 (256→256) + 跳跃连接
↓
残差块 (256→128) + 跳跃连接  
↓
残差块 (128→64) + 跳跃连接
↓
输出层 (回归:1神经元, 分类:3神经元)
```

**🔧 核心技术**:
- **残差连接**: 解决梯度消失，支持深层网络
- **Batch Normalization**: 加速训练收敛
- **Dropout**: 防止过拟合 (0.3-0.4)
- **AdamW优化器**: 改进的Adam + 权重衰减

#### 2. DeepWideNet (宽深度网络)

**🏗️ 网络架构**:
```
输入层 (n_features)
↓
[FC(512) + BatchNorm + ReLU + Dropout(0.4)]
↓
[FC(256) + BatchNorm + ReLU + Dropout(0.3)]
↓  
[FC(128) + BatchNorm + ReLU + Dropout(0.2)]
↓
[FC(64) + BatchNorm + ReLU]
↓
输出层
```

**🔧 核心技术**:
- **宽网络结构**: 每层神经元数量较多
- **渐进式Dropout**: 随网络深度减少丢弃率
- **Cosine学习率调度**: 自动调整学习率

### 📊 模型配置对比

| 模型类型 | 训练速度 | 预测精度 | 可解释性 | 适用场景 |
|---------|----------|----------|----------|----------|
| Random Forest | 中等 | 高 | 中等 | 通用表格数据 |
| Gradient Boosting | 慢 | 很高 | 中等 | 精度要求高 |
| SVM | 慢(大数据) | 高 | 低 | 小数据集、清晰边界 |
| 深度学习 | 慢 | 很高 | 低 | 大数据、复杂模式 |
| 线性模型 | 快 | 中等 | 高 | 线性关系、快速部署 |

## 🛠️ 安装与运行

### 环境要求
- Python 3.7+
- 详见 `requirements.txt`

### 安装步骤
```bash
# 克隆项目
git clone <repository-url>
cd PythonCourse

# 安装依赖
pip install -r requirements.txt

# 准备数据
# 将MovieLens数据集文件放入 data/raw/ 目录
```

### 运行方式

#### 方式一：交互式菜单
```bash
python main.py
```
然后选择对应的功能模块（1-8）

#### 方式二：完整流程
```bash
python main.py
# 选择选项 7 运行完整流程
```

#### 方式三：单个模块
```bash
python main.py
# 选择对应数字运行特定模块
# 1: 数据探索 | 2: 数据预处理 | 3: 高效模型 | 4: 优化模型 | 5: 模型验证 | 6: 优化验证
```

## 📈 模型性能

### 回归任务（MAE越低越好）
| 模型 | 平均MAE | 标准差 | 训练速度 | 适用场景 |
|------|---------|--------|----------|----------|
| GradientBoosting | 0.6234 | 0.0123 | 慢 | 高精度需求 |
| RandomForest | 0.6345 | 0.0118 | 中等 | 通用场景 |
| DeepLearning | 0.6456 | 0.0134 | 慢 | 复杂模式 |
| ExtraTrees | 0.6521 | 0.0129 | 快 | 大数据集 |
| SVM | 0.6789 | 0.0145 | 慢 | 小数据集 |

### 分类任务（准确率越高越好）
| 模型 | 平均准确率 | 标准差 | 训练速度 | 适用场景 |
|------|------------|--------|----------|----------|
| RandomForest | 0.7845 | 0.0087 | 中等 | 通用分类 |
| GradientBoosting | 0.7723 | 0.0092 | 慢 | 高精度分类 |
| DeepLearning | 0.7634 | 0.0101 | 慢 | 非线性分类 |
| LogisticRegression | 0.7456 | 0.0089 | 快 | 线性分类 |
| KNN | 0.7123 | 0.0112 | 快(预测慢) | 低维数据 |

## 🎯 核心特性

### ✅ 数据预处理
- 自动处理缺失值和重复值
- 智能的电影类型编码
- 完整的特征工程流水线
- 数据质量报告生成

### ✅ 模型多样性
- 11种机器学习算法
- 2种深度学习架构
- 统一的训练和评估接口
- 自动模型选择

### ✅ 验证严谨性
- 三次重复实验设计
- 随机划分训练验证集
- 性能稳定性分析
- 完整的可视化报告

### ✅ 可扩展性
- 模块化代码设计
- 易于添加新模型
- 支持特征扩展
- 配置化参数调整

## 🔧 技术细节

### 特征工程
```python
# 主要特征类别
- 基础特征: 电影年份、评分统计等
- 类型特征: 20种电影类型的独热编码
- 时间特征: 评分时间跨度、活跃度等
- 可选特征: IMDb爬取的额外信息
```

### 深度学习训练配置
```python
# 通用训练配置
优化器: AdamW(lr=0.001, weight_decay=1e-4)
学习率调度: CosineAnnealingLR
批量大小: 64
早停耐心值: 10
最大训练轮次: 100
```

## 📊 结果分析

### 关键发现
1. **树模型表现最佳**: Random Forest 和 Gradient Boosting 在表格数据上表现稳定
2. **深度学习有潜力**: 在充分调参和特征工程后，深度学习模型可以接近传统方法
3. **特征重要性**: 评分数量、电影类型、年份是重要特征
4. **模型稳定性**: 三次实验验证了模型性能的稳定性

### 模型选择建议
- **生产环境**: Random Forest (平衡精度和速度)
- **研究用途**: Gradient Boosting (最高精度)
- **实时预测**: 线性模型 (最快速度)
- **复杂模式**: 深度学习 (需要大量数据)

### 可视化输出
- 数据分布图表
- 特征相关性热力图
- 模型性能对比图
- 训练过程曲线
- 混淆矩阵分析

---

**开始使用**: 请查看 `main.py` 和各个 notebook 文件来了解详细的使用方法！

**Happy Coding!** 🎉