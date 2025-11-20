# """
# 特征工程模块
# 构建有效的特征并进行建模
# """

# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
# import requests
# from bs4 import BeautifulSoup
# import time
# import os
# from pathlib import Path
# import json
# import warnings
# warnings.filterwarnings('ignore')

# class IMDbCrawler:
#     """IMDb爬虫类，用于获取额外电影信息"""
    
#     def __init__(self, delay=1.0):
#         self.delay = delay  # 请求延迟，避免被封IP
#         self.headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#         }
#         self.cache_file = Path("data/processed/imdb_cache.json")
#         self.cache = self._load_cache()
    
#     def _load_cache(self):
#         """加载缓存数据"""
#         if self.cache_file.exists():
#             with open(self.cache_file, 'r', encoding='utf-8') as f:
#                 return json.load(f)
#         return {}
    
#     def _save_cache(self):
#         """保存缓存数据"""
#         self.cache_file.parent.mkdir(parents=True, exist_ok=True)
#         with open(self.cache_file, 'w', encoding='utf-8') as f:
#             json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
#     def get_imdb_info(self, imdb_id):
#         """获取IMDb电影信息"""
#         if not imdb_id or imdb_id == 0:
#             return {}
        
#         # 检查缓存
#         cache_key = str(imdb_id)
#         if cache_key in self.cache:
#             return self.cache[cache_key]
        
#         try:
#             # 构造URL
#             url = f"https://www.imdb.com/title/tt{imdb_id:07d}/"
#             print(f"正在爬取: {url}")
            
#             # 发送请求
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code != 200:
#                 print(f"请求失败，状态码: {response.status_code}")
#                 return {}
            
#             # 解析HTML
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # 提取电影信息
#             movie_info = {}
            
#             # 提取评分
#             rating_element = soup.find('span', class_='sc-bde20123-1')
#             if rating_element:
#                 try:
#                     movie_info['imdb_rating'] = float(rating_element.text)
#                 except:
#                     movie_info['imdb_rating'] = None
            
#             # 提取投票数
#             votes_element = soup.find('div', class_='sc-bde20123-3')
#             if votes_element:
#                 movie_info['imdb_votes'] = votes_element.text
            
#             # 提取剧情简介
#             summary_element = soup.find('span', class_='sc-466bb6c-0')
#             if summary_element:
#                 movie_info['summary'] = summary_element.text.strip()
            
#             # 提取导演
#             director_elements = soup.find_all('a', class_='ipc-metadata-list-item__list-content-item')
#             directors = []
#             for element in director_elements:
#                 if 'director' in element.text.lower():
#                     directors.append(element.text.strip())
#             if directors:
#                 movie_info['directors'] = directors
            
#             # 缓存结果
#             self.cache[cache_key] = movie_info
#             self._save_cache()
            
#             # 延迟，避免请求过快
#             time.sleep(self.delay)
            
#             return movie_info
            
#         except Exception as e:
#             print(f"爬取IMDb信息失败 (ID: {imdb_id}): {e}")
#             return {}

# class MovieFeatureEngineer:
#     """电影特征工程类"""
    
#     def __init__(self, project_path="D:/VSCodeProjects/PythonCourse"):
#         self.project_path = Path(project_path)
#         self.processed_path = self.project_path / "data" / "processed"
#         self.scaler = StandardScaler()
#         self.imdb_crawler = IMDbCrawler(delay=0.5)  # 降低延迟提高速度
    
#     def load_processed_data(self):
#         """加载处理后的数据"""
#         print("📁 加载处理后的数据...")
        
#         movie_features = pd.read_csv(self.processed_path / "movie_features_clean.csv")
        
#         print(f"✅ 数据加载完成: {movie_features.shape}")
#         return movie_features
    
#     def create_basic_features(self, df):
#         """创建基础特征"""
#         print("🏗️ 创建基础特征...")
        
#         # 复制数据框
#         features_df = df.copy()
        
#         # 1. 时间相关特征
#         current_year = 2024  # 假设当前年份
#         features_df['movie_age'] = current_year - features_df['year']
        
#         # 2. 评分相关特征
#         features_df['rating_count_log'] = np.log1p(features_df['rating_count'])
#         features_df['has_high_rating_count'] = (features_df['rating_count'] > features_df['rating_count'].median()).astype(int)
        
#         # 3. 类型数量特征
#         genre_cols = [col for col in features_df.columns if col.startswith('genre_')]
#         features_df['genre_count'] = features_df[genre_cols].sum(axis=1)
#         features_df['has_multiple_genres'] = (features_df['genre_count'] > 1).astype(int)
        
#         # 4. 评分稳定性特征
#         features_df['rating_stability'] = 1 / (1 + features_df['rating_std'].fillna(0))
        
#         print(f"   创建了 {len(features_df.columns) - len(df.columns)} 个新特征")
#         return features_df
    
#     def add_imdb_features(self, df, sample_size=None):
#         """添加IMDb特征（可选）"""
#         print("🎬 添加IMDb特征...")
        
#         # 为了演示，只处理部分数据
#         if sample_size:
#             df_sample = df.head(sample_size).copy()
#         else:
#             df_sample = df.copy()
        
#         imdb_features = []
        
#         for idx, row in df_sample.iterrows():
#             imdb_id = row['imdbId']
#             features = {}
            
#             if imdb_id and imdb_id != 0:
#                 movie_info = self.imdb_crawler.get_imdb_info(imdb_id)
                
#                 # 提取数值型特征
#                 features['imdb_rating'] = movie_info.get('imdb_rating', 0)
#                 features['has_imdb_rating'] = 1 if movie_info.get('imdb_rating') else 0
#                 features['summary_length'] = len(movie_info.get('summary', ''))
#                 features['director_count'] = len(movie_info.get('directors', []))
#             else:
#                 # 没有IMDb ID的电影
#                 features.update({
#                     'imdb_rating': 0,
#                     'has_imdb_rating': 0,
#                     'summary_length': 0,
#                     'director_count': 0
#                 })
            
#             imdb_features.append(features)
            
#             # 显示进度
#             if (idx + 1) % 10 == 0:
#                 print(f"   已处理 {idx + 1}/{len(df_sample)} 部电影")
        
#         # 合并特征
#         imdb_df = pd.DataFrame(imdb_features)
#         result_df = pd.concat([df_sample.reset_index(drop=True), imdb_df], axis=1)
        
#         print(f"   添加了 {len(imdb_df.columns)} 个IMDb特征")
#         return result_df
    
#     def prepare_modeling_features(self, df, use_imdb=False, sample_size=100):
#         """准备建模特征"""
#         print("🔧 准备建模特征...")
        
#         # 1. 创建基础特征
#         features_df = self.create_basic_features(df)
        
#         # 2. 可选：添加IMDb特征
#         if use_imdb:
#             features_df = self.add_imdb_features(features_df, sample_size)
        
#         # 3. 选择特征列
#         # 排除非特征列
#         exclude_cols = ['movieId', 'title', 'genres', 'genres_list', 'first_rating_date', 'last_rating_date']
        
#         # 选择数值型特征
#         numeric_features = features_df.select_dtypes(include=[np.number]).columns
#         feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
#         # 4. 处理目标变量
#         # 选项1: 回归任务（预测具体评分）
#         # 选项2: 分类任务（将评分转为类别）
#         features_df = self.create_target_variable(features_df)
        
#         print(f"   最终特征数量: {len(feature_cols)}")
#         print(f"   特征列: {feature_cols}")
        
#         return features_df, feature_cols
    
#     def create_target_variable(self, df):
#         """创建目标变量"""
#         print("🎯 创建目标变量...")
        
#         # 方法1: 回归任务 - 直接使用平均评分
#         df['target_regression'] = df['avg_rating']
        
#         # 方法2: 分类任务 - 将评分分为3个类别
#         # 0: 低分(0-2.5), 1: 中分(2.5-4), 2: 高分(4-5)
#         conditions = [
#             df['avg_rating'] <= 2.5,
#             (df['avg_rating'] > 2.5) & (df['avg_rating'] <= 4.0),
#             df['avg_rating'] > 4.0
#         ]
#         choices = [0, 1, 2]  # 低, 中, 高
#         df['target_classification'] = np.select(conditions, choices, default=1)
        
#         # 方法3: 二分类 - 是否高于平均分
#         rating_mean = df['avg_rating'].mean()
#         df['target_binary'] = (df['avg_rating'] > rating_mean).astype(int)
        
#         print(f"   目标变量分布:")
#         print(f"   - 回归目标范围: {df['target_regression'].min():.2f} - {df['target_regression'].max():.2f}")
#         print(f"   - 分类目标分布: {df['target_classification'].value_counts().sort_index().to_dict()}")
#         print(f"   - 二分类分布: {df['target_binary'].value_counts().sort_index().to_dict()}")
        
#         return df

# class MovieDataset(Dataset):
#     """PyTorch数据集类"""
    
#     def __init__(self, features, targets):
#         self.features = torch.FloatTensor(features)
#         self.targets = torch.FloatTensor(targets) if targets.dtype == float else torch.LongTensor(targets)
    
#     def __len__(self):
#         return len(self.features)
    
#     def __getitem__(self, idx):
#         return self.features[idx], self.targets[idx]

# class RatingPredictor(nn.Module):
#     """电影评分预测模型"""
    
#     def __init__(self, input_size, output_type='regression', hidden_sizes=[128, 64, 32]):
#         super().__init__()
#         self.output_type = output_type
        
#         # 构建动态网络层
#         layers = []
#         prev_size = input_size
        
#         for hidden_size in hidden_sizes:
#             layers.extend([
#                 nn.Linear(prev_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#                 nn.BatchNorm1d(hidden_size)
#             ])
#             prev_size = hidden_size
        
#         self.network = nn.Sequential(*layers)
        
#         # 输出层
#         if output_type == 'regression':
#             self.output_layer = nn.Linear(prev_size, 1)
#         elif output_type == 'binary':
#             self.output_layer = nn.Linear(prev_size, 1)
#             self.sigmoid = nn.Sigmoid()
#         else:  # multiclass
#             self.output_layer = nn.Linear(prev_size, 3)  # 3个类别
    
#     def forward(self, x):
#         x = self.network(x)
#         x = self.output_layer(x)
        
#         if self.output_type == 'binary':
#             x = self.sigmoid(x)
        
#         return x.squeeze()

# class ModelTrainer:
#     """模型训练器"""
    
#     def __init__(self, model, device='cpu'):
#         self.model = model.to(device)
#         self.device = device
        
#     def train_model(self, train_loader, val_loader, output_type='regression', 
#                    epochs=100, lr=0.001):
#         """训练模型"""
        
#         # 选择损失函数和优化器
#         if output_type == 'regression':
#             criterion = nn.MSELoss()
#         elif output_type == 'binary':
#             criterion = nn.BCELoss()
#         else:  # multiclass
#             criterion = nn.CrossEntropyLoss()
        
#         optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
#         train_losses = []
#         val_losses = []
#         val_metrics = []
        
#         print("🚀 开始训练模型...")
        
#         for epoch in range(epochs):
#             # 训练阶段
#             self.model.train()
#             train_loss = 0
            
#             for batch_features, batch_targets in train_loader:
#                 batch_features = batch_features.to(self.device)
#                 batch_targets = batch_targets.to(self.device)
                
#                 optimizer.zero_grad()
#                 outputs = self.model(batch_features)
                
#                 if output_type == 'multiclass':
#                     loss = criterion(outputs, batch_targets.long())
#                 else:
#                     loss = criterion(outputs, batch_targets)
                
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()
            
#             # 验证阶段
#             self.model.eval()
#             val_loss = 0
#             all_predictions = []
#             all_targets = []
            
#             with torch.no_grad():
#                 for batch_features, batch_targets in val_loader:
#                     batch_features = batch_features.to(self.device)
#                     batch_targets = batch_targets.to(self.device)
                    
#                     outputs = self.model(batch_features)
                    
#                     if output_type == 'multiclass':
#                         loss = criterion(outputs, batch_targets.long())
#                         predictions = torch.argmax(outputs, dim=1)
#                     else:
#                         loss = criterion(outputs, batch_targets)
#                         predictions = outputs
                    
#                     val_loss += loss.item()
#                     all_predictions.extend(predictions.cpu().numpy())
#                     all_targets.extend(batch_targets.cpu().numpy())
            
#             # 计算指标
#             avg_train_loss = train_loss / len(train_loader)
#             avg_val_loss = val_loss / len(val_loader)
            
#             train_losses.append(avg_train_loss)
#             val_losses.append(avg_val_loss)
            
#             # 计算验证集指标
#             val_metric = self.calculate_metrics(all_predictions, all_targets, output_type)
#             val_metrics.append(val_metric)
            
#             scheduler.step(avg_val_loss)
            
#             if (epoch + 1) % 20 == 0:
#                 print(f'Epoch [{epoch+1}/{epochs}], '
#                       f'Train Loss: {avg_train_loss:.4f}, '
#                       f'Val Loss: {avg_val_loss:.4f}, '
#                       f'Val Metric: {val_metric:.4f}')
        
#         return train_losses, val_losses, val_metrics
    
#     def calculate_metrics(self, predictions, targets, output_type):
#         """计算评估指标"""
#         if output_type == 'regression':
#             return mean_absolute_error(targets, predictions)
#         else:
#             return accuracy_score(targets, predictions)
    
#     def evaluate_model(self, data_loader, output_type='regression'):
#         """评估模型"""
#         self.model.eval()
#         all_predictions = []
#         all_targets = []
        
#         with torch.no_grad():
#             for batch_features, batch_targets in data_loader:
#                 batch_features = batch_features.to(self.device)
#                 batch_targets = batch_targets.to(self.device)
                
#                 outputs = self.model(batch_features)
                
#                 if output_type == 'multiclass':
#                     predictions = torch.argmax(outputs, dim=1)
#                 elif output_type == 'binary':
#                     predictions = (outputs > 0.5).int()
#                 else:  # regression
#                     predictions = outputs
                
#                 all_predictions.extend(predictions.cpu().numpy())
#                 all_targets.extend(batch_targets.cpu().numpy())
        
#         return all_predictions, all_targets

# def run_experiment(features, targets, output_type='regression', experiment_num=1):
#     """运行单次实验"""
#     print(f"\n🔬 实验 {experiment_num} - {output_type.upper()} 任务")
    
#     # 创建数据集
#     dataset = MovieDataset(features, targets)
    
#     # 划分训练集和验证集
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
#     # 创建数据加载器
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
#     # 创建模型
#     input_size = features.shape[1]
#     model = RatingPredictor(input_size, output_type)
    
#     # 训练模型
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     trainer = ModelTrainer(model, device)
    
#     train_losses, val_losses, val_metrics = trainer.train_model(
#         train_loader, val_loader, output_type, epochs=50, lr=0.001
#     )
    
#     # 最终评估
#     predictions, true_targets = trainer.evaluate_model(val_loader, output_type)
    
#     if output_type == 'regression':
#         mae = mean_absolute_error(true_targets, predictions)
#         print(f"✅ 实验 {experiment_num} 完成 - MAE: {mae:.4f}")
#         return mae
#     else:
#         accuracy = accuracy_score(true_targets, predictions)
#         print(f"✅ 实验 {experiment_num} 完成 - 准确率: {accuracy:.4f}")
#         return accuracy

# if __name__ == "__main__":
#     # 运行特征工程和建模
#     engineer = MovieFeatureEngineer()
#     movie_data = engineer.load_processed_data()
    
#     # 准备特征（不使用IMDb数据以加快速度）
#     features_df, feature_cols = engineer.prepare_modeling_features(
#         movie_data, use_imdb=False
#     )
    
#     # 选择特征和目标
#     X = features_df[feature_cols].values
#     y_regression = features_df['target_regression'].values
#     y_classification = features_df['target_classification'].values
    
#     print(f"特征矩阵形状: {X.shape}")
#     print(f"回归目标形状: {y_regression.shape}")
#     print(f"分类目标形状: {y_classification.shape}")

"""
特征工程模块
构建有效的特征并进行建模
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MovieFeatureEngineer:
    """电影特征工程类"""
    
    def __init__(self, project_path="D:/VSCodeProjects/PythonCourse"):
        self.project_path = Path(project_path)
        self.processed_path = self.project_path / "data" / "processed"
        self.scaler = StandardScaler()
    
    def load_processed_data(self):
        """加载处理后的数据"""
        print("📁 加载处理后的数据...")
        
        movie_features = pd.read_csv(self.processed_path / "movie_features_clean.csv")
        
        print(f"✅ 数据加载完成: {movie_features.shape}")
        return movie_features
    
    def create_basic_features(self, df):
        """创建基础特征"""
        print("🏗️ 创建基础特征...")
        
        # 复制数据框
        features_df = df.copy()
        
        # 1. 时间相关特征
        current_year = 2024  # 假设当前年份
        features_df['movie_age'] = current_year - features_df['year']
        
        # 2. 评分相关特征
        features_df['rating_count_log'] = np.log1p(features_df['rating_count'])
        features_df['has_high_rating_count'] = (features_df['rating_count'] > features_df['rating_count'].median()).astype(int)
        
        # 3. 类型数量特征
        genre_cols = [col for col in features_df.columns if col.startswith('genre_')]
        features_df['genre_count'] = features_df[genre_cols].sum(axis=1)
        features_df['has_multiple_genres'] = (features_df['genre_count'] > 1).astype(int)
        
        # 4. 评分稳定性特征
        features_df['rating_stability'] = 1 / (1 + features_df['rating_std'].fillna(0))
        
        print(f"   创建了 {len(features_df.columns) - len(df.columns)} 个新特征")
        return features_df
    
    def add_synthetic_imdb_features(self, df):
        """添加合成的IMDb特征（基于现有数据生成）"""
        print("🎬 添加合成IMDb特征...")
        
        features_df = df.copy()
        
        # 基于现有评分生成模拟的IMDb评分
        np.random.seed(42)  # 保证可重复性
        
        # 1. 模拟IMDb评分（与现有评分相关但略有差异）
        features_df['imdb_rating'] = features_df['avg_rating'] + np.random.normal(0, 0.3, len(features_df))
        features_df['imdb_rating'] = features_df['imdb_rating'].clip(1, 5)  # 限制在1-5范围内
        
        # 2. 模拟是否有IMDb评分（大多数电影都有）
        features_df['has_imdb_rating'] = np.random.choice([0, 1], len(features_df), p=[0.1, 0.9])
        
        # 3. 模拟简介长度（与电影年份和类型数量相关）
        base_length = 100
        features_df['summary_length'] = (
            base_length + 
            features_df['genre_count'] * 20 + 
            (2024 - features_df['year']) * 2 +
            np.random.normal(0, 50, len(features_df))
        ).astype(int).clip(50, 500)
        
        # 4. 模拟导演数量
        features_df['director_count'] = np.random.choice([1, 2, 3], len(features_df), p=[0.7, 0.25, 0.05])
        
        print(f"   添加了 4 个合成IMDb特征")
        return features_df
    
    def prepare_modeling_features(self, df, use_synthetic_imdb=True):
        """准备建模特征"""
        print("🔧 准备建模特征...")
        
        # 1. 创建基础特征
        features_df = self.create_basic_features(df)
        
        # 2. 可选：添加合成IMDb特征
        if use_synthetic_imdb:
            features_df = self.add_synthetic_imdb_features(features_df)
        
        # 3. 选择特征列
        # 排除非特征列
        exclude_cols = ['movieId', 'title', 'genres', 'genres_list', 'first_rating_date', 'last_rating_date']
        
        # 选择数值型特征
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        # 4. 处理目标变量
        features_df = self.create_target_variable(features_df)
        
        print(f"   最终特征数量: {len(feature_cols)}")
        print(f"   特征列: {feature_cols}")
        
        return features_df, feature_cols
    
    def create_target_variable(self, df):
        """创建目标变量"""
        print("🎯 创建目标变量...")
        
        # 方法1: 回归任务 - 直接使用平均评分
        df['target_regression'] = df['avg_rating']
        
        # 方法2: 分类任务 - 将评分分为3个类别
        # 0: 低分(0-2.5), 1: 中分(2.5-4), 2: 高分(4-5)
        conditions = [
            df['avg_rating'] <= 2.5,
            (df['avg_rating'] > 2.5) & (df['avg_rating'] <= 4.0),
            df['avg_rating'] > 4.0
        ]
        choices = [0, 1, 2]  # 低, 中, 高
        df['target_classification'] = np.select(conditions, choices, default=1)
        
        # 方法3: 二分类 - 是否高于平均分
        rating_mean = df['avg_rating'].mean()
        df['target_binary'] = (df['avg_rating'] > rating_mean).astype(int)
        
        print(f"   目标变量分布:")
        print(f"   - 回归目标范围: {df['target_regression'].min():.2f} - {df['target_regression'].max():.2f}")
        print(f"   - 分类目标分布: {df['target_classification'].value_counts().sort_index().to_dict()}")
        print(f"   - 二分类分布: {df['target_binary'].value_counts().sort_index().to_dict()}")
        
        return df

class MovieDataset(Dataset):
    """PyTorch数据集类"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets.dtype == float else torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class RatingPredictor(nn.Module):
    """电影评分预测模型"""
    
    def __init__(self, input_size, output_type='regression', hidden_sizes=[128, 64, 32]):
        super().__init__()
        self.output_type = output_type
        
        # 构建动态网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        
        # 输出层
        if output_type == 'regression':
            self.output_layer = nn.Linear(prev_size, 1)
        elif output_type == 'binary':
            self.output_layer = nn.Linear(prev_size, 1)
            self.sigmoid = nn.Sigmoid()
        else:  # multiclass
            self.output_layer = nn.Linear(prev_size, 3)  # 3个类别
    
    def forward(self, x):
        x = self.network(x)
        x = self.output_layer(x)
        
        if self.output_type == 'binary':
            x = self.sigmoid(x)
        
        return x.squeeze()

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def train_model(self, train_loader, val_loader, output_type='regression', 
                   epochs=100, lr=0.001):
        """训练模型"""
        
        # 选择损失函数和优化器
        if output_type == 'regression':
            criterion = nn.MSELoss()
        elif output_type == 'binary':
            criterion = nn.BCELoss()
        else:  # multiclass
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        train_losses = []
        val_losses = []
        val_metrics = []
        
        print("🚀 开始训练模型...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                
                if output_type == 'multiclass':
                    loss = criterion(outputs, batch_targets.long())
                else:
                    loss = criterion(outputs, batch_targets)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_features)
                    
                    if output_type == 'multiclass':
                        loss = criterion(outputs, batch_targets.long())
                        predictions = torch.argmax(outputs, dim=1)
                    else:
                        loss = criterion(outputs, batch_targets)
                        predictions = outputs
                    
                    val_loss += loss.item()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch_targets.cpu().numpy())
            
            # 计算指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 计算验证集指标
            val_metric = self.calculate_metrics(all_predictions, all_targets, output_type)
            val_metrics.append(val_metric)
            
            scheduler.step(avg_val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, '
                      f'Val Metric: {val_metric:.4f}')
        
        return train_losses, val_losses, val_metrics
    
    def calculate_metrics(self, predictions, targets, output_type):
        """计算评估指标"""
        if output_type == 'regression':
            return mean_absolute_error(targets, predictions)
        else:
            return accuracy_score(targets, predictions)
    
    def evaluate_model(self, data_loader, output_type='regression'):
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in data_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_features)
                
                if output_type == 'multiclass':
                    predictions = torch.argmax(outputs, dim=1)
                elif output_type == 'binary':
                    predictions = (outputs > 0.5).int()
                else:  # regression
                    predictions = outputs
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        
        return all_predictions, all_targets

def run_experiment(features, targets, output_type='regression', experiment_num=1):
    """运行单次实验"""
    print(f"\n🔬 实验 {experiment_num} - {output_type.upper()} 任务")
    
    # 创建数据集
    dataset = MovieDataset(features, targets)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    input_size = features.shape[1]
    model = RatingPredictor(input_size, output_type)
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModelTrainer(model, device)
    
    train_losses, val_losses, val_metrics = trainer.train_model(
        train_loader, val_loader, output_type, epochs=50, lr=0.001
    )
    
    # 最终评估
    predictions, true_targets = trainer.evaluate_model(val_loader, output_type)
    
    if output_type == 'regression':
        mae = mean_absolute_error(true_targets, predictions)
        print(f"✅ 实验 {experiment_num} 完成 - MAE: {mae:.4f}")
        return mae
    else:
        accuracy = accuracy_score(true_targets, predictions)
        print(f"✅ 实验 {experiment_num} 完成 - 准确率: {accuracy:.4f}")
        return accuracy

def main():
    """主函数"""
    print("🎬 电影评分预测模型训练开始...")
    
    # 运行特征工程和建模
    engineer = MovieFeatureEngineer()
    
    try:
        movie_data = engineer.load_processed_data()
    except FileNotFoundError:
        print("❌ 找不到处理后的数据文件，请先运行数据预处理模块")
        return
    
    # 准备特征（使用合成IMDb数据）
    features_df, feature_cols = engineer.prepare_modeling_features(
        movie_data, use_synthetic_imdb=True
    )
    
    # 选择特征和目标
    X = features_df[feature_cols].values
    y_regression = features_df['target_regression'].values
    y_classification = features_df['target_classification'].values
    y_binary = features_df['target_binary'].values
    
    print(f"\n📊 数据准备完成:")
    print(f"   特征矩阵形状: {X.shape}")
    print(f"   回归目标形状: {y_regression.shape}")
    print(f"   分类目标形状: {y_classification.shape}")
    print(f"   二分类目标形状: {y_binary.shape}")
    
    # 运行不同任务的实验
    results = {}
    
    # 实验1: 回归任务
    results['regression'] = run_experiment(X, y_regression, 'regression', 1)
    
    # 实验2: 多分类任务
    results['classification'] = run_experiment(X, y_classification, 'multiclass', 2)
    
    # 实验3: 二分类任务
    results['binary'] = run_experiment(X, y_binary, 'binary', 3)
    
    print(f"\n🎯 所有实验完成!")
    print(f"   回归任务 MAE: {results['regression']:.4f}")
    print(f"   多分类准确率: {results['classification']:.4f}")
    print(f"   二分类准确率: {results['binary']:.4f}")

if __name__ == "__main__":
    main()