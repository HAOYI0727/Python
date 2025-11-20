"""
data_preprocessing.py
数据预处理模块
处理缺失值、重复值、多用户评分问题、文本标签编码等
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, project_path="D:/VSCodeProjects/PythonCourse"):
        self.project_path = Path(project_path)
        self.data_path = self.project_path / "data"
        self.raw_data_path = self.data_path / "raw"
        self.processed_data_path = self.data_path / "processed"
        
        # 创建必要的目录
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        # 初始化编码器
        self.genre_encoder = MultiLabelBinarizer()
        self.label_encoders = {}
        
    def load_raw_data(self):
        # 加载原始数据
        print("加载原始数据...")
        self.movies = pd.read_csv(self.raw_data_path / "movies.csv")
        self.ratings = pd.read_csv(self.raw_data_path / "ratings.csv")
        self.links = pd.read_csv(self.raw_data_path / "links.csv")
        self.tags = pd.read_csv(self.raw_data_path / "tags.csv")
        
        print(f"数据加载完成:")
        print(f" - Movies: {self.movies.shape}")
        print(f" - Ratings: {self.ratings.shape}")
        print(f" - Links: {self.links.shape}")
        print(f" - Tags: {self.tags.shape}")
        
        return self.movies, self.ratings, self.links, self.tags
    
    def check_missing_values(self):
        # 检查缺失值
        print("\n检查缺失值...")
        datasets = { 'movies': self.movies, 'ratings': self.ratings, 
                     'links': self.links, 'tags': self.tags }
        missing_summary = {}
        for name, df in datasets.items():
            missing = df.isnull().sum()
            missing_percent = (missing / len(df)) * 100
            missing_info = pd.DataFrame({'missing_count': missing, 'missing_percent': missing_percent})
            missing_info = missing_info[missing_info['missing_count'] > 0]
            missing_summary[name] = missing_info
            
            if len(missing_info) > 0:
                print(f"{name} 缺失值:")
                for col in missing_info.index:
                    count = missing_info.loc[col, 'missing_count']
                    percent = missing_info.loc[col, 'missing_percent']
                    print(f" - {col}: {count} ({percent:.2f}%)")
            else:
                print(f"{name}: 无缺失值")
                
        return missing_summary
    
    def handle_missing_values(self):
        # 处理缺失值
        print("\n处理缺失值...")
        # 处理movies中的年份缺失
        if 'year' not in self.movies.columns:
            self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)')
        self.movies['year'] = pd.to_numeric(self.movies['year'], errors='coerce')
        year_median = self.movies['year'].median()
        self.movies['year'] = self.movies['year'].fillna(year_median)
        print(f"   - 填充电影年份缺失值: {year_median}")
        
        # 处理links中的外部ID缺失
        self.links['imdbId'] = self.links['imdbId'].fillna(0)
        self.links['tmdbId'] = self.links['tmdbId'].fillna(0)
        print("   - 填充外部链接ID缺失值为0")
        
        # 处理tags中的timestamp缺失（如果有）
        if 'timestamp' in self.tags.columns and self.tags['timestamp'].isnull().any():
            self.tags['timestamp'] = self.tags['timestamp'].fillna(0)
            print("   - 填充标签时间戳缺失值为0")
    
    def check_duplicates(self):
        # 检查重复值
        print("\n检查重复值...")
        movies_dup = self.movies[['movieId', 'title', 'genres']].duplicated().sum()
        print(f" - Movies重复行: {movies_dup}")
        ratings_dup = self.ratings.duplicated().sum()
        print(f" - Ratings重复行: {ratings_dup}")
        links_dup = self.links.duplicated().sum()
        print(f" - Links重复行: {links_dup}")
        tags_dup = self.tags.duplicated().sum()
        print(f" - Tags重复行: {tags_dup}")
        
        return {'movies': movies_dup, 'ratings': ratings_dup, 'links': links_dup, 'tags': tags_dup}
    
    def handle_duplicates(self):
        # 处理重复值
        print("\n处理重复值...")
        # 移除重复的评分记录（同一用户对同一电影的多次评分）
        initial_count = len(self.ratings)
        self.ratings = self.ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last')
        removed_count = initial_count - len(self.ratings)
        print(f" - 移除重复评分记录: {removed_count} 条")
        
        # 移除重复的标签记录
        initial_tags = len(self.tags)
        self.tags = self.tags.drop_duplicates()
        removed_tags = initial_tags - len(self.tags)
        print(f" - 移除重复标签记录: {removed_tags} 条")
        
        # 确保movies和links没有重复的movieId
        self.movies = self.movies.drop_duplicates(subset=['movieId'], keep='first')
        self.links = self.links.drop_duplicates(subset=['movieId'], keep='first')
        print(f" - 确保movies和links的movieId唯一")
    
    def handle_multiple_ratings(self):
        # 处理多用户评分问题 - 聚合评分数据
        print("\n处理多用户评分问题...")
        # 分析每个电影的评分数量
        movie_rating_stats = self.ratings.groupby('movieId').agg({'rating': ['count', 'mean', 'std'], 'userId': 'nunique'}).round(3)
        movie_rating_stats.columns = ['rating_count', 'rating_mean', 'rating_std', 'unique_users']
        print(f" - 电影评分统计:")
        print(f"   平均每个电影评分数: {movie_rating_stats['rating_count'].mean():.2f}")
        print(f"   评分最多的电影: {movie_rating_stats['rating_count'].max()} 个评分")
        print(f"   评分最少的电影: {movie_rating_stats['rating_count'].min()} 个评分")
        
        # 可视化评分数量分布
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        movie_rating_stats['rating_count'].hist(bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Movie Rating Count Distribution')
        plt.xlabel('Number of Ratings per Movie')
        plt.ylabel('Number of Movies')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        # 显示评分数量与平均评分的关系
        plt.scatter(movie_rating_stats['rating_count'], movie_rating_stats['rating_mean'], 
                   alpha=0.5, color='coral')
        plt.title('Rating Count vs Average Rating')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Average Rating')
        plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.project_path / "results/figures/rating_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return movie_rating_stats
    
    def encode_genres(self):
        # 对电影类型进行编码
        print("\n对电影类型进行编码...")
        # 解析电影类型
        self.movies['genres_list'] = self.movies['genres'].str.split('|')
        # 获取所有唯一的电影类型
        all_genres = set()
        for genres in self.movies['genres_list']:
            if isinstance(genres, list):
                all_genres.update(genres)
        print(f" - 发现 {len(all_genres)} 种电影类型: {sorted(all_genres)}")
        
        # 为每个类型创建虚拟变量
        for genre in all_genres:
            self.movies[f'genre_{genre}'] = self.movies['genres_list'].apply(lambda x: 1 if genre in x else 0)
        # 统计类型分布
        genre_counts = {}
        for genre in all_genres:
            genre_counts[genre] = self.movies[f'genre_{genre}'].sum()
        print(f" - 电影类型分布:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {genre}: {count} 部电影")
        
        return all_genres, genre_counts
    
    def create_movie_features(self):
        # 创建电影特征数据集
        print("\n创建电影特征数据集...")
        
        # 计算每个电影的评分统计
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std', 'min', 'max'],
            'userId': 'nunique', 'timestamp': ['min', 'max']}).round(3)
        # 扁平化列名
        movie_stats.columns = ['avg_rating', 'rating_count', 'rating_std', 'min_rating', 'max_rating', 
                               'unique_users', 'first_rating_date', 'last_rating_date']
        
        # 计算评分时间跨度（天）
        movie_stats['rating_period_days'] = (movie_stats['last_rating_date'] - movie_stats['first_rating_date']) / (24 * 3600)
        # 合并电影基本信息和评分统计
        movie_features = self.movies.merge(movie_stats, left_on='movieId', right_index=True, how='left')
        # 合并链接信息
        movie_features = movie_features.merge(self.links, on='movieId', how='left')
        
        # 处理合并后的缺失值
        numeric_cols = ['avg_rating', 'rating_count', 'rating_std', 'min_rating', 'max_rating', 
                        'unique_users', 'first_rating_date', 'last_rating_date', 'rating_period_days']
        # 对于评分相关列，缺失意味着没有评分，填充0
        for col in numeric_cols:
            if col in movie_features.columns:
                movie_features[col] = movie_features[col].fillna(0)
        print(f" - 创建特征数据集: {movie_features.shape}")
        print(f" - 特征列: {list(movie_features.columns)}")
        print(f" - 剩余缺失值: {movie_features.isnull().sum().sum()}")
        
        return movie_features
    
    def save_processed_data(self, movie_features):
        # 保存处理后的数据
        print("\n保存处理后的数据...")
        # 保存电影特征数据
        movie_features.to_csv(self.processed_data_path / "movie_features.csv", index=False)
        print(f" - 保存电影特征数据: movie_features.csv")
        # 保存处理后的原始数据
        self.movies.to_csv(self.processed_data_path / "movies_processed.csv", index=False)
        self.ratings.to_csv(self.processed_data_path / "ratings_processed.csv", index=False)
        self.links.to_csv(self.processed_data_path / "links_processed.csv", index=False)
        self.tags.to_csv(self.processed_data_path / "tags_processed.csv", index=False)
        print(" - 保存所有处理后的数据文件")
    
    def run_preprocessing_pipeline(self):
        # 运行完整的数据预处理流程
        print("=" * 60)
        print("开始数据预处理流程")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_raw_data()
        # 2. 检查和处理缺失值
        self.check_missing_values()
        self.handle_missing_values()
        # 3. 检查和处理重复值
        self.check_duplicates()
        self.handle_duplicates()
        # 4. 处理多用户评分问题
        rating_stats = self.handle_multiple_ratings()
        # 5. 文本标签编码
        genres, genre_counts = self.encode_genres()
        # 6. 创建特征数据集
        movie_features = self.create_movie_features()
        # 7. 保存处理后的数据
        self.save_processed_data(movie_features)
        
        print("\n" + "=" * 60)
        print("数据预处理完成!")
        print("=" * 60)
        
        # 返回处理结果摘要
        summary = {'final_movies_shape': self.movies.shape, 'final_ratings_shape': self.ratings.shape, 
                   'movie_features_shape': movie_features.shape, 'genres_count': len(genres), 'rating_stats': rating_stats}
        return summary

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    summary = preprocessor.run_preprocessing_pipeline()
    print("处理完成摘要:", summary)