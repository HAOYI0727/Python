# """
# 运行数据预处理流程，生成特征工程所需文件
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# import sys

# # 添加src目录到Python路径
# project_path = Path("D:/VSCodeProjects/PythonCourse")
# src_path = project_path / "src"
# sys.path.append(str(src_path))

# def complete_preprocessing():
#     """完成数据预处理流程"""
#     print("🔄 开始数据预处理流程...")
    
#     # 加载原始数据
#     raw_path = project_path / "data" / "raw"
#     movies = pd.read_csv(raw_path / "movies.csv")
#     ratings = pd.read_csv(raw_path / "ratings.csv")
#     links = pd.read_csv(raw_path / "links.csv")
#     tags = pd.read_csv(raw_path / "tags.csv")
    
#     print("✅ 原始数据加载完成")
#     print(f"   Movies: {movies.shape}")
#     print(f"   Ratings: {ratings.shape}")
#     print(f"   Links: {links.shape}")
#     print(f"   Tags: {tags.shape}")
    
#     # 数据预处理步骤
#     processed_data = preprocess_data(movies, ratings, links, tags)
    
#     # 保存处理后的数据
#     save_processed_data(processed_data)
    
#     print("🎉 数据预处理完成！")

# def preprocess_data(movies, ratings, links, tags):
#     """数据预处理"""
#     print("\n🔧 开始数据预处理...")
    
#     # 1. 处理电影数据
#     movies_processed = process_movies(movies)
    
#     # 2. 处理评分数据
#     ratings_processed = process_ratings(ratings)
    
#     # 3. 创建电影特征数据集
#     movie_features = create_movie_features(movies_processed, ratings_processed, links)
    
#     # 4. 处理缺失值
#     movie_features_clean = handle_missing_values(movie_features)
    
#     return {
#         'movies_processed': movies_processed,
#         'ratings_processed': ratings_processed,
#         'movie_features': movie_features,
#         'movie_features_clean': movie_features_clean
#     }

# def process_movies(movies):
#     """处理电影数据"""
#     print("   🎬 处理电影数据...")
    
#     # 创建副本
#     movies_proc = movies.copy()
    
#     # 从标题中提取年份
#     movies_proc['year'] = movies_proc['title'].str.extract(r'\((\d{4})\)')
#     movies_proc['year'] = pd.to_numeric(movies_proc['year'], errors='coerce')
    
#     # 填充年份缺失值
#     year_median = movies_proc['year'].median()
#     movies_proc['year'].fillna(year_median, inplace=True)
    
#     # 处理电影类型
#     movies_proc['genres_list'] = movies_proc['genres'].str.split('|')
    
#     # 创建类型虚拟变量
#     all_genres = set()
#     for genres in movies_proc['genres_list'].dropna():
#         all_genres.update(genres)
    
#     for genre in all_genres:
#         movies_proc[f'genre_{genre}'] = movies_proc['genres_list'].apply(
#             lambda x: 1 if genre in x else 0
#         )
    
#     print(f"      创建了 {len(all_genres)} 种电影类型编码")
#     return movies_proc

# def process_ratings(ratings):
#     """处理评分数据"""
#     print("   ⭐ 处理评分数据...")
    
#     # 移除重复评分（同一用户对同一电影的多次评分）
#     initial_count = len(ratings)
#     ratings_proc = ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last')
#     removed_count = initial_count - len(ratings_proc)
    
#     print(f"      移除了 {removed_count} 条重复评分")
#     return ratings_proc

# def create_movie_features(movies, ratings, links):
#     """创建电影特征数据集"""
#     print("   🏗️ 创建电影特征数据集...")
    
#     # 计算每个电影的评分统计
#     movie_stats = ratings.groupby('movieId').agg({
#         'rating': ['mean', 'count', 'std', 'min', 'max'],
#         'userId': 'nunique',
#         'timestamp': ['min', 'max']
#     }).round(3)
    
#     # 扁平化列名
#     movie_stats.columns = [
#         'avg_rating', 'rating_count', 'rating_std', 
#         'min_rating', 'max_rating', 'unique_users',
#         'first_rating_date', 'last_rating_date'
#     ]
    
#     # 计算评分时间跨度（天）
#     movie_stats['rating_period_days'] = (
#         movie_stats['last_rating_date'] - movie_stats['first_rating_date']
#     ) / (24 * 3600)
    
#     # 合并电影基本信息和评分统计
#     movie_features = movies.merge(
#         movie_stats, 
#         left_on='movieId', 
#         right_index=True, 
#         how='left'
#     )
    
#     # 合并链接信息
#     movie_features = movie_features.merge(
#         links, 
#         on='movieId', 
#         how='left'
#     )
    
#     print(f"      创建的特征数据集: {movie_features.shape}")
#     return movie_features

# def handle_missing_values(movie_features):
#     """处理缺失值"""
#     print("   🧹 处理缺失值...")
    
#     # 创建副本
#     movie_features_clean = movie_features.copy()
    
#     # 定义填充策略
#     fill_strategy = {
#         # 评分相关列：缺失表示无评分，填充0
#         'avg_rating': 0,
#         'min_rating': 0, 
#         'max_rating': 0,
#         'rating_std': 0,
#         'unique_users': 0,
#         'rating_count': 0,
#         'first_rating_date': 0,
#         'last_rating_date': 0, 
#         'rating_period_days': 0,
#         # 外部ID：填充0
#         'imdbId': 0,
#         'tmdbId': 0,
#         # 年份：填充中位数
#         'year': movie_features['year'].median()
#     }
    
#     # 应用填充策略
#     for col, value in fill_strategy.items():
#         if col in movie_features_clean.columns:
#             before = movie_features_clean[col].isnull().sum()
#             movie_features_clean[col] = movie_features_clean[col].fillna(value)
#             after = movie_features_clean[col].isnull().sum()
#             if before > 0:
#                 print(f"      填充 {col}: {before} → {after} 个缺失值")
    
#     # 检查剩余缺失值
#     remaining_missing = movie_features_clean.isnull().sum().sum()
#     print(f"      剩余缺失值总数: {remaining_missing}")
    
#     return movie_features_clean

# def save_processed_data(processed_data):
#     """保存处理后的数据"""
#     print("\n💾 保存处理后的数据...")
    
#     processed_path = project_path / "data" / "processed"
#     processed_path.mkdir(parents=True, exist_ok=True)
    
#     # 保存各个数据文件
#     processed_data['movies_processed'].to_csv(processed_path / "movies_processed.csv", index=False)
#     processed_data['ratings_processed'].to_csv(processed_path / "ratings_processed.csv", index=False)
#     processed_data['movie_features'].to_csv(processed_path / "movie_features.csv", index=False)
#     processed_data['movie_features_clean'].to_csv(processed_path / "movie_features_clean.csv", index=False)
    
#     print("✅ 数据保存完成:")
#     print(f"   - movies_processed.csv")
#     print(f"   - ratings_processed.csv") 
#     print(f"   - movie_features.csv")
#     print(f"   - movie_features_clean.csv")
    
#     # 数据质量报告
#     movie_features_clean = processed_data['movie_features_clean']
#     print(f"\n📊 最终数据质量:")
#     print(f"   数据集形状: {movie_features_clean.shape}")
#     print(f"   总缺失值: {movie_features_clean.isnull().sum().sum()}")
#     print(f"   电影数量: {len(movie_features_clean)}")
#     print(f"   特征数量: {len(movie_features_clean.columns)}")

# if __name__ == "__main__":
#     complete_preprocessing()