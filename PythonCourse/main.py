"""
电影评分预测项目 - 主程序入口
整合所有六个主要模块，包含完整的数据预处理功能
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加src目录到Python路径
project_path = Path("D:/VSCodeProjects/PythonCourse")
src_path = project_path / "src"
sys.path.append(str(src_path))

def setup_environment():
    """设置项目环境"""
    print("🎬 电影评分预测项目")
    print("=" * 60)
    
    # 检查必要目录
    necessary_dirs = [
        project_path / "data" / "raw",
        project_path / "data" / "processed", 
        project_path / "results" / "figures",
        project_path / "models"
    ]
    
    for dir_path in necessary_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 检查数据文件
    data_files = list((project_path / "data" / "raw").glob("*.csv"))
    if not data_files:
        print("❌ 未找到数据文件，请确保以下文件存在于 data/raw/ 目录:")
        print("   - movies.csv")
        print("   - ratings.csv") 
        print("   - links.csv")
        print("   - tags.csv")
        return False
    
    print("✅ 环境检查完成")
    return True

def complete_preprocessing():
    """完成数据预处理流程 - 集成自 run_preprocessing.py"""
    print("🔄 开始数据预处理流程...")
    
    # 加载原始数据
    raw_path = project_path / "data" / "raw"
    try:
        movies = pd.read_csv(raw_path / "movies.csv")
        ratings = pd.read_csv(raw_path / "ratings.csv")
        links = pd.read_csv(raw_path / "links.csv")
        tags = pd.read_csv(raw_path / "tags.csv")
    except FileNotFoundError as e:
        print(f"❌ 数据文件加载失败: {e}")
        return False
    
    print("✅ 原始数据加载完成")
    print(f"   Movies: {movies.shape}")
    print(f"   Ratings: {ratings.shape}")
    print(f"   Links: {links.shape}")
    print(f"   Tags: {tags.shape}")
    
    # 数据预处理步骤
    processed_data = preprocess_data(movies, ratings, links, tags)
    
    # 保存处理后的数据
    save_processed_data(processed_data)
    
    print("🎉 数据预处理完成！")
    return True

def preprocess_data(movies, ratings, links, tags):
    """数据预处理"""
    print("\n🔧 开始数据预处理...")
    
    # 1. 处理电影数据
    movies_processed = process_movies(movies)
    
    # 2. 处理评分数据
    ratings_processed = process_ratings(ratings)
    
    # 3. 创建电影特征数据集
    movie_features = create_movie_features(movies_processed, ratings_processed, links)
    
    # 4. 处理缺失值
    movie_features_clean = handle_missing_values(movie_features)
    
    return {
        'movies_processed': movies_processed,
        'ratings_processed': ratings_processed,
        'movie_features': movie_features,
        'movie_features_clean': movie_features_clean
    }

def process_movies(movies):
    """处理电影数据"""
    print("   🎬 处理电影数据...")
    
    # 创建副本
    movies_proc = movies.copy()
    
    # 从标题中提取年份
    movies_proc['year'] = movies_proc['title'].str.extract(r'\((\d{4})\)')
    movies_proc['year'] = pd.to_numeric(movies_proc['year'], errors='coerce')
    
    # 填充年份缺失值
    year_median = movies_proc['year'].median()
    movies_proc['year'] = movies_proc['year'].fillna(year_median)
    
    # 处理电影类型
    movies_proc['genres_list'] = movies_proc['genres'].str.split('|')
    
    # 创建类型虚拟变量
    all_genres = set()
    for genres in movies_proc['genres_list'].dropna():
        if isinstance(genres, list):
            all_genres.update(genres)
    
    for genre in all_genres:
        movies_proc[f'genre_{genre}'] = movies_proc['genres_list'].apply(
            lambda x: 1 if genre in x else 0
        )
    
    print(f"      创建了 {len(all_genres)} 种电影类型编码")
    return movies_proc

def process_ratings(ratings):
    """处理评分数据"""
    print("   ⭐ 处理评分数据...")
    
    # 移除重复评分（同一用户对同一电影的多次评分）
    initial_count = len(ratings)
    ratings_proc = ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last')
    removed_count = initial_count - len(ratings_proc)
    
    print(f"      移除了 {removed_count} 条重复评分")
    return ratings_proc

def create_movie_features(movies, ratings, links):
    """创建电影特征数据集"""
    print("   🏗️ 创建电影特征数据集...")
    
    # 计算每个电影的评分统计
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count', 'std', 'min', 'max'],
        'userId': 'nunique',
        'timestamp': ['min', 'max']
    }).round(3)
    
    # 扁平化列名
    movie_stats.columns = [
        'avg_rating', 'rating_count', 'rating_std', 
        'min_rating', 'max_rating', 'unique_users',
        'first_rating_date', 'last_rating_date'
    ]
    
    # 计算评分时间跨度（天）
    movie_stats['rating_period_days'] = (
        movie_stats['last_rating_date'] - movie_stats['first_rating_date']
    ) / (24 * 3600)
    
    # 合并电影基本信息和评分统计
    movie_features = movies.merge(
        movie_stats, 
        left_on='movieId', 
        right_index=True, 
        how='left'
    )
    
    # 合并链接信息
    movie_features = movie_features.merge(
        links, 
        on='movieId', 
        how='left'
    )
    
    print(f"      创建的特征数据集: {movie_features.shape}")
    return movie_features

def handle_missing_values(movie_features):
    """处理缺失值"""
    print("   🧹 处理缺失值...")
    
    # 创建副本
    movie_features_clean = movie_features.copy()
    
    # 定义填充策略
    fill_strategy = {
        # 评分相关列：缺失表示无评分，填充0
        'avg_rating': 0,
        'min_rating': 0, 
        'max_rating': 0,
        'rating_std': 0,
        'unique_users': 0,
        'rating_count': 0,
        'first_rating_date': 0,
        'last_rating_date': 0, 
        'rating_period_days': 0,
        # 外部ID：填充0
        'imdbId': 0,
        'tmdbId': 0,
        # 年份：填充中位数
        'year': movie_features['year'].median()
    }
    
    # 应用填充策略
    for col, value in fill_strategy.items():
        if col in movie_features_clean.columns:
            before = movie_features_clean[col].isnull().sum()
            movie_features_clean[col] = movie_features_clean[col].fillna(value)
            after = movie_features_clean[col].isnull().sum()
            if before > 0:
                print(f"      填充 {col}: {before} → {after} 个缺失值")
    
    # 检查剩余缺失值
    remaining_missing = movie_features_clean.isnull().sum().sum()
    print(f"      剩余缺失值总数: {remaining_missing}")
    
    return movie_features_clean

def save_processed_data(processed_data):
    """保存处理后的数据"""
    print("\n💾 保存处理后的数据...")
    
    processed_path = project_path / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # 保存各个数据文件
    processed_data['movies_processed'].to_csv(processed_path / "movies_processed.csv", index=False)
    processed_data['ratings_processed'].to_csv(processed_path / "ratings_processed.csv", index=False)
    processed_data['movie_features'].to_csv(processed_path / "movie_features.csv", index=False)
    processed_data['movie_features_clean'].to_csv(processed_path / "movie_features_clean.csv", index=False)
    
    print("✅ 数据保存完成:")
    print(f"   - movies_processed.csv")
    print(f"   - ratings_processed.csv") 
    print(f"   - movie_features.csv")
    print(f"   - movie_features_clean.csv")
    
    # 数据质量报告
    movie_features_clean = processed_data['movie_features_clean']
    print(f"\n📊 最终数据质量:")
    print(f"   数据集形状: {movie_features_clean.shape}")
    print(f"   总缺失值: {movie_features_clean.isnull().sum().sum()}")
    print(f"   电影数量: {len(movie_features_clean)}")
    print(f"   特征数量: {len(movie_features_clean.columns)}")

def run_data_exploration():
    """运行数据探索模块"""
    print("\n📊 第一步：数据探索")
    print("-" * 30)
    
    try:
        # 这里可以调用数据探索相关的函数
        # 或者提示用户查看notebook
        print("正在执行数据探索分析...")
        print("请查看 notebooks/01_data_exploration.ipynb 获取完整分析结果")
        
        # 快速数据概览
        raw_path = project_path / "data" / "raw"
        movies = pd.read_csv(raw_path / "movies.csv")
        ratings = pd.read_csv(raw_path / "ratings.csv")
        
        print("📈 数据概览:")
        print(f"   电影数量: {len(movies)}")
        print(f"   评分数量: {len(ratings)}")
        print(f"   用户数量: {ratings['userId'].nunique()}")
        print(f"   评分时间范围: {pd.to_datetime(ratings['timestamp'], unit='s').min()} 到 {pd.to_datetime(ratings['timestamp'], unit='s').max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据探索失败: {e}")
        return False

def run_data_preprocessing():
    """运行数据预处理模块"""
    print("\n🔧 第二步：数据预处理")
    print("-" * 30)
    
    try:
        return complete_preprocessing()
        
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        return False

def run_efficient_models():
    """运行高效模型模块"""
    print("\n🤖 第三步：高效模型训练")
    print("-" * 30)
    
    try:
        from efficient_models import EfficientModelTrainer, run_multiple_experiments
        from feature_engineering import MovieFeatureEngineer
        
        print("加载处理后的数据...")
        engineer = MovieFeatureEngineer()
        movie_data = engineer.load_processed_data()
        features_df, feature_cols = engineer.prepare_modeling_features(movie_data)
        
        print(f"数据集: {len(features_df)} 样本, {len(feature_cols)} 特征")
        print("开始高效模型训练...")
        
        regression_results, classification_results = run_multiple_experiments(
            features_df, feature_cols, num_experiments=3
        )
        
        print("✅ 高效模型训练完成!")
        
        return True
        
    except Exception as e:
        print(f"❌ 高效模型训练失败: {e}")
        return False

def run_optimized_models():
    """运行优化版模型模块"""
    print("\n🚀 第四步：优化版模型训练")
    print("-" * 30)
    
    try:
        from optimized_models import OptimizedModelTrainer, run_optimized_multiple_experiments
        from feature_engineering import MovieFeatureEngineer
        
        print("加载处理后的数据...")
        engineer = MovieFeatureEngineer()
        movie_data = engineer.load_processed_data()
        features_df, feature_cols = engineer.prepare_modeling_features(movie_data)
        
        print(f"数据集: {len(features_df)} 样本, {len(feature_cols)} 特征")
        print("开始优化版模型训练...")
        
        regression_results, classification_results = run_optimized_multiple_experiments(
            features_df, feature_cols, num_experiments=3
        )
        
        print("✅ 优化版模型训练完成!")
        
        return True
        
    except Exception as e:
        print(f"❌ 优化版模型训练失败: {e}")
        return False

def run_model_validation():
    """运行模型验证模块"""
    print("\n🎯 第五步：模型验证")
    print("-" * 30)
    
    try:
        from model_validation import run_complete_validation
        from feature_engineering import MovieFeatureEngineer
        
        print("加载处理后的数据...")
        engineer = MovieFeatureEngineer()
        movie_data = engineer.load_processed_data()
        features_df, feature_cols = engineer.prepare_modeling_features(movie_data)
        
        print("开始三次随机划分验证...")
        validation_results = run_complete_validation(features_df, feature_cols)
        
        print("✅ 模型验证完成!")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型验证失败: {e}")
        return False

def run_optimized_validation():
    """运行优化版验证模块"""
    print("\n🔍 第六步：优化版模型验证")
    print("-" * 30)
    
    try:
        from optimized_model_validation import run_optimized_complete_validation
        from feature_engineering import MovieFeatureEngineer
        
        print("加载处理后的数据...")
        engineer = MovieFeatureEngineer()
        movie_data = engineer.load_processed_data()
        features_df, feature_cols = engineer.prepare_modeling_features(movie_data)
        
        print("开始优化版三次随机划分验证...")
        validation_results = run_optimized_complete_validation(features_df, feature_cols)
        
        print("✅ 优化版模型验证完成!")
        
        return True
        
    except Exception as e:
        print(f"❌ 优化版模型验证失败: {e}")
        return False

def run_single_module(module_choice):
    """运行单个模块"""
    modules = {
        '1': ("数据探索", run_data_exploration),
        '2': ("数据预处理", run_data_preprocessing),
        '3': ("高效模型", run_efficient_models),
        '4': ("优化版模型", run_optimized_models),
        '5': ("模型验证", run_model_validation),
        '6': ("优化版验证", run_optimized_validation)
    }
    
    if module_choice in modules:
        module_name, module_function = modules[module_choice]
        print(f"\n🎯 运行 {module_name} 模块")
        print("=" * 50)
        return module_function()
    else:
        print("❌ 无效的选择")
        return False

def run_complete_pipeline():
    """运行完整流程"""
    print("\n🚀 开始完整项目流程")
    print("=" * 60)
    
    # 检查环境
    if not setup_environment():
        return
    
    steps = [
        ("数据探索", run_data_exploration),
        ("数据预处理", run_data_preprocessing),
        ("高效模型训练", run_efficient_models),
        ("优化版模型训练", run_optimized_models),
        ("模型验证", run_model_validation),
        ("优化版验证", run_optimized_validation)
    ]
    
    successful_steps = 0
    total_steps = len(steps)
    
    for step_name, step_function in steps:
        print(f"\n🔹 步骤 {successful_steps + 1}/{total_steps}: {step_name}")
        print("-" * 40)
        
        try:
            if step_function():
                successful_steps += 1
                print(f"✅ {step_name} - 完成")
            else:
                print(f"❌ {step_name} - 失败")
                break
        except Exception as e:
            print(f"❌ {step_name} - 错误: {e}")
            break
    
    print(f"\n🎉 流程执行完成: {successful_steps}/{total_steps} 个步骤成功")
    
    if successful_steps == total_steps:
        print("✅ 所有步骤均成功完成!")
    else:
        print("⚠️ 部分步骤未能完成，请检查错误信息")

def display_menu():
    """显示主菜单"""
    print("\n请选择要执行的操作:")
    print("1. 数据探索")
    print("2. 数据预处理") 
    print("3. 高效模型训练")
    print("4. 优化版模型训练")
    print("5. 模型验证")
    print("6. 优化版模型验证")
    print("7. 运行完整流程")
    print("8. 退出")

def main():
    """主函数"""
    
    # 环境检查
    if not setup_environment():
        print("❌ 环境设置失败，请检查数据文件")
        return
    
    while True:
        display_menu()
        choice = input("\n请输入选择 (1-8): ").strip()
        
        if choice == '1':
            run_single_module('1')
        elif choice == '2':
            run_single_module('2')
        elif choice == '3':
            run_single_module('3')
        elif choice == '4':
            run_single_module('4')
        elif choice == '5':
            run_single_module('5')
        elif choice == '6':
            run_single_module('6')
        elif choice == '7':
            run_complete_pipeline()
        elif choice == '8':
            print("👋 感谢使用电影评分预测项目!")
            break
        else:
            print("❌ 无效选择，请重新输入")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        print("请检查数据文件和环境配置")