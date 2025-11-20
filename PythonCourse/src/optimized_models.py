"""
优化版高效模型实现
在原有基础上增加更多模型和改进的深度学习网络
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                             GradientBoostingRegressor, GradientBoostingClassifier,
                             ExtraTreesRegressor, ExtraTreesClassifier,
                             AdaBoostRegressor, AdaBoostClassifier)
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class OptimizedModelTrainer:
    """优化版模型训练器"""
    
    def __init__(self):
        self.results = {}
        self.scaler = StandardScaler()
    
    def prepare_data(self, features_df, feature_cols, target_type='regression', test_size=0.2):
        """准备训练和测试数据"""
        X = features_df[feature_cols].values
        
        if target_type == 'regression':
            y = features_df['target_regression'].values
        elif target_type == 'multiclass':
            y = features_df['target_classification'].values
        else:  # binary
            y = features_df['target_binary'].values
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=y if target_type != 'regression' else None
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_advanced_sklearn_models(self, X_train, X_test, y_train, y_test, target_type='regression'):
        """训练更多sklearn模型"""
        print("🤖 训练高级Scikit-learn模型...")
        
        models = {}
        predictions = {}
        scores = {}
        
        # 扩展的模型配置
        if target_type == 'regression':
            model_configs = {
                # 树模型家族
                'RandomForest': RandomForestRegressor(
                    n_estimators=200, max_depth=20, min_samples_split=5, 
                    random_state=42, n_jobs=-1
                ),
                'ExtraTrees': ExtraTreesRegressor(
                    n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1, 
                    subsample=0.8, random_state=42
                ),
                'AdaBoost': AdaBoostRegressor(
                    n_estimators=100, learning_rate=0.1, random_state=42
                ),
                'DecisionTree': DecisionTreeRegressor(
                    max_depth=15, random_state=42
                ),
                
                # 线性模型
                'Ridge': Ridge(alpha=1.0, random_state=42),
                'Lasso': Lasso(alpha=0.1, random_state=42),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
                
                # 其他模型
                'KNN': KNeighborsRegressor(n_neighbors=7, weights='distance'),
                'SVM': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'MLP': MLPRegressor(
                    hidden_layer_sizes=(100, 50), activation='relu',
                    learning_rate_init=0.001, max_iter=500, random_state=42
                )
            }
        else:
            model_configs = {
                # 树模型家族
                'RandomForest': RandomForestClassifier(
                    n_estimators=200, max_depth=20, min_samples_split=5, 
                    random_state=42, n_jobs=-1
                ),
                'ExtraTrees': ExtraTreesClassifier(
                    n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1, 
                    subsample=0.8, random_state=42
                ),
                'AdaBoost': AdaBoostClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=42
                ),
                'DecisionTree': DecisionTreeClassifier(
                    max_depth=15, random_state=42
                ),
                
                # 线性模型
                'LogisticRegression': LogisticRegression(
                    C=1.0, random_state=42, max_iter=1000, n_jobs=-1
                ),
                
                # 其他模型
                'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
                'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
                'MLP': MLPClassifier(
                    hidden_layer_sizes=(100, 50), activation='relu',
                    learning_rate_init=0.001, max_iter=500, random_state=42
                )
            }
        
        # 训练每个模型
        for name, model in model_configs.items():
            try:
                print(f"   训练 {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                models[name] = model
                predictions[name] = y_pred
                
                # 计算分数
                if target_type == 'regression':
                    score = mean_absolute_error(y_test, y_pred)
                    scores[name] = score
                    print(f"     {name} MAE: {score:.4f}")
                else:
                    score = accuracy_score(y_test, y_pred)
                    scores[name] = score
                    print(f"     {name} 准确率: {score:.4f}")
                    
            except Exception as e:
                print(f"     {name} 训练失败: {e}")
        
        return models, predictions, scores
    
    def train_advanced_neural_network(self, X_train, X_test, y_train, y_test, target_type='regression'):
        """训练改进的深度学习模型"""
        print("🧠 训练改进的深度学习模型...")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        
        if target_type == 'regression':
            y_train_tensor = torch.FloatTensor(y_train)
            y_test_tensor = torch.FloatTensor(y_test)
        else:
            y_train_tensor = torch.LongTensor(y_train)
            y_test_tensor = torch.LongTensor(y_test)
        
        # 创建数据集
        class AdvancedMovieDataset(Dataset):
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets
            
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                return self.features[idx], self.targets[idx]
        
        train_dataset = AdvancedMovieDataset(X_train_tensor, y_train_tensor)
        test_dataset = AdvancedMovieDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # 定义多种改进的神经网络架构
        class ResidualBlock(nn.Module):
            """残差块"""
            def __init__(self, input_size, output_size):
                super().__init__()
                self.linear1 = nn.Linear(input_size, output_size)
                self.bn1 = nn.BatchNorm1d(output_size)
                self.linear2 = nn.Linear(output_size, output_size)
                self.bn2 = nn.BatchNorm1d(output_size)
                self.dropout = nn.Dropout(0.3)
                self.shortcut = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
            
            def forward(self, x):
                residual = self.shortcut(x)
                out = self.linear1(x)
                out = self.bn1(out)
                out = torch.relu(out)
                out = self.dropout(out)
                out = self.linear2(out)
                out = self.bn2(out)
                out += residual
                out = torch.relu(out)
                return out
        
        class AdvancedNetV1(nn.Module):
            """深度残差网络"""
            def __init__(self, input_size, output_type='regression'):
                super().__init__()
                self.output_type = output_type
                
                self.input_layer = nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.4)
                )
                
                self.res_blocks = nn.Sequential(
                    ResidualBlock(256, 256),
                    ResidualBlock(256, 128),
                    ResidualBlock(128, 64),
                )
                
                if output_type == 'regression':
                    self.output_layer = nn.Linear(64, 1)
                elif output_type == 'binary':
                    self.output_layer = nn.Sequential(
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                    )
                else:  # multiclass
                    self.output_layer = nn.Sequential(
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 3),
                        nn.Softmax(dim=1)
                    )
            
            def forward(self, x):
                x = self.input_layer(x)
                x = self.res_blocks(x)
                return self.output_layer(x).squeeze()
        
        class AdvancedNetV2(nn.Module):
            """宽网络架构"""
            def __init__(self, input_size, output_type='regression'):
                super().__init__()
                self.output_type = output_type
                
                self.network = nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                )
                
                if output_type == 'regression':
                    self.output_layer = nn.Linear(64, 1)
                elif output_type == 'binary':
                    self.output_layer = nn.Sequential(
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                    )
                else:
                    self.output_layer = nn.Sequential(
                        nn.Linear(64, 3),
                        nn.LogSoftmax(dim=1)
                    )
            
            def forward(self, x):
                x = self.network(x)
                return self.output_layer(x).squeeze()
        
        # 训练多个深度学习模型
        input_size = X_train.shape[1]
        dl_models = {
            'DeepResNet': AdvancedNetV1(input_size, target_type),
            'DeepWideNet': AdvancedNetV2(input_size, target_type)
        }
        
        dl_results = {}
        
        for model_name, model in dl_models.items():
            print(f"   训练 {model_name}...")
            
            # 训练配置
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            if target_type == 'regression':
                criterion = nn.MSELoss()
            elif target_type == 'binary':
                criterion = nn.BCELoss()
            else:
                criterion = nn.NLLLoss()
            
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            
            # 训练循环
            model.train()
            best_val_loss = float('inf')
            patience = 10
            counter = 0
            
            for epoch in range(100):
                # 训练
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    if target_type == 'multiclass':
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                scheduler.step()
                
                # 验证
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        
                        if target_type == 'multiclass':
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    counter += 1
                
                if counter >= patience:
                    break
            
            # 加载最佳模型
            model.load_state_dict(best_model_state)
            
            # 最终预测
            model.eval()
            with torch.no_grad():
                test_predictions = []
                for batch_X, _ in test_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    
                    if target_type == 'multiclass':
                        _, predicted = torch.max(outputs, 1)
                        test_predictions.extend(predicted.cpu().numpy())
                    elif target_type == 'binary':
                        predicted = (outputs > 0.5).int()
                        test_predictions.extend(predicted.cpu().numpy())
                    else:
                        test_predictions.extend(outputs.cpu().numpy())
            
            # 计算分数
            if target_type == 'regression':
                score = mean_absolute_error(y_test, test_predictions)
                print(f"     {model_name} MAE: {score:.4f}")
            else:
                score = accuracy_score(y_test, test_predictions)
                print(f"     {model_name} 准确率: {score:.4f}")
            
            dl_results[model_name] = {
                'model': model,
                'predictions': test_predictions,
                'score': score
            }
        
        # 选择最佳深度学习模型
        best_dl_model = min(dl_results.items(), key=lambda x: x[1]['score'])[0] if target_type == 'regression' else \
                       max(dl_results.items(), key=lambda x: x[1]['score'])[0]
        
        print(f"   最佳深度学习模型: {best_dl_model}")
        
        return dl_results[best_dl_model]['model'], dl_results[best_dl_model]['predictions'], \
               dl_results[best_dl_model]['score'], dl_results
    
    def run_optimized_experiment(self, features_df, feature_cols, target_type='regression'):
        """运行优化版实验"""
        print(f"\n🎯 开始优化版{target_type.upper()}任务实验")
        print("=" * 60)
        
        # 准备数据
        X_train, X_test, y_train, y_test = self.prepare_data(
            features_df, feature_cols, target_type
        )
        
        # 训练更多sklearn模型
        sklearn_models, sklearn_predictions, sklearn_scores = self.train_advanced_sklearn_models(
            X_train, X_test, y_train, y_test, target_type
        )
        
        # 训练改进的深度学习模型
        dl_model, dl_predictions, dl_score, all_dl_results = self.train_advanced_neural_network(
            X_train, X_test, y_train, y_test, target_type
        )
        
        # 合并结果
        all_scores = sklearn_scores.copy()
        all_scores['DeepLearning'] = dl_score
        
        all_predictions = sklearn_predictions.copy()
        all_predictions['DeepLearning'] = dl_predictions
        
        # 显示结果排名
        self.display_ranking(all_scores, target_type)
        
        return {
            'scores': all_scores,
            'predictions': all_predictions,
            'sklearn_models': sklearn_models,
            'dl_model': dl_model,
            'all_dl_results': all_dl_results,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def display_ranking(self, scores, target_type):
        """显示模型排名"""
        print(f"\n🏆 {target_type.upper()}任务模型排名:")
        print("-" * 40)
        
        if target_type == 'regression':
            # MAE越低越好
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            for i, (model, score) in enumerate(sorted_scores, 1):
                print(f"   {i}. {model}: MAE = {score:.4f}")
        else:
            # 准确率越高越好
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for i, (model, score) in enumerate(sorted_scores, 1):
                print(f"   {i}. {model}: 准确率 = {score:.4f}")

def run_optimized_multiple_experiments(features_df, feature_cols, num_experiments=3):
    """运行多次优化版实验"""
    print("🔬 开始优化版多次实验验证")
    print("=" * 60)
    
    all_regression_results = []
    all_classification_results = []
    
    for exp in range(num_experiments):
        print(f"\n🏆 实验 {exp+1}/{num_experiments}")
        print("-" * 40)
        
        # 设置不同的随机种子
        np.random.seed(42 + exp)
        torch.manual_seed(42 + exp)
        
        trainer = OptimizedModelTrainer()
        
        # 回归任务
        reg_results = trainer.run_optimized_experiment(
            features_df, feature_cols, 'regression'
        )
        all_regression_results.append(reg_results)
        
        # 分类任务
        cls_results = trainer.run_optimized_experiment(
            features_df, feature_cols, 'multiclass'
        )
        all_classification_results.append(cls_results)
    
    return all_regression_results, all_classification_results