"""
高效机器学习与深度学习模型实现
使用sklearn和PyTorch
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EfficientModelTrainer:
    """高效模型训练器"""
    
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
            X, y, test_size=test_size, random_state=42, stratify=y if target_type != 'regression' else None
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_sklearn_models(self, X_train, X_test, y_train, y_test, target_type='regression'):
        """训练sklearn模型"""
        print("🤖 训练Scikit-learn模型...")
        
        models = {}
        predictions = {}
        scores = {}
        
        # 定义模型配置
        if target_type == 'regression':
            model_configs = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=200, max_depth=15, min_samples_split=5, 
                    random_state=42, n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1, 
                    random_state=42
                ),
                'Ridge': Ridge(alpha=1.0, random_state=42),
                'Lasso': Lasso(alpha=0.1, random_state=42),
                'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
                'SVM': SVR(kernel='rbf', C=1.0, gamma='scale')
            }
        else:
            model_configs = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=5, 
                    random_state=42, n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1, 
                    random_state=42
                ),
                'LogisticRegression': LogisticRegression(
                    C=1.0, random_state=42, max_iter=1000, n_jobs=-1
                ),
                'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
                'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
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
    
    def train_deep_learning_model(self, X_train, X_test, y_train, y_test, target_type='regression'):
        """训练深度学习模型"""
        print("🧠 训练深度学习模型...")
        
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
        class MovieDataset(Dataset):
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets
            
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                return self.features[idx], self.targets[idx]
        
        train_dataset = MovieDataset(X_train_tensor, y_train_tensor)
        test_dataset = MovieDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # 定义改进的神经网络
        class AdvancedNet(nn.Module):
            def __init__(self, input_size, output_type='regression'):
                super().__init__()
                self.output_type = output_type
                
                self.network = nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                )
                
                if output_type == 'regression':
                    self.output_layer = nn.Linear(32, 1)
                elif output_type == 'binary':
                    self.output_layer = nn.Sequential(
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )
                else:  # multiclass
                    self.output_layer = nn.Sequential(
                        nn.Linear(32, 3),
                        nn.LogSoftmax(dim=1)
                    )
            
            def forward(self, x):
                x = self.network(x)
                return self.output_layer(x).squeeze()
        
        # 训练配置
        input_size = X_train.shape[1]
        model = AdvancedNet(input_size, target_type)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        if target_type == 'regression':
            criterion = nn.MSELoss()
        elif target_type == 'binary':
            criterion = nn.BCELoss()
        else:
            criterion = nn.NLLLoss()
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 15
        counter = 0
        
        for epoch in range(100):
            # 训练
            model.train()
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
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            # 早停
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_state = model.state_dict().copy()
            else:
                counter += 1
            
            if counter >= patience:
                print(f"    早停在第 {epoch+1} 轮")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"    轮次 {epoch+1}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        
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
            print(f"    深度学习模型 MAE: {score:.4f}")
        else:
            score = accuracy_score(y_test, test_predictions)
            print(f"    深度学习模型 准确率: {score:.4f}")
        
        return model, test_predictions, score, train_losses, val_losses
    
    def run_complete_experiment(self, features_df, feature_cols, target_type='regression'):
        """运行完整实验"""
        print(f"\n🎯 开始{target_type.upper()}任务实验")
        print("=" * 60)
        
        # 准备数据
        X_train, X_test, y_train, y_test = self.prepare_data(
            features_df, feature_cols, target_type
        )
        
        # 训练sklearn模型
        sklearn_models, sklearn_predictions, sklearn_scores = self.train_sklearn_models(
            X_train, X_test, y_train, y_test, target_type
        )
        
        # 训练深度学习模型
        dl_model, dl_predictions, dl_score, train_losses, val_losses = self.train_deep_learning_model(
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
            'train_losses': train_losses,
            'val_losses': val_losses,
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

def run_multiple_experiments(features_df, feature_cols, num_experiments=3):
    """运行多次实验"""
    print("🔬 开始多次实验验证")
    print("=" * 60)
    
    all_regression_results = []
    all_classification_results = []
    
    for exp in range(num_experiments):
        print(f"\n🏆 实验 {exp+1}/{num_experiments}")
        print("-" * 40)
        
        # 设置不同的随机种子
        np.random.seed(42 + exp)
        torch.manual_seed(42 + exp)
        
        trainer = EfficientModelTrainer()
        
        # 回归任务
        reg_results = trainer.run_complete_experiment(
            features_df, feature_cols, 'regression'
        )
        all_regression_results.append(reg_results)
        
        # 分类任务
        cls_results = trainer.run_complete_experiment(
            features_df, feature_cols, 'multiclass'
        )
        all_classification_results.append(cls_results)
    
    return all_regression_results, all_classification_results