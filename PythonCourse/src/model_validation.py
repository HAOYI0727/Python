"""
模型训练与验证模块
严格按照要求进行三次随机划分、训练和验证
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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ThreeFoldValidator:
    """三次划分验证器"""
    
    def __init__(self):
        self.results = {}
        
    def prepare_data(self, features_df, feature_cols, target_type='regression'):
        """准备数据"""
        X = features_df[feature_cols].values
        
        if target_type == 'regression':
            y = features_df['target_regression'].values
        elif target_type == 'multiclass':
            y = features_df['target_classification'].values
        else:  # binary
            y = features_df['target_binary'].values
            
        return X, y
    
    def create_train_val_split(self, X, y, split_ratio=0.8, random_state=None):
        """创建训练集和验证集划分"""
        # 设置随机种子
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
        
        # 随机打乱数据
        indices = np.random.permutation(len(X))
        split_point = int(len(X) * split_ratio)
        
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler
    
    def train_sklearn_model(self, model, X_train, y_train):
        """训练sklearn模型"""
        model.fit(X_train, y_train)
        return model
    
    def evaluate_sklearn_model(self, model, X_val, y_val, target_type):
        """评估sklearn模型"""
        y_pred = model.predict(X_val)
        
        if target_type == 'regression':
            score = mean_absolute_error(y_val, y_pred)
        else:
            score = accuracy_score(y_val, y_pred)
            
        return score, y_pred
    
    def train_pytorch_model(self, model, X_train, y_train, target_type, epochs=100):
        """训练PyTorch模型"""
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train)
        if target_type == 'regression':
            y_train_tensor = torch.FloatTensor(y_train)
        else:
            y_train_tensor = torch.LongTensor(y_train)
        
        # 创建数据集和数据加载器
        class SimpleDataset(Dataset):
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets
            
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                return self.features[idx], self.targets[idx]
        
        dataset = SimpleDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 训练配置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        if target_type == 'regression':
            criterion = nn.MSELoss()
        elif target_type == 'binary':
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练循环
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
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
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"      轮次 {epoch+1}, 损失: {epoch_loss/len(train_loader):.4f}")
        
        return model
    
    def evaluate_pytorch_model(self, model, X_val, y_val, target_type):
        """评估PyTorch模型"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            
            if target_type == 'multiclass':
                _, predictions = torch.max(outputs, 1)
                predictions = predictions.cpu().numpy()
            elif target_type == 'binary':
                predictions = (outputs > 0.5).int().cpu().numpy()
            else:
                predictions = outputs.cpu().numpy()
        
        if target_type == 'regression':
            score = mean_absolute_error(y_val, predictions)
        else:
            score = accuracy_score(y_val, predictions)
            
        return score, predictions
    
    def run_single_experiment(self, features_df, feature_cols, target_type='regression', 
                            experiment_num=1, models_to_run=None):
        """运行单次实验（一次划分）"""
        print(f"\n🔬 实验 {experiment_num} - {target_type.upper()}任务")
        print("-" * 40)
        
        # 准备数据
        X, y = self.prepare_data(features_df, feature_cols, target_type)
        
        # 随机划分训练集和验证集
        X_train, X_val, y_train, y_val, scaler = self.create_train_val_split(
            X, y, split_ratio=0.8, random_state=42 + experiment_num
        )
        
        print(f"   数据划分: 训练集 {len(X_train)} 样本, 验证集 {len(X_val)} 样本")
        
        # 定义要训练的模型
        if models_to_run is None:
            models_to_run = ['RandomForest', 'GradientBoosting', 'DeepLearning']
        
        experiment_results = {}
        
        for model_name in models_to_run:
            print(f"   训练 {model_name}...")
            
            if model_name == 'RandomForest':
                if target_type == 'regression':
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                
                trained_model = self.train_sklearn_model(model, X_train, y_train)
                score, _ = self.evaluate_sklearn_model(trained_model, X_val, y_val, target_type)
                
            elif model_name == 'GradientBoosting':
                if target_type == 'regression':
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                else:
                    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                
                trained_model = self.train_sklearn_model(model, X_train, y_train)
                score, _ = self.evaluate_sklearn_model(trained_model, X_val, y_val, target_type)
                
            elif model_name == 'SVM':
                if target_type == 'regression':
                    model = SVR(kernel='rbf', C=1.0)
                else:
                    model = SVC(kernel='rbf', C=1.0, random_state=42)
                
                trained_model = self.train_sklearn_model(model, X_train, y_train)
                score, _ = self.evaluate_sklearn_model(trained_model, X_val, y_val, target_type)
                
            elif model_name == 'DeepLearning':
                # 定义深度学习模型
                class DeepNet(nn.Module):
                    def __init__(self, input_size, output_type='regression'):
                        super().__init__()
                        self.output_type = output_type
                        
                        self.network = nn.Sequential(
                            nn.Linear(input_size, 128),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(64, 32),
                            nn.ReLU(),
                        )
                        
                        if output_type == 'regression':
                            self.output_layer = nn.Linear(32, 1)
                        elif output_type == 'binary':
                            self.output_layer = nn.Sequential(
                                nn.Linear(32, 1),
                                nn.Sigmoid()
                            )
                        else:
                            self.output_layer = nn.Linear(32, 3)
                    
                    def forward(self, x):
                        x = self.network(x)
                        return self.output_layer(x).squeeze()
                
                input_size = X_train.shape[1]
                model = DeepNet(input_size, target_type)
                trained_model = self.train_pytorch_model(model, X_train, y_train, target_type, epochs=50)
                score, _ = self.evaluate_pytorch_model(trained_model, X_val, y_val, target_type)
            
            # 记录结果
            experiment_results[model_name] = score
            
            metric_name = "MAE" if target_type == 'regression' else "准确率"
            print(f"     {model_name}: {metric_name} = {score:.4f}")
        
        return experiment_results
    
    def run_three_experiments(self, features_df, feature_cols, target_type='regression'):
        """运行三次实验并计算平均准确率"""
        print(f"\n🎯 开始三次实验验证 - {target_type.upper()}任务")
        print("=" * 60)
        
        all_results = {}
        
        for exp_num in range(1, 4):
            # 运行单次实验
            results = self.run_single_experiment(
                features_df, feature_cols, target_type, exp_num
            )
            all_results[f'experiment_{exp_num}'] = results
        
        # 计算平均准确率
        average_scores = self.calculate_average_scores(all_results)
        
        # 显示最终结果
        self.display_final_results(all_results, average_scores, target_type)
        
        return all_results, average_scores
    
    def calculate_average_scores(self, all_results):
        """计算平均分数"""
        models = list(all_results['experiment_1'].keys())
        average_scores = {}
        
        for model in models:
            scores = []
            for exp_name in all_results.keys():
                scores.append(all_results[exp_name][model])
            average_scores[model] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        return average_scores
    
    def display_final_results(self, all_results, average_scores, target_type):
        """显示最终结果"""
        print(f"\n📊 {target_type.upper()}任务 - 三次实验最终结果")
        print("=" * 50)
        
        metric_name = "MAE" if target_type == 'regression' else "准确率"
        
        # 显示每次实验的结果
        print("各次实验结果:")
        for exp_name, results in all_results.items():
            print(f"  {exp_name}: ", end="")
            for model, score in results.items():
                print(f"{model}: {score:.4f}  ", end="")
            print()
        
        print(f"\n平均{metric_name}:")
        for model, stats in average_scores.items():
            print(f"  {model}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # 找出最佳模型
        if target_type == 'regression':
            best_model = min(average_scores.items(), key=lambda x: x[1]['mean'])[0]
            best_score = average_scores[best_model]['mean']
            print(f"\n🏆 最佳模型: {best_model} (平均MAE: {best_score:.4f})")
        else:
            best_model = max(average_scores.items(), key=lambda x: x[1]['mean'])[0]
            best_score = average_scores[best_model]['mean']
            print(f"\n🏆 最佳模型: {best_model} (平均准确率: {best_score:.4f})")

def run_complete_validation(features_df, feature_cols):
    """运行完整的验证流程"""
    print("🚀 开始完整的模型训练与验证流程")
    print("=" * 60)
    
    validator = ThreeFoldValidator()
    
    # 回归任务三次实验
    print("\n" + "="*60)
    reg_results, reg_averages = validator.run_three_experiments(
        features_df, feature_cols, 'regression'
    )
    
    # 分类任务三次实验
    print("\n" + "="*60)
    cls_results, cls_averages = validator.run_three_experiments(
        features_df, feature_cols, 'multiclass'
    )
    
    return {
        'regression': {'results': reg_results, 'averages': reg_averages},
        'classification': {'results': cls_results, 'averages': cls_averages}
    }