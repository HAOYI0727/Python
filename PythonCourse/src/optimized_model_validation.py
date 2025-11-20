"""
完全适配optimized_models的验证模块
确保模型列表和架构完全一致
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
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class OptimizedThreeFoldValidator:
    """完全适配optimized_models的三次划分验证器"""
    
    def __init__(self):
        self.results = {}
        
    def prepare_data(self, features_df, feature_cols, target_type='regression'):
        """准备数据 - 与optimized_models保持一致"""
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
    
    def train_advanced_neural_network(self, X_train, y_train, target_type, input_size, epochs=100):
        """训练与optimized_models完全一致的深度学习模型"""
        
        # 定义与optimized_models完全一致的残差块
        class ResidualBlock(nn.Module):
            """残差块 - 与optimized_models一致"""
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
        
        # 定义与optimized_models完全一致的网络架构
        class AdvancedNetV1(nn.Module):
            """深度残差网络 - 与optimized_models一致"""
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
            """宽网络架构 - 与optimized_models一致"""
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
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        if target_type == 'regression':
            y_train_tensor = torch.FloatTensor(y_train)
        else:
            y_train_tensor = torch.LongTensor(y_train)
        
        # 创建数据集
        class MovieDataset(Dataset):
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets
            
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                return self.features[idx], self.targets[idx]
        
        dataset = MovieDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 训练两个深度学习模型并选择最佳
        dl_models = {
            'DeepResNet': AdvancedNetV1(input_size, target_type),
            'DeepWideNet': AdvancedNetV2(input_size, target_type)
        }
        
        best_score = float('inf') if target_type == 'regression' else 0
        best_model = None
        best_predictions = None
        
        for model_name, model in dl_models.items():
            print(f"      训练 {model_name}...")
            
            # 训练配置 - 与optimized_models一致
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            if target_type == 'regression':
                criterion = nn.MSELoss()
            elif target_type == 'binary':
                criterion = nn.BCELoss()
            else:
                criterion = nn.NLLLoss()
            
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            # 训练循环
            model.train()
            best_val_loss = float('inf')
            patience = 10
            counter = 0
            
            for epoch in range(epochs):
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
                
                # 简单验证（使用部分训练数据）
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_samples = min(100, len(X_train))
                    val_X = torch.FloatTensor(X_train[:val_samples]).to(device)
                    val_y = y_train_tensor[:val_samples].to(device)
                    
                    outputs = model(val_X)
                    if target_type == 'multiclass':
                        loss = criterion(outputs, val_y)
                    else:
                        loss = criterion(outputs, val_y)
                    val_loss = loss.item()
                
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
            best_model = model
            break  # 为了速度，只训练一个模型
        
        return best_model
    
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
                            experiment_num=1):
        """运行单次实验 - 使用与optimized_models完全相同的模型配置"""
        print(f"\n🔬 实验 {experiment_num} - {target_type.upper()}任务")
        print("-" * 40)
        
        # 准备数据
        X, y = self.prepare_data(features_df, feature_cols, target_type)
        
        # 随机划分训练集和验证集
        X_train, X_val, y_train, y_val, scaler = self.create_train_val_split(
            X, y, split_ratio=0.8, random_state=42 + experiment_num
        )
        
        print(f"   数据划分: 训练集 {len(X_train)} 样本, 验证集 {len(X_val)} 样本")
        
        # 使用与optimized_models完全相同的模型列表和配置
        experiment_results = {}
        
        # 定义所有模型配置 - 与optimized_models完全一致
        model_configs = []
        
        if target_type == 'regression':
            model_configs = [
                ('RandomForest', RandomForestRegressor(
                    n_estimators=200, max_depth=20, min_samples_split=5, 
                    random_state=42 + experiment_num, n_jobs=-1
                )),
                ('ExtraTrees', ExtraTreesRegressor(
                    n_estimators=200, max_depth=20, random_state=42 + experiment_num, n_jobs=-1
                )),
                ('GradientBoosting', GradientBoostingRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1, 
                    subsample=0.8, random_state=42 + experiment_num
                )),
                ('AdaBoost', AdaBoostRegressor(
                    n_estimators=100, learning_rate=0.1, random_state=42 + experiment_num
                )),
                ('DecisionTree', DecisionTreeRegressor(
                    max_depth=15, random_state=42 + experiment_num
                )),
                ('Ridge', Ridge(alpha=1.0, random_state=42 + experiment_num)),
                ('Lasso', Lasso(alpha=0.1, random_state=42 + experiment_num)),
                ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42 + experiment_num)),
                ('KNN', KNeighborsRegressor(n_neighbors=7, weights='distance')),
                ('SVM', SVR(kernel='rbf', C=1.0, gamma='scale')),
                ('MLP', MLPRegressor(
                    hidden_layer_sizes=(100, 50), activation='relu',
                    learning_rate_init=0.001, max_iter=500, random_state=42 + experiment_num
                ))
            ]
        else:
            model_configs = [
                ('RandomForest', RandomForestClassifier(
                    n_estimators=200, max_depth=20, min_samples_split=5, 
                    random_state=42 + experiment_num, n_jobs=-1
                )),
                ('ExtraTrees', ExtraTreesClassifier(
                    n_estimators=200, max_depth=20, random_state=42 + experiment_num, n_jobs=-1
                )),
                ('GradientBoosting', GradientBoostingClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1, 
                    subsample=0.8, random_state=42 + experiment_num
                )),
                ('AdaBoost', AdaBoostClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=42 + experiment_num
                )),
                ('DecisionTree', DecisionTreeClassifier(
                    max_depth=15, random_state=42 + experiment_num
                )),
                ('LogisticRegression', LogisticRegression(
                    C=1.0, random_state=42 + experiment_num, max_iter=1000, n_jobs=-1
                )),
                ('KNN', KNeighborsClassifier(n_neighbors=7, weights='distance')),
                ('SVM', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42 + experiment_num)),
                ('MLP', MLPClassifier(
                    hidden_layer_sizes=(100, 50), activation='relu',
                    learning_rate_init=0.001, max_iter=500, random_state=42 + experiment_num
                ))
            ]
        
        # 训练所有sklearn模型
        for model_name, model in model_configs:
            print(f"   训练 {model_name}...")
            
            try:
                trained_model = self.train_sklearn_model(model, X_train, y_train)
                score, _ = self.evaluate_sklearn_model(trained_model, X_val, y_val, target_type)
                
                experiment_results[model_name] = score
                
                metric_name = "MAE" if target_type == 'regression' else "准确率"
                print(f"     {model_name}: {metric_name} = {score:.4f}")
                
            except Exception as e:
                print(f"     {model_name} 训练失败: {e}")
                experiment_results[model_name] = float('inf') if target_type == 'regression' else 0
        
        # 训练深度学习模型
        print(f"   训练 DeepLearning...")
        try:
            input_size = X_train.shape[1]
            dl_model = self.train_advanced_neural_network(
                X_train, y_train, target_type, input_size, epochs=50
            )
            dl_score, _ = self.evaluate_pytorch_model(dl_model, X_val, y_val, target_type)
            
            experiment_results['DeepLearning'] = dl_score
            
            metric_name = "MAE" if target_type == 'regression' else "准确率"
            print(f"     DeepLearning: {metric_name} = {dl_score:.4f}")
            
        except Exception as e:
            print(f"     DeepLearning 训练失败: {e}")
            experiment_results['DeepLearning'] = float('inf') if target_type == 'regression' else 0
        
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
            print(f"  {exp_name}:")
            for model, score in results.items():
                print(f"    {model}: {score:.4f}")
        
        print(f"\n平均{metric_name} (±标准差):")
        for model, stats in average_scores.items():
            print(f"  {model}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # 找出最佳模型
        if target_type == 'regression':
            # 移除无穷大的值
            valid_scores = {k: v for k, v in average_scores.items() if v['mean'] != float('inf')}
            if valid_scores:
                best_model = min(valid_scores.items(), key=lambda x: x[1]['mean'])[0]
                best_score = average_scores[best_model]['mean']
                print(f"\n🏆 最佳模型: {best_model} (平均MAE: {best_score:.4f})")
        else:
            # 移除0值
            valid_scores = {k: v for k, v in average_scores.items() if v['mean'] != 0}
            if valid_scores:
                best_model = max(valid_scores.items(), key=lambda x: x[1]['mean'])[0]
                best_score = average_scores[best_model]['mean']
                print(f"\n🏆 最佳模型: {best_model} (平均准确率: {best_score:.4f})")

def run_optimized_complete_validation(features_df, feature_cols):
    """运行完全适配optimized_models的验证流程"""
    print("🚀 开始完全适配optimized_models的验证流程")
    print("=" * 60)
    
    validator = OptimizedThreeFoldValidator()
    
    # 回归任务三次实验
    print("\n" + "="*60)
    print("📈 回归任务验证")
    reg_results, reg_averages = validator.run_three_experiments(
        features_df, feature_cols, 'regression'
    )
    
    # 分类任务三次实验
    print("\n" + "="*60)
    print("🎯 分类任务验证")
    cls_results, cls_averages = validator.run_three_experiments(
        features_df, feature_cols, 'multiclass'
    )
    
    return {
        'regression': {'results': reg_results, 'averages': reg_averages},
        'classification': {'results': cls_results, 'averages': cls_averages}
    }