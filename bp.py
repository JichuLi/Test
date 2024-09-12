import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1. 读取Excel文件并准备数据
df = pd.read_excel(r"C:\Users\26777\Desktop\Code\第二问自变量表 - 副本.xlsx")

# 假设前五列是特征，最后一列是标签
X = df.iloc[:, :-1].values  # 特征
y = df.iloc[:, -1].values   # 标签

# 进行标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 2. 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 3. 设置交叉验证
kf = KFold(n_splits=7, shuffle=True, random_state=42)

mse_scores = []
rmse_scores = []
mae_scores = []
r2_scores = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f'Fold {fold + 1}')
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # 初始化模型、损失函数和优化器
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 测试模型并计算评估指标
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        y_test_np = y_test.numpy()
        
        mse = mean_squared_error(y_test_np, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test_np, predictions)
        r2 = r2_score(y_test_np, predictions)
        
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
    
    print(f'Fold {fold + 1} MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')

# 4. 输出交叉验证的平均评估结果
print(f'\nAverage MSE: {np.mean(mse_scores):.4f}')
print(f'Average RMSE: {np.mean(rmse_scores):.4f}')
print(f'Average MAE: {np.mean(mae_scores):.4f}')
print(f'Average R²: {np.mean(r2_scores):.4f}')



model.eval()
max=0
l=[]
for i in range(250,475,25):
    for j in [0.228903705,1.182039926,-0.254777362,3.932518736,-0.500245504,0.425658254,-0.417608594,-0.582882414,-0.257195768,-0.694685293,-0.621770372,-0.378720636]:
        for k in[0.044575259,-0.315088344,-0.674751947,-1.034415551,-1.250213713,-0.919323198,-0.430180697,-0.702154889,1.373617907,1.455020762,1.497544641,1.483229672]:
            for e in [0.0654735353497044,-3.063031958,-1.522126267,3.288782226]:
                for s in [0.381267926,-1.125502918,-2.284557414,-1.125502918,1.192606073]:
                    with torch.no_grad():
                        predictions = model(torch.tensor([(i-315.5701754)/52.31270329,j,k,e,s]))
                        predictions = float(predictions)
                        if predictions>max:
                            l=[i,j,k,e,s]
                            max=predictions


# 输出预测结果
print(l)
print(max)