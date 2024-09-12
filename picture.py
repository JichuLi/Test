import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 读取Excel文件并准备数据
df = pd.read_excel(r"C:\Users\26777\Desktop\Code\第二问自变量表 - 副本.xlsx")

# 假设前五列是特征，最后一列是标签
X = df.iloc[:, :-1].values  # 特征
y = df.iloc[:, -1].values   # 标签

# 进行标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 2. 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
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

model = SimpleNN()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 假设是回归问题
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 测试模型
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 测试模型并计算评估指标
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    predictions = model(X_test)
    
    # 转换为NumPy数组以便使用sklearn计算指标
    predictions = predictions.numpy()
    y_test_np = y_test.numpy()
    
    # 计算MSE, RMSE, MAE, R²
    mse = mean_squared_error(y_test_np, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test_np, predictions)
    r2 = r2_score(y_test_np, predictions)
    
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')
