import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 示例数据
X = np.array([
    [10, 20, 30, 40, 50],
    [12, 22, 32, 42, 52],
    [14, 24, 34, 44, 54],
    [16, 26, 36, 46, 56],
    [18, 28, 38, 48, 58],
    [20, 30, 40, 50, 60],
    [22, 32, 42, 52, 62],
    [24, 34, 44, 54, 64],
    [26, 36, 46, 56, 66],
    [28, 38, 48, 58, 68]
])

Y = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

# 数据标准化
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# 使用PLS回归，提取2个潜在因子
pls = PLSRegression(n_components=2)
pls.fit(X_train, Y_train)

# 预测
Y_pred = pls.predict(X_test)

# 还原预测结果到原始尺度
Y_pred_original = scaler_Y.inverse_transform(Y_pred)

# 打印结果
print("测试集真实值:", scaler_Y.inverse_transform(Y_test))
print("测试集预测值:", Y_pred_original)
