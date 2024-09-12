import numpy as np
import pandas as pd

# 定义已知参数
theta = 120  # 换能器的开角，单位为度
alpha = 1.5  # 海底坡度，单位为度
D_center = 70  # 海域中心点的水深，单位为米
distances = np.array([-800, -600, -400, -200, 0, 200, 400, 600, 800])  # 测线距离，单位为米
d = 200  # 相邻测线的间距，单位为米

# 计算每个位置的水深 D(x)
tan_alpha = np.tan(np.radians(alpha))
D = D_center - tan_alpha * distances

# 计算覆盖宽度 W(x)
W = 2 * D * np.sqrt(3) * np.cos(np.radians(alpha))

# 计算重叠率 η
eta = 1 - d / W

# 将计算结果存储在表格中
result = pd.DataFrame({
    "测线距中心点的距离/m": distances,
    "海水深度/m": np.round(D, 2),
    "覆盖宽度/m": np.round(W, 2),
    "重叠率/%": np.where(eta >= 0, np.round(eta * 100, 2), '漏测')
})

# 将结果保存为 Excel 文件
# result.to_excel("/mnt/data/result1_detailed.xlsx", index=False)

# 显示结果
print(result)