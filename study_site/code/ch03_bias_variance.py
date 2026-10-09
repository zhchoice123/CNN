"""偏差-方差权衡：用不同阶数的多项式拟合带噪声的正弦曲线（Bishop 第 1 章的经典例子）。"""
import numpy as np

rng = np.random.default_rng(3)
f = lambda x: np.sin(2 * np.pi * x)
x_tr = rng.uniform(0, 1, 10); y_tr = f(x_tr) + rng.normal(0, 0.25, 10)
x_te = rng.uniform(0, 1, 500); y_te = f(x_te) + rng.normal(0, 0.25, 500)

print("阶数  训练RMSE  测试RMSE")
for deg in (0, 1, 3, 6, 9):
    coef = np.polyfit(x_tr, y_tr, deg)
    rmse = lambda x, y: np.sqrt(np.mean((np.polyval(coef, x) - y) ** 2))
    print(f"{deg:3d}   {rmse(x_tr, y_tr):8.3f}  {rmse(x_te, y_te):8.3f}")
print("低阶: 训练、测试误差都高 (欠拟合/高偏差)；9 阶: 训练误差≈0 但测试误差暴涨 (过拟合/高方差)")

# 增加数据量可以缓解过拟合
x_big = rng.uniform(0, 1, 200); y_big = f(x_big) + rng.normal(0, 0.25, 200)
coef = np.polyfit(x_big, y_big, 9)
print(f"9 阶模型改用 200 个训练点后的测试 RMSE: {np.sqrt(np.mean((np.polyval(coef, x_te) - y_te) ** 2)):.3f}")
