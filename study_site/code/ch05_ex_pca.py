"""例题：二维数据协方差矩阵的主成分。"""
import numpy as np
S = np.array([[4.0, 2.0], [2.0, 3.0]])
w, V = np.linalg.eigh(S)
order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
print("特征值(方差):", np.round(w, 4))
print("第一主成分方向:", np.round(V[:, 0], 4))
print("第一主成分解释方差比例:", round(w[0] / w.sum(), 4))
