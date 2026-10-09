"""决策树如何选择分裂：计算基尼不纯度与信息增益，在所有候选阈值中找最好的划分。"""
import numpy as np


def gini(y):
    if len(y) == 0: return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1 - np.sum(p ** 2)


def entropy(y):
    p = np.bincount(y, minlength=2) / len(y); p = p[p > 0]
    return -np.sum(p * np.log2(p))


# 是否出去打网球：特征 = 温度，标签 1=去 0=不去
temp = np.array([64, 65, 68, 69, 70, 71, 72, 75, 80, 81, 83, 85])
play = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
print(f"根节点: 基尼 {gini(play):.4f}  熵 {entropy(play):.4f}")

best = None
for th in (temp[:-1] + temp[1:]) / 2:
    L, R = play[temp <= th], play[temp > th]
    w_gini = (len(L) * gini(L) + len(R) * gini(R)) / len(play)
    gain = entropy(play) - (len(L) * entropy(L) + len(R) * entropy(R)) / len(play)
    if best is None or w_gini < best[1]:
        best = (th, w_gini, gain)
print(f"最佳阈值 温度 <= {best[0]}: 加权基尼 {best[1]:.4f}, 信息增益 {best[2]:.4f} 比特")
