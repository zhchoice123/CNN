"""K 近邻分类 + 5 折交叉验证选择 K。K 太小过拟合，K 太大欠拟合。"""
import numpy as np

rng = np.random.default_rng(0)
n = 300
X = rng.normal(size=(n, 2))
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.2).astype(int)        # 圆形决策边界
flip = rng.random(n) < 0.1; y[flip] = 1 - y[flip]             # 10% 标签噪声


def knn_predict(Xtr, ytr, Xq, k):
    d = ((Xq[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :k]
    return (ytr[idx].mean(1) > 0.5).astype(int)


folds = np.array_split(rng.permutation(n), 5)
print("  K   训练准确率  5折CV准确率")
best = None
for k in (1, 3, 7, 15, 31, 101):
    cv = []
    for i in range(5):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
        cv.append((knn_predict(X[tr], y[tr], X[te], k) == y[te]).mean())
    train_acc = (knn_predict(X, y, X, k) == y).mean()
    print(f"{k:4d}   {train_acc:8.3f}   {np.mean(cv):8.3f}")
    if best is None or np.mean(cv) > best[1]:
        best = (k, np.mean(cv))
print(f"交叉验证选出的 K = {best[0]} (CV 准确率 {best[1]:.3f})")
