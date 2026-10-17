"""集成学习：单棵深树 vs 装袋(bagging) vs 梯度提升(boosting)，回归任务的测试误差对比。"""
import numpy as np

rng = np.random.default_rng(0)
f = lambda x: np.sin(3 * x) + 0.5 * x
Xtr = rng.uniform(-2, 2, 150); ytr = f(Xtr) + rng.normal(0, 0.3, 150)
Xte = np.linspace(-2, 2, 400); yte = f(Xte)


def fit_tree(x, y, depth):
    """一维回归树：递归地选择使平方误差最小的切分点。"""
    if depth == 0 or len(y) < 4:
        return float(y.mean())
    xs = np.sort(np.unique(x)); best = None
    for th in (xs[:-1] + xs[1:]) / 2:
        l, r = y[x <= th], y[x > th]
        sse = ((l - l.mean()) ** 2).sum() + ((r - r.mean()) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, th)
    if best is None: return float(y.mean())
    th = best[1]
    return (th, fit_tree(x[x <= th], y[x <= th], depth - 1), fit_tree(x[x > th], y[x > th], depth - 1))


def predict(t, x):
    if not isinstance(t, tuple): return np.full_like(x, t, dtype=float)
    th, l, r = t
    return np.where(x <= th, predict(l, x), predict(r, x))


mse = lambda p: np.mean((p - yte) ** 2)
deep = fit_tree(Xtr, ytr, 8)
print(f"单棵深树 (深度8)        测试 MSE {mse(predict(deep, Xte)):.4f}")

preds = []
for b in range(30):                                     # 装袋：自助采样 + 平均
    idx = rng.integers(0, 150, 150)
    preds.append(predict(fit_tree(Xtr[idx], ytr[idx], 8), Xte))
print(f"装袋 30 棵深树          测试 MSE {mse(np.mean(preds, 0)):.4f}  <- 方差降低")

F_tr = np.full(150, ytr.mean()); F_te = np.full(400, ytr.mean())
for m in range(100):                                    # 梯度提升：每棵浅树拟合残差
    t = fit_tree(Xtr, ytr - F_tr, 2)
    F_tr += 0.1 * predict(t, Xtr); F_te += 0.1 * predict(t, Xte)
print(f"梯度提升 100 棵深度2树  测试 MSE {mse(F_te):.4f}  <- 偏差逐步降低")
