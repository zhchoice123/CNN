"""感知机学习规则：能学会 AND、OR，却学不会 XOR（线性不可分）——Minsky & Papert 1969 的经典结论。"""
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tasks = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}

for name, y in tasks.items():
    y = np.array(y); w = np.zeros(2); b = 0.0
    for epoch in range(100):
        errors = 0
        for xi, yi in zip(X, y):
            pred = int(w @ xi + b > 0)               # 阶跃激活
            if pred != yi:                           # 只在出错时更新
                w += (yi - pred) * xi; b += (yi - pred); errors += 1
        if errors == 0: break
    acc = np.mean((X @ w + b > 0).astype(int) == y)
    print(f"{name:3s}: 第 {epoch+1:3d} 轮结束, w={w}, b={b:+.0f}, 准确率 {acc:.2f}", "(收敛)" if errors == 0 else "(无法收敛)")
