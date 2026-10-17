"""例题：对 f(w) = (w - 3)^2 做梯度下降，学习率 0.1，起点 w = 0。"""
w, lr = 0.0, 0.1
for t in range(1, 6):
    g = 2 * (w - 3)
    w = w - lr * g
    print(f"第 {t} 步: 梯度 {g:7.3f} -> w = {w:.4f}")
print("误差每步缩小为原来的 (1 - 2*lr) =", 1 - 2 * lr, "倍")
for lr in (0.5, 1.0, 1.1):
    print(f"lr={lr}: 收缩因子 |1-2lr| = {abs(1 - 2*lr):.1f}", "-> 发散" if abs(1 - 2*lr) > 1 else ("-> 来回震荡不收敛" if abs(1 - 2*lr) == 1 else "-> 收敛"))
