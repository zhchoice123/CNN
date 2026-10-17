"""例题：折扣回报与一次 Q 学习更新。"""
rewards, gamma = [1, 0, 0, 5], 0.9
G = sum(gamma ** k * r for k, r in enumerate(rewards))
print("折扣回报 G =", " + ".join(f"{gamma}^{k}*{r}" for k, r in enumerate(rewards)), f"= {G:.4f}")
Q, alpha, r, max_next = 2.0, 0.1, 1.0, 3.0
target = r + gamma * max_next
print(f"TD 目标 = {r} + {gamma}*{max_next} = {target}; 新 Q = {Q} + {alpha}*({target} - {Q}) = {Q + alpha * (target - Q):.2f}")
