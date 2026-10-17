"""贝叶斯定理：罕见病检测。患病率 1%，灵敏度 99%，特异度 95%。检测阳性时真正患病的概率是多少？"""
import numpy as np

prior, sens, spec = 0.01, 0.99, 0.95
p_pos = sens * prior + (1 - spec) * (1 - prior)          # 全概率公式
posterior = sens * prior / p_pos
print(f"P(阳性) = {p_pos:.4f}")
print(f"P(患病 | 阳性) = {posterior:.4f}  <- 远低于直觉上的 99%")

# 用 100 万人模拟验证
rng = np.random.default_rng(42)
n = 1_000_000
sick = rng.random(n) < prior
positive = np.where(sick, rng.random(n) < sens, rng.random(n) < 1 - spec)
print(f"模拟: 阳性者中真正患病的比例 = {sick[positive].mean():.4f}")
# 再测一次（把第一次的后验当作新的先验）
p2 = sens * posterior / (sens * posterior + (1 - spec) * (1 - posterior))
print(f"连续两次阳性后患病概率 = {p2:.4f}")
