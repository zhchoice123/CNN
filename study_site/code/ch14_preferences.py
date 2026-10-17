"""RLHF 的两块基石：用 Bradley–Terry 模型从成对偏好中学习奖励模型；以及 DPO 损失的计算。"""
import numpy as np

rng = np.random.default_rng(3)
d = 5
w_true = np.array([2.0, -1.0, 0.5, 0.0, 1.0])     # 人类“真实”偏好的隐藏权重（回答的特征 -> 奖励）
A = rng.normal(size=(2000, d)); B = rng.normal(size=(2000, d))
p_prefer_A = 1 / (1 + np.exp(-(A - B) @ w_true))   # Bradley–Terry: P(A ≻ B) = σ(r(A) − r(B))
y = (rng.random(2000) < p_prefer_A).astype(float)
chosen = np.where(y[:, None] == 1, A, B); rejected = np.where(y[:, None] == 1, B, A)

w = np.zeros(d)
for _ in range(500):                               # 最小化 −log σ(r(chosen) − r(rejected))
    m = (chosen - rejected) @ w
    grad = -((1 - 1 / (1 + np.exp(-m)))[:, None] * (chosen - rejected)).mean(0)
    w -= 0.5 * grad
print("真实奖励权重:", w_true)
print("学到的奖励权重:", np.round(w, 2))
acc = (((chosen - rejected) @ w) > 0).mean()
print(f"奖励模型在训练偏好上的一致率 {acc:.3f} (标注本身有噪声，上限低于 1)")

# DPO：直接用策略与参考模型的对数概率比构造损失，无需单独的奖励模型
beta = 0.1
logp_pol_w, logp_ref_w = -12.0, -14.0              # 偏好回答: 策略比参考模型更喜欢它 (+2)
logp_pol_l, logp_ref_l = -15.0, -13.0              # 被拒回答: 策略比参考模型更不喜欢它 (-2)
margin = beta * ((logp_pol_w - logp_ref_w) - (logp_pol_l - logp_ref_l))
print(f"\nDPO 隐式奖励差 = {margin:.2f},  损失 = -log σ(差) = {-np.log(1 / (1 + np.exp(-margin))):.4f}")
print(f"若策略与参考模型完全相同，损失 = log 2 = {np.log(2):.4f}")
