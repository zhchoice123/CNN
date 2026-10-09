"""策略梯度 REINFORCE：∇J = E[∇log π(a) · (R − b)]，基线 b 降低方差但不改变期望。"""
import numpy as np

rng = np.random.default_rng(2)
true_reward = np.array([0.2, 0.5, 0.8])            # 三个动作的平均奖励
theta = np.zeros(3)                                # softmax 策略的参数


def policy(th):
    e = np.exp(th - th.max()); return e / e.sum()


for use_baseline in (False, True):
    th = theta.copy(); b = 0.0; grads = []
    for it in range(1, 2001):
        p = policy(th); a = rng.choice(3, p=p)
        R = rng.normal(true_reward[a], 0.5)
        grad_logp = -p; grad_logp[a] += 1          # ∇_θ log softmax(θ)_a = onehot(a) − p
        adv = R - (b if use_baseline else 0)
        g = grad_logp * adv; grads.append(g)
        th += 0.05 * g
        b += 0.05 * (R - b)                        # 基线 = 奖励的滑动平均
    tag = "有基线" if use_baseline else "无基线"
    print(f"{tag}: 最终策略 {np.round(policy(th), 3)}, 梯度估计的方差 {np.var(grads[:500], 0).mean():.4f}")
print("策略逐渐集中到奖励最高的动作 2；减去基线后梯度方差更小，学习更稳定")
