"""粒子滤波：同样的门走廊，但状态连续。粒子能表示多峰的信念，并在新观测后坍缩到真实位置。"""
import numpy as np

rng = np.random.default_rng(1)
L = 20.0
doors = np.array([3.0, 8.0, 14.0])


def door_likelihood(x, see_door):
    near = np.min(np.abs(((x[:, None] - doors[None]) + L / 2) % L - L / 2), 1) < 0.5
    p = np.where(near, 0.8, 0.1)
    return p if see_door else 1 - p


M = 2000
particles = rng.uniform(0, L, M)                # 均匀撒点：完全不知道自己在哪
true_x = 3.2
for step, action in enumerate(["sense", "move", "sense", "move", "sense"]):
    if action == "move":
        true_x = (true_x + 5.0) % L
        particles = (particles + 5.0 + rng.normal(0, 0.2, M)) % L      # 按运动模型移动每个粒子
        continue
    see = np.min(np.abs(true_x - doors)) < 0.5
    w = door_likelihood(particles, see); w /= w.sum()                   # 按观测似然加权
    ess = 1 / np.sum(w ** 2)
    particles = particles[rng.choice(M, M, p=w)]                        # 重采样
    hist, edges = np.histogram(particles, bins=20, range=(0, L))
    peaks = edges[np.argsort(hist)[::-1][:3]].tolist()
    print(f"观测到{'门' if see else '墙'}: 有效粒子数 {ess:6.0f}, 粒子最密集的区间起点 {peaks}, 真实位置 {true_x:.1f}")
print(f"最终估计位置 {np.median(particles):.2f}, 真实位置 {true_x:.1f}")
