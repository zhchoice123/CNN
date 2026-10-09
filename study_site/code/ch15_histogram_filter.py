"""直方图（离散贝叶斯）滤波：机器人在有 3 扇门的一维走廊中全局定位（《概率机器人》开篇的经典例子）。"""
import numpy as np

n = 20
doors = np.zeros(n, bool); doors[[3, 8, 14]] = True
p_hit, p_false = 0.8, 0.1                       # 门前看到门的概率 / 非门处误报门的概率


def sense(bel, see_door):
    like = np.where(doors, p_hit, p_false) if see_door else np.where(doors, 1 - p_hit, 1 - p_false)
    bel = like * bel                            # 修正步：乘以似然
    return bel / bel.sum()                      # 归一化


def move(bel, k):                               # 预测步：向右移动 k 格，带运动噪声
    return 0.8 * np.roll(bel, k) + 0.1 * np.roll(bel, k - 1) + 0.1 * np.roll(bel, k + 1)


bel = np.ones(n) / n                            # 一开始不知道自己在哪：均匀分布
true_pos = 3
for step, see in enumerate([True, None, True, None, False]):
    if see is None:
        bel = move(bel, 5); true_pos = (true_pos + 5) % n; what = "向右走 5 格"
    else:
        bel = sense(bel, doors[true_pos]); what = f"观测到{'门' if doors[true_pos] else '墙'}"
    top = np.argsort(bel)[::-1][:3]
    print(f"{what:8s} -> 最可能的位置 {top.tolist()} 概率 {np.round(bel[top], 3).tolist()}  (真实位置 {true_pos})")
