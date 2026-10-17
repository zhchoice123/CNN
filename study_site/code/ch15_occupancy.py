"""占据栅格建图：用对数几率累加多次激光测量，l = l + log(p/(1-p)) - l0。"""
import numpy as np

cells = 10
l = np.zeros(cells)                              # 对数几率 0 <=> 占据概率 0.5（未知）
l_occ, l_free = np.log(0.7 / 0.3), np.log(0.3 / 0.7)
for scan in range(1, 4):                         # 激光三次都测到第 6 格有障碍
    hit = 6
    l[:hit] += l_free                            # 光束穿过的格子：更可能空闲
    l[hit] += l_occ                              # 光束末端：更可能被占据
    p = 1 - 1 / (1 + np.exp(l))
    print(f"第 {scan} 次扫描后各格子的占据概率: {np.round(p, 2).tolist()}")
print("障碍物（第 7 格）后面的格子被挡住、从未被观测，保持 0.5（未知）")
