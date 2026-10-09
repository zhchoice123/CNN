"""Q 学习：在 4x4 网格世界中学习走到终点。Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]"""
import numpy as np

rng = np.random.default_rng(1)
N = 4; goal = (3, 3); holes = {(1, 1), (2, 3), (3, 0)}
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]; names = "↑↓←→"


def step(s, a):
    r, c = s[0] + moves[a][0], s[1] + moves[a][1]
    if not (0 <= r < N and 0 <= c < N): r, c = s
    if (r, c) == goal: return (r, c), 1.0, True
    if (r, c) in holes: return (r, c), -1.0, True
    return (r, c), -0.01, False                     # 每走一步小惩罚，鼓励走最短路


Q = np.zeros((N, N, 4)); alpha, gamma = 0.5, 0.95
for ep in range(2000):
    s, done = (0, 0), False
    eps = max(0.05, 1 - ep / 1000)                 # 探索率逐渐衰减
    while not done:
        a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))
        s2, r, done = step(s, a)
        target = r + (0 if done else gamma * Q[s2].max())
        Q[s][a] += alpha * (target - Q[s][a])
        s = s2

print("学到的策略 (G=终点, H=陷阱):")
for r in range(N):
    print(" ".join("G" if (r, c) == goal else "H" if (r, c) in holes else names[int(np.argmax(Q[r, c]))] for c in range(N)))
s, path = (0, 0), [(0, 0)]
for _ in range(10):
    s, _, done = step(s, int(np.argmax(Q[s]))); path.append(s)
    if done: break
print("贪心路径:", path)
print(f"起点状态价值 V(0,0) = max_a Q = {Q[0, 0].max():.3f}")
