"""探索与利用：10 臂老虎机上比较纯贪心与 ε-贪心策略。"""
import numpy as np

rng = np.random.default_rng(0)
k, steps, runs = 10, 1000, 300
results = {}
for eps in (0.0, 0.01, 0.1):
    total, optimal = 0.0, 0.0
    for _ in range(runs):
        q_true = rng.normal(0, 1, k); best = q_true.argmax()
        Q = np.zeros(k); N = np.zeros(k)
        for t in range(steps):
            a = rng.integers(k) if rng.random() < eps else int(np.argmax(Q))   # ε 概率随机探索
            r = rng.normal(q_true[a], 1)
            N[a] += 1; Q[a] += (r - Q[a]) / N[a]                              # 增量式平均
            total += r; optimal += (a == best)
    results[eps] = (total / runs / steps, optimal / runs / steps)
for eps, (avg_r, opt) in results.items():
    print(f"ε={eps:<5}: 平均奖励 {avg_r:.3f}, 选到最优臂的比例 {opt:.1%}")
print("纯贪心容易过早锁定一个次优选项；适度探索能找到更好的选择")
