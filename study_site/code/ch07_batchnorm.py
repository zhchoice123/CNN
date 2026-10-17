"""批量归一化的前向计算：对小批量逐特征标准化，再用可学习的 γ、β 缩放平移。"""
import numpy as np

rng = np.random.default_rng(0)
z = rng.normal(loc=[5.0, -3.0, 0.0], scale=[10.0, 0.1, 1.0], size=(32, 3))   # 三个特征尺度差异巨大
mu, var = z.mean(0), z.var(0)
z_hat = (z - mu) / np.sqrt(var + 1e-5)
gamma, beta = np.array([1.0, 2.0, 0.5]), np.array([0.0, 1.0, -1.0])
out = gamma * z_hat + beta
np.set_printoptions(precision=3, suppress=True)
print("归一化前 均值:", z.mean(0), " 标准差:", z.std(0))
print("标准化后 均值:", z_hat.mean(0), " 标准差:", z_hat.std(0))
print("γ,β 变换后 均值:", out.mean(0), " 标准差:", out.std(0), " (= β 与 γ)")
# 推理时用训练期间累积的滑动平均，而不是当前批的统计量
running_mu = np.zeros(3)
for _ in range(200):
    batch = rng.normal(loc=[5.0, -3.0, 0.0], scale=[10.0, 0.1, 1.0], size=(32, 3))
    running_mu = 0.9 * running_mu + 0.1 * batch.mean(0)
print("滑动平均估计的总体均值:", running_mu)
