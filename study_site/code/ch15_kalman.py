"""卡尔曼滤波：用带噪声的位置测量跟踪一个匀速运动的目标，同时估计位置与速度。"""
import numpy as np

rng = np.random.default_rng(0)
dt, T = 1.0, 50
F = np.array([[1, dt], [0, 1]])                 # 状态转移：位置 += 速度*dt
H = np.array([[1.0, 0.0]])                      # 只能测量位置
Q = 0.01 * np.eye(2)                            # 过程噪声
R = np.array([[4.0]])                           # 测量噪声方差（标准差 2）

x_true = np.array([0.0, 1.0]); x = np.array([0.0, 0.0]); P = np.eye(2) * 10
err_meas, err_kf = [], []
for t in range(T):
    x_true = F @ x_true + rng.multivariate_normal([0, 0], Q)
    z = H @ x_true + rng.normal(0, 2.0, 1)
    x = F @ x; P = F @ P @ F.T + Q                       # 预测
    S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
    x = x + K @ (z - H @ x); P = (np.eye(2) - K @ H) @ P # 更新
    err_meas.append(z[0] - x_true[0]); err_kf.append(x[0] - x_true[0])
    if t in (0, 4, 49):
        print(f"t={t:2d}: 卡尔曼增益 K={np.round(K.ravel(), 3)}  速度估计 {x[1]:.3f} (真实 {x_true[1]:.3f})")
rm = lambda e: np.sqrt(np.mean(np.square(e[10:])))
print(f"位置误差 RMSE: 直接用测量 {rm(err_meas):.3f}  卡尔曼滤波 {rm(err_kf):.3f}")
