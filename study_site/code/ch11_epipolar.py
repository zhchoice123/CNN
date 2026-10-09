"""双视图几何：用归一化八点法估计基础矩阵 F，验证对极约束 x2^T F x1 = 0，再三角化恢复三维点。"""
import numpy as np

rng = np.random.default_rng(1)
K = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1.0]])
X = np.column_stack([rng.uniform(-2, 2, 50), rng.uniform(-1, 1, 50), rng.uniform(4, 8, 50)])
a = np.deg2rad(8)
R = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
t = np.array([-1.0, 0.0, 0.1])                                    # 第二台相机向右平移约 1 个单位
P1 = K @ np.column_stack([np.eye(3), np.zeros(3)]); P2 = K @ np.column_stack([R, t])


def proj(P, X):
    x = np.column_stack([X, np.ones(len(X))]) @ P.T
    return x[:, :2] / x[:, 2:]


x1 = proj(P1, X) + rng.normal(0, 0.3, (50, 2)); x2 = proj(P2, X) + rng.normal(0, 0.3, (50, 2))


def norm_T(p):
    c = p.mean(0); s = np.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


T1, T2 = norm_T(x1), norm_T(x2)
h = lambda p: np.column_stack([p, np.ones(len(p))])
n1, n2 = h(x1) @ T1.T, h(x2) @ T2.T
A = np.array([np.kron(b, a_) for a_, b in zip(n1, n2)])           # 每对点一行：x2^T F x1 = 0
F = np.linalg.svd(A)[2][-1].reshape(3, 3)
U, S, Vt = np.linalg.svd(F); S[2] = 0; F = U @ np.diag(S) @ Vt    # 强制秩为 2
F = T2.T @ F @ T1; F /= np.linalg.norm(F)
res = np.abs(np.sum(h(x2) * (h(x1) @ F.T), 1))
print(f"对极约束残差 |x2^T F x1| 平均 {res.mean():.2e}，F 的秩 = {np.linalg.matrix_rank(F, tol=1e-8)}")

E = K.T @ F @ K
s = np.linalg.svd(E, compute_uv=False)
print("本质矩阵的奇异值 (理论上两个相等、一个为 0):", np.round(s / s[0], 4))


def triangulate(p1, p2):
    A = np.array([p1[0] * P1[2] - P1[0], p1[1] * P1[2] - P1[1], p2[0] * P2[2] - P2[0], p2[1] * P2[2] - P2[1]])
    Xh = np.linalg.svd(A)[2][-1]; return Xh[:3] / Xh[3]


Xr = np.array([triangulate(p, q) for p, q in zip(x1, x2)])
print(f"三角化的三维点平均误差 {np.linalg.norm(Xr - X, axis=1).mean():.4f} (场景深度 4~8)")
