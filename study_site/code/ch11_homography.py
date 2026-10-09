"""单应矩阵：归一化的直接线性变换（DLT）+ RANSAC，抵抗错误匹配。"""
import numpy as np

rng = np.random.default_rng(0)
H_true = np.array([[1.2, 0.1, 30], [-0.05, 0.9, 10], [0.0005, 0.0002, 1]])


def apply(H, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:]


def normalize(p):
    c = p.mean(0); s = np.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return apply(T, p), T


def dlt(src, dst):
    (s, Ts), (d, Td) = normalize(src), normalize(dst)            # Hartley 归一化
    A = []
    for (x, y), (u, v) in zip(s, d):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    H = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)          # 最小奇异值对应的右奇异向量
    H = np.linalg.inv(Td) @ H @ Ts
    return H / H[2, 2]


src = rng.uniform(0, 500, (60, 2))
dst = apply(H_true, src) + rng.normal(0, 0.5, (60, 2))
out = rng.random(60) < 0.3; dst[out] = rng.uniform(0, 600, (out.sum(), 2))     # 30% 错误匹配
err = lambda H: np.linalg.norm(apply(H, src) - dst, axis=1)

H_all = dlt(src, dst)
print(f"直接用全部点 DLT: 内点的平均重投影误差 {err(H_all)[~out].mean():.2f} 像素")

best = None
for it in range(500):                                            # RANSAC
    idx = rng.choice(60, 4, replace=False)
    H = dlt(src[idx], dst[idx])
    inl = err(H) < 3.0
    if best is None or inl.sum() > best.sum(): best = inl
H_r = dlt(src[best], dst[best])                                  # 用全部内点重新估计
print(f"RANSAC: 找到内点 {best.sum()} 个 (真实内点 {(~out).sum()} 个), 内点平均误差 {err(H_r)[~out].mean():.2f} 像素")
np.set_printoptions(suppress=True)
print("估计的 H (真实 H 为 [[1.2,0.1,30],[-0.05,0.9,10],[0.0005,0.0002,1]]):\n", np.round(H_r, 4))
