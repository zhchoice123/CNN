"""针孔相机模型：x = K [R | t] X。把三维点投影到图像上，并观察平行线汇聚到消失点。"""
import numpy as np

f, cx, cy = 800.0, 320.0, 240.0
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])                  # 内参：焦距与主点
theta = np.deg2rad(10)
R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])  # 绕 y 轴转 10°
t = np.array([0.0, 0.0, 5.0])                                      # 外参：平移
P = K @ np.column_stack([R, t])                                    # 3x4 相机矩阵


def project(X):
    x = P @ np.append(X, 1.0)                                      # 齐次坐标
    return x[:2] / x[2]                                            # 除以第三个分量（透视除法）


for X in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 10)]:
    print(f"三维点 {X} -> 像素 {np.round(project(np.array(X, float)), 1)}")

# 两条平行于 z 轴的直线，走得越远，投影越接近同一个消失点
for z in (10, 100, 10000):
    a, b = project(np.array([-1.0, 0, z])), project(np.array([1.0, 0, z]))
    print(f"z={z:6d}: 左线 {np.round(a, 1)}  右线 {np.round(b, 1)}")
d = R @ np.array([0, 0, 1.0])                                      # 方向 (0,0,1) 的消失点 = K R d
v = K @ d
print("理论消失点:", np.round(v[:2] / v[2], 1))
