"""分割与形态学：大津法自动选阈值，开运算去除小噪点，再数连通区域个数。"""
import numpy as np

rng = np.random.default_rng(3)
img = np.full((80, 80), 60.0)
for cy, cx, r in [(20, 20, 9), (25, 60, 7), (60, 30, 11), (58, 62, 6)]:  # 4 个亮的“细胞”
    yy, xx = np.mgrid[:80, :80]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 170
img += rng.normal(0, 15, img.shape)
img[rng.random(img.shape) < 0.01] = 200                                     # 零星亮噪点
img = np.clip(img, 0, 255).astype(np.uint8)

hist = np.bincount(img.ravel(), minlength=256) / img.size
best_t, best_var = 0, -1
for t in range(1, 256):                                                      # 大津法：最大化类间方差
    w0, w1 = hist[:t].sum(), hist[t:].sum()
    if w0 == 0 or w1 == 0: continue
    m0 = (np.arange(t) * hist[:t]).sum() / w0; m1 = (np.arange(t, 256) * hist[t:]).sum() / w1
    var = w0 * w1 * (m0 - m1) ** 2
    if var > best_var: best_t, best_var = t, var
binary = img >= best_t
print("大津法阈值:", best_t)


def erode(b): p = np.pad(b, 1); return np.all([p[i:i+80, j:j+80] for i in range(3) for j in range(3)], 0)
def dilate(b): p = np.pad(b, 1); return np.any([p[i:i+80, j:j+80] for i in range(3) for j in range(3)], 0)


def count_components(b):
    lab = np.zeros(b.shape, int); n = 0
    for s in zip(*np.nonzero(b)):
        if lab[s]: continue
        n += 1; stack = [s]; lab[s] = n
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = (y + dy, x + dx)
                if 0 <= v[0] < 80 and 0 <= v[1] < 80 and b[v] and not lab[v]:
                    lab[v] = n; stack.append(v)
    return n


print("阈值分割后的连通区域数:", count_components(binary))
opened = dilate(erode(binary))                                               # 开运算 = 先腐蚀再膨胀
print("开运算后的连通区域数:", count_components(opened), "(应为 4 个细胞)")
