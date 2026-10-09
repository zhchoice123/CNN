"""平均精度 AP：按置信度排序检测结果，画出精确率-召回率曲线，再计算曲线下面积。"""
import numpy as np

# 10 个检测结果（已按置信度从高到低排序），1 = 与某个真实框 IoU≥0.5 的正确检测；图中共有 6 个真实目标
tp = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
n_gt = 6
cum_tp = np.cumsum(tp); cum_fp = np.cumsum(1 - tp)
precision = cum_tp / (cum_tp + cum_fp); recall = cum_tp / n_gt
for k in range(len(tp)):
    print(f"前 {k+1:2d} 个: 精确率 {precision[k]:.3f}  召回率 {recall[k]:.3f}")

# 所有点插值：把精确率改成“右侧最大值”，再对召回率积分（VOC2010+ / COCO 的做法）
p_interp = np.maximum.accumulate(precision[::-1])[::-1]
r = np.concatenate([[0], recall])
ap = np.sum((r[1:] - r[:-1]) * p_interp)
print(f"AP = {ap:.4f}")
