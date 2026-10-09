"""分割评估与上采样：像素准确率的陷阱、IoU 与 Dice 系数；转置卷积的输出尺寸。"""
import numpy as np

H = W = 100
gt = np.zeros((H, W), bool); gt[40:60, 40:60] = True                      # 目标只占 4% 的像素
pred_empty = np.zeros_like(gt)                                            # 什么都不预测
pred_shift = np.zeros_like(gt); pred_shift[45:65, 45:65] = True           # 预测框偏移了 5 像素


def metrics(p, g):
    inter = (p & g).sum(); union = (p | g).sum()
    return (p == g).mean(), inter / union if union else 1.0, 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0


for name, p in [("全预测为背景", pred_empty), ("偏移 5 像素", pred_shift)]:
    acc, iou, dice = metrics(p, gt)
    print(f"{name:10s} 像素准确率 {acc:.3f}  IoU {iou:.3f}  Dice {dice:.3f}")

print("\n转置卷积输出尺寸 o = (n-1)*s - 2p + k:")
for n, k, s, p in [(7, 4, 2, 1), (14, 4, 2, 1), (28, 2, 2, 0)]:
    print(f"  输入 {n:2d}, 核 {k}, 步长 {s}, 填充 {p} -> 输出 {(n - 1) * s - 2 * p + k}")
