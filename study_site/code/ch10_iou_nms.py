"""目标检测的基本工具：交并比 IoU 与非极大值抑制 NMS。框格式为 (x1, y1, x2, y2)。"""
import numpy as np


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def nms(boxes, scores, thr=0.5):
    order = list(np.argsort(scores)[::-1]); keep = []
    while order:
        i = order.pop(0); keep.append(i)                     # 保留当前得分最高的框
        order = [j for j in order if iou(boxes[i], boxes[j]) < thr]   # 删除与它重叠过大的框
    return keep


print("IoU 例子:", round(iou((0, 0, 10, 10), (5, 5, 15, 15)), 4))
boxes = [(10, 10, 60, 60), (12, 14, 62, 58), (8, 12, 55, 65), (100, 100, 150, 160), (105, 98, 148, 150)]
scores = np.array([0.90, 0.75, 0.60, 0.85, 0.70])
keep = [int(i) for i in nms(boxes, scores)]
print("NMS 保留的框索引:", keep, " 对应得分:", scores[keep].tolist())
for i in range(1, 3):
    print(f"框0 与 框{i} 的 IoU = {iou(boxes[0], boxes[i]):.3f}")
