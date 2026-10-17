"""分类评估：混淆矩阵、精确率、召回率、F1，以及 ROC 曲线下的面积 AUC。"""
import numpy as np

rng = np.random.default_rng(1)
y = (rng.random(1000) < 0.1).astype(int)                     # 正类只占 10%
score = np.where(y == 1, rng.normal(1.0, 1, 1000), rng.normal(-0.5, 1, 1000))


def report(th):
    pred = (score >= th).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / (tp + fn); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"阈值 {th:5.2f}: TP={tp:3d} FP={fp:3d} FN={fn:3d} TN={tn:3d}  准确率={(tp+tn)/1000:.3f} 精确率={prec:.3f} 召回率={rec:.3f} F1={f1:.3f}")


for th in (-0.5, 0.5, 1.5):
    report(th)
print(f"全部预测为负类的“傻瓜模型”准确率: {(y == 0).mean():.3f}  <- 准确率悖论")

# AUC = 随机取一个正例和一个负例，正例得分更高的概率
pos, neg = score[y == 1], score[y == 0]
auc = (pos[:, None] > neg[None, :]).mean()
print(f"AUC = {auc:.3f}")
