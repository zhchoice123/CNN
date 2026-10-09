"""例题：由混淆矩阵计算各项指标。"""
tp, fp, fn, tn = 40, 10, 20, 930
acc = (tp + tn) / (tp + fp + fn + tn)
prec = tp / (tp + fp); rec = tp / (tp + fn)
f1 = 2 * prec * rec / (prec + rec)
spec = tn / (tn + fp)
print(f"准确率 {acc:.3f}  精确率 {prec:.3f}  召回率 {rec:.3f}  F1 {f1:.3f}  特异度 {spec:.3f}")
