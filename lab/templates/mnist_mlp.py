"""MNIST + 全连接网络（最小的完整训练循环）。
从这里读懂：数据 → 模型 → 损失 → 反向传播 → 更新 → 评估 这条主线。"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from labkit import get_device, sync, load_tensors, log_epoch

# ===== 可修改的超参数 =====
LR, EPOCHS, BATCH = 0.1, 5, 128
HIDDEN = 256

device = get_device()
xtr, ytr, xte, yte = load_tensors("mnist", device)          # 已归一化，直接读本地数据集

model = nn.Sequential(nn.Flatten(), nn.Linear(784, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 10)).to(device)
opt = torch.optim.SGD(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(xtr), device=device)
    sync(device); t0 = time.time()
    loss_sum, correct = 0.0, 0
    for i in range(0, len(xtr), BATCH):
        idx = perm[i:i + BATCH]
        x, y = xtr[idx], ytr[idx]
        out = model(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()          # 反向传播 + 参数更新
        loss_sum += loss.item() * len(idx); correct += (out.argmax(1) == y).sum().item()
    sync(device); dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        out = model(xte)
        test_loss, test_acc = F.cross_entropy(out, yte).item(), (out.argmax(1) == yte).float().mean().item()
    log_epoch(epoch, train_loss=loss_sum / len(xtr), train_acc=correct / len(xtr), test_loss=test_loss,
              test_acc=test_acc, epoch_time=dt, img_per_s=len(xtr) / dt, lr=LR)
