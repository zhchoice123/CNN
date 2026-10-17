"""MNIST + LeNet-5：最经典的卷积网络（卷积 → 池化 → 卷积 → 池化 → 全连接）。"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from labkit import get_device, sync, load_tensors, log_epoch

# ===== 可修改的超参数 =====
LR, EPOCHS, BATCH = 0.002, 5, 128

device = get_device()
xtr, ytr, xte, yte = load_tensors("mnist", device)

model = nn.Sequential(
    nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),     # 28×28 → 14×14，6 个特征图
    nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),               # 10×10 → 5×5，16 个特征图
    nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

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
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * len(idx); correct += (out.argmax(1) == y).sum().item()
    sync(device); dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(xte[j:j + 1000]) for j in range(0, len(xte), 1000)])
        test_loss, test_acc = F.cross_entropy(out, yte).item(), (out.argmax(1) == yte).float().mean().item()
    log_epoch(epoch, train_loss=loss_sum / len(xtr), train_acc=correct / len(xtr), test_loss=test_loss,
              test_acc=test_acc, epoch_time=dt, img_per_s=len(xtr) / dt, lr=LR)
