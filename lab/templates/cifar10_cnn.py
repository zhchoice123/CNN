"""CIFAR-10 + 带批量归一化和 Dropout 的 CNN，并加入随机翻转/平移增强。
对比实验建议：去掉 BatchNorm、去掉增强，看准确率和收敛速度怎么变。"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from labkit import get_device, sync, load_tensors, log_epoch

# ===== 可修改的超参数 =====
LR, EPOCHS, BATCH = 0.002, 10, 128
USE_BN, DROPOUT, AUGMENT = True, 0.3, True

device = get_device()
xtr, ytr, xte, yte = load_tensors("cifar10", device)


def block(cin, cout):
    layers = [nn.Conv2d(cin, cout, 3, padding=1)]
    if USE_BN:
        layers.append(nn.BatchNorm2d(cout))
    return nn.Sequential(*layers, nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))


model = nn.Sequential(block(3, 32), block(32, 64), block(64, 128),         # 32→16→8→4
                      nn.Flatten(), nn.Dropout(DROPOUT), nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
                      nn.Dropout(DROPOUT), nn.Linear(256, 10)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)


def augment(x):
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)                # 随机左右翻转
    dy, dx = torch.randint(-3, 4, (2,)).tolist()                            # 整批随机平移（简化版）
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(xtr), device=device)
    sync(device); t0 = time.time()
    loss_sum, correct = 0.0, 0
    for i in range(0, len(xtr), BATCH):
        idx = perm[i:i + BATCH]
        x, y = xtr[idx], ytr[idx]
        if AUGMENT:
            x = augment(x)
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
