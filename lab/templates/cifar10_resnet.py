"""CIFAR-10 + 手写残差块（ResNet-20 风格）+ SGD 动量 + OneCycle 学习率。
残差连接的关键就一行：return relu(F(x) + x)。"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from labkit import get_device, sync, load_tensors, log_epoch

# ===== 可修改的超参数 =====
LR, EPOCHS, BATCH, WD = 0.1, 10, 128, 5e-4
BLOCKS_PER_STAGE = 3                                             # 3 → ResNet-20

device = get_device()
xtr, ytr, xte, yte = load_tensors("cifar10", device)


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)
        self.short = nn.Identity() if (stride == 1 and cin == cout) else \
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + self.short(x))                        # ← 残差连接


layers, c = [nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU()], 16
for i, out in enumerate((16, 32, 64)):
    for j in range(BLOCKS_PER_STAGE):
        layers.append(Block(c, out, 2 if (j == 0 and i > 0) else 1)); c = out
model = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)).to(device)

steps = (len(xtr) + BATCH - 1) // BATCH
opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps)


def augment(x):
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    dy, dx = torch.randint(-3, 4, (2,)).tolist()
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(xtr), device=device)
    sync(device); t0 = time.time()
    loss_sum, correct = 0.0, 0
    for i in range(0, len(xtr), BATCH):
        idx = perm[i:i + BATCH]
        x, y = augment(xtr[idx]), ytr[idx]
        out = model(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        loss_sum += loss.item() * len(idx); correct += (out.argmax(1) == y).sum().item()
    sync(device); dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(xte[j:j + 1000]) for j in range(0, len(xte), 1000)])
        test_loss, test_acc = F.cross_entropy(out, yte).item(), (out.argmax(1) == yte).float().mean().item()
    log_epoch(epoch, train_loss=loss_sum / len(xtr), train_acc=correct / len(xtr), test_loss=test_loss,
              test_acc=test_acc, epoch_time=dt, img_per_s=len(xtr) / dt, lr=opt.param_groups[0]["lr"])
