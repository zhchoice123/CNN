"""统一训练脚本（由 server.py 以子进程方式启动，也可以手动运行）。

    .venv/bin/python -I train.py --run-dir runs/demo --config '{"dataset":"mnist","model":"lenet","epochs":3}'

设计要点（为了让“参数对比”公平且快）：
  * 整个数据集一次性放进设备内存（MNIST/CIFAR 都很小），批量切片 + 在设备上做增强，不用 DataLoader，没有 CPU 瓶颈。
  * 每个 epoch 写一行 JSON 到 metrics.jsonl，含训练/测试损失与准确率、耗时、吞吐（图/秒）。
  * 计时前会同步设备（MPS 是异步的），所以 epoch_time 是真实耗时。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # -I 模式不会自动加入脚本目录

import torch                          # noqa: E402
import torch.nn.functional as F       # noqa: E402

import datasets_manager as dm         # noqa: E402
import platform_utils as pu           # noqa: E402
import lab_models                     # noqa: E402
import lab_schema                     # noqa: E402

MEAN_STD = {
    "mnist": ((0.1307,), (0.3081,)), "fmnist": ((0.2860,), (0.3530,)), "kmnist": ((0.1918,), (0.3483,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
}

STOP = {"flag": False}


def _on_term(*_):
    STOP["flag"] = True


def pick_device(want):
    """auto：CUDA（NVIDIA）→ MPS（Apple 芯片）→ CPU。指定的设备不可用时退回 CPU 并提示。"""
    if want in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if want in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if want not in ("auto", "cpu"):
        print(f"⚠ 这台电脑没有可用的 {want}，改用 CPU", flush=True)
    return torch.device("cpu")


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def prep(xb, mean, std):
    return (xb.float() / 255.0 - mean) / std


def augment(x, mode, r):
    """x: 已归一化的 B×C×H×W；r: 3×B 的 [0,1) 随机数（每个 epoch 在 CPU 上生成一次、整体搬到设备，避免每步同步）。
    翻转和平移合并成一次 grid_sample：最近邻 + 零填充，等价于“补 4 像素再随机裁剪”（零 = 归一化后的均值）。"""
    B, C, H, W = x.shape
    flip = (r[0] < 0.5) if "flip" in mode else torch.zeros(B, dtype=torch.bool, device=x.device)
    if "crop" in mode:
        dx, dy = torch.floor(r[1] * 9) - 4, torch.floor(r[2] * 9) - 4          # -4 … +4 像素
    else:
        dx = dy = torch.zeros(B, device=x.device)
    theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
    sign = torch.where(flip, -1.0, 1.0)
    theta[:, 0, 0] = sign
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = sign * dx * 2.0 / W          # 翻转后平移方向也要跟着反过来，才等价于“先翻转、再裁剪”
    theta[:, 1, 2] = dy * 2.0 / H
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)


def make_optimizer(cfg, params):
    o, lr, wd, mom = cfg["optimizer"], cfg["lr"], cfg["weight_decay"], cfg["momentum"]
    if o == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    if o == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd)
    if o == "nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd, nesterov=True)
    if o == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if o == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd if wd else 0.01)
    if o == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=mom * 0, weight_decay=wd)
    raise ValueError(o)


def make_scheduler(cfg, opt, steps_per_epoch):
    total = cfg["epochs"] * steps_per_epoch
    s = cfg["scheduler"]
    if s == "none":
        return None
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, total // 3), gamma=0.1)
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    if s == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total, pct_start=0.25)
    raise ValueError(s)


@torch.no_grad()
def evaluate(model, x, y, mean, std, bs=1000):
    model.eval()
    tot_loss, correct = torch.zeros((), device=x.device), torch.zeros((), device=x.device)
    for i in range(0, len(x), bs):
        xb, yb = prep(x[i:i + bs], mean, std), y[i:i + bs]
        out = model(xb)
        tot_loss += F.cross_entropy(out, yb, reduction="sum")
        correct += (out.argmax(1) == yb).sum()
    return tot_loss.item() / len(x), correct.item() / len(x)


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default="{}")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pu.install_stop_handlers(_on_term)      # macOS: SIGTERM/SIGINT；Windows: CTRL_BREAK(SIGBREAK)/SIGINT

    cfg = lab_schema.normalize(json.loads(args.config))
    if not dm.is_ready(cfg["dataset"]):
        print(f"数据集 {cfg['dataset']} 尚未下载。请先在“数据集”面板下载，或运行：python -I datasets_manager.py download {cfg['dataset']}", flush=True)
        sys.exit(3)

    torch.manual_seed(cfg["seed"])
    dev = pick_device(cfg["device"])
    cpu_g = torch.Generator().manual_seed(cfg["seed"])

    t0 = time.time()
    xtr, ytr = dm.load_arrays(cfg["dataset"], "train")
    xte, yte = dm.load_arrays(cfg["dataset"], "test")
    if cfg["train_subset"] and cfg["train_subset"] < len(xtr):
        idx = torch.randperm(len(xtr), generator=cpu_g)[:cfg["train_subset"]]
        xtr, ytr = xtr[idx], ytr[idx]
    spec = dm.SPECS[cfg["dataset"]]
    in_ch, size, ncls = xtr.shape[1], xtr.shape[2], spec["classes"]
    mean_t, std_t = MEAN_STD[cfg["dataset"]]
    mean = torch.tensor(mean_t, device=dev).view(1, -1, 1, 1)
    std = torch.tensor(std_t, device=dev).view(1, -1, 1, 1)
    xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)
    load_time = time.time() - t0

    model = lab_models.build(cfg["model"], in_ch, ncls, size, width=cfg["width"], depth=cfg["depth"],
                             dropout=cfg["dropout"], norm=cfg["norm"]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(cfg, model.parameters())
    N, bs = len(xtr), cfg["batch_size"]
    steps = math.ceil(N / bs)
    sched = make_scheduler(cfg, opt, steps)

    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True   # 输入尺寸固定时让 cuDNN 选最快的卷积算法（RTX 3060 等 NVIDIA 显卡）
    info = dict(device=str(dev), hw=pu.hardware_info(), params=n_params, n_train=N, n_test=len(xte), steps_per_epoch=steps, load_time=round(load_time, 2),
                torch=torch.__version__)
    write_json(run_dir / "info.json", info)
    print(f"[{cfg['model']}] 参数 {n_params:,} · 设备 {dev} · 训练样本 {N} · 每轮 {steps} 步 · 数据载入 {load_time:.1f}s", flush=True)

    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    status, train_elapsed, t_start = "done", 0.0, time.time()
    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(N, generator=cpu_g).to(dev)
        rnd = torch.rand(3, N, generator=cpu_g).to(dev) if cfg["augment"] != "none" else None
        loss_sum = torch.zeros((), device=dev)
        correct = torch.zeros((), device=dev)
        sync(dev)
        t_ep = time.time()
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            xb, yb = prep(xtr[idx], mean, std), ytr[idx]
            if cfg["augment"] != "none":
                xb = augment(xb, cfg["augment"], rnd[:, s * bs:(s + 1) * bs])
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if sched is not None:
                sched.step()
            loss_sum += loss.detach() * len(idx)
            correct += (out.argmax(1) == yb).sum()
            if s % 25 == 0 or s == steps - 1:
                if STOP["flag"]:
                    break
                write_json(run_dir / "progress.json", dict(epoch=epoch, step=s + 1, steps=steps, loss=float(loss.item()), t=time.time()))
                if not math.isfinite(float(loss.item())):
                    status = "diverged"
                    break
        sync(dev)
        ep_time = time.time() - t_ep
        if STOP["flag"]:
            status = "stopped"
        if status in ("stopped", "diverged"):
            break
        train_elapsed += ep_time
        t_eval = time.time()
        te_loss, te_acc = evaluate(model, xte, yte, mean, std)
        eval_time = time.time() - t_eval
        row = dict(epoch=epoch, train_loss=loss_sum.item() / N, train_acc=correct.item() / N, test_loss=te_loss, test_acc=te_acc,
                   lr=opt.param_groups[0]["lr"], epoch_time=ep_time, eval_time=eval_time, train_elapsed=train_elapsed,
                   wall=time.time() - t_start, img_per_s=N / ep_time)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if te_acc > best_acc:
            best_acc, best_epoch = te_acc, epoch
        print(f"epoch {epoch:>3}/{cfg['epochs']}  训练损失 {row['train_loss']:.4f}  训练准确率 {row['train_acc']:.4f}  "
              f"测试准确率 {te_acc:.4f}  用时 {ep_time:.1f}s  {row['img_per_s']:.0f} 图/秒", flush=True)

    summary = dict(status=status, best_test_acc=best_acc, best_epoch=best_epoch, train_time=train_elapsed,
                   wall_time=time.time() - t_start, **info)
    write_json(run_dir / "summary.json", summary)
    print(f"结束：{status} · 最佳测试准确率 {best_acc:.4f}（第 {best_epoch} 轮）· 纯训练耗时 {train_elapsed:.1f}s", flush=True)
    sys.exit(0 if status in ("done", "stopped") else 4)


if __name__ == "__main__":
    main()
