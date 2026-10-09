"""优化器对比：在一个狭长的“峡谷”形二次函数上比较 SGD、动量、RMSProp 与 Adam。"""
import numpy as np

A = np.diag([1.0, 50.0])                      # 条件数 50：一个方向平缓、一个方向陡峭
f = lambda w: 0.5 * w @ A @ w
grad = lambda w: A @ w
start = np.array([10.0, 1.0])


def run(update):
    w, state, log = start.copy(), {}, []
    for t in range(1, 201):
        w = update(w, grad(w), state, t)
        if t in (10, 50, 200): log.append(f"{f(w):9.2e}")
    return "   ".join(log)


def sgd(w, g, s, t, lr=0.035): return w - lr * g
def momentum(w, g, s, t, lr=0.035, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g; return w - lr * s["v"]
def rmsprop(w, g, s, t, lr=1.0, beta=0.9, eps=1e-8):
    lr = lr / np.sqrt(t)                      # 学习率衰减，否则会在最小值附近以 lr 的幅度来回跳
    s["r"] = beta * s.get("r", 0) + (1 - beta) * g ** 2; return w - lr * g / (np.sqrt(s["r"]) + eps)
def adam(w, g, s, t, lr=0.3, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g; s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    m_hat = s["m"] / (1 - b1 ** t); v_hat = s["v"] / (1 - b2 ** t)
    return w - lr * m_hat / (np.sqrt(v_hat) + eps)


print(f"起点损失 f = {f(start):.1f}")
for name, fn in [("SGD", sgd), ("Momentum", momentum), ("RMSProp", rmsprop), ("Adam", adam)]:
    print(f"{name:9s} 第10/50/200步的损失: {run(fn)}")
print("SGD 的学习率受陡峭方向限制 (需 < 2/50)，在平缓方向上走得很慢；动量在平缓方向上累积速度；")
print("RMSProp/Adam 按每个方向梯度的大小单独缩放步长，两个方向的步子差不多大。没有哪种优化器在所有问题上都最好。")
