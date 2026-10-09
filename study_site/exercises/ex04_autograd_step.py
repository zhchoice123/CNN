"""
@title: 拆开黑盒：手写反向传播之后的 step 与 zero_grad
@level: 进阶
@dataset: 无（自己生成的线性回归数据）
@chapter: ch04
@minutes: 30
@goal: optimizer.step() 和 zero_grad() 到底做了什么？用最简单的线性回归，自己写出“梯度下降一步”，再证明它和 torch.optim.SGD 的结果一模一样。
@task: 不用优化器，手写梯度下降把 w、b 学出来；再用 torch.optim.SGD 重做一遍，两者结果完全一致。
@names: X, y, true_w, w, b, loss, loss_before, lr, w2, b2, opt
"""
# @step data 造一份已知答案的数据
# @explain
# 先造数据：真实规律是 $y = 2x_1 - 3x_2 + 0.5x_3 + 1$，再加一点噪声。因为答案已知，训练完可以直接检查学得对不对。
#
# `X` 是 200×3 的矩阵（200 个样本、3 个特征），`X @ true_w` 是矩阵乘向量，得到 200 个预测值。
# @end
import torch

torch.manual_seed(0)
X = torch.randn(200, 3)
true_w = torch.tensor([2.0, -3.0, 0.5])
# @todo: 用矩阵乘法算出每个样本的真实值，再加上截距 1 和少量噪声
# @blank matmul mask=X @ true_w
# @hint: 200×3 的矩阵乘以长度 3 的向量，用哪个运算符？
# @option: X * true_w
# @option: true_w @ X
# @option: X + true_w
# @accept: X.matmul(true_w)
# @accept: torch.matmul(X, true_w)
# @bad: y = (X * true_w) + 1.0 + 0.1 * torch.randn(200)
# @why: @ 是矩阵乘法，结果形状 (200,)；X * true_w 是逐元素相乘（广播），结果是 (200, 3)。
y = X @ true_w + 1.0 + 0.1 * torch.randn(200)
# @endblank
print("X:", tuple(X.shape), " y:", tuple(y.shape))
# @check: assert tuple(y.shape) == (200,), f"y 应该是长度 200 的向量，实际 {tuple(y.shape)}"

# @step params 要学习的参数：告诉 PyTorch 记录梯度
# @explain
# 模型 $\hat y = Xw + b$ 有两个参数 w（3 个数）和 b（1 个数）。创建时加上 `requires_grad=True`，PyTorch 才会记录用到它们的每一步运算，之后 `backward()` 才能算出它们的梯度。
#
# nn.Linear 里的权重默认就是这样创建的，这里我们手动做一遍。
# @end
# @todo: w 初始化为 3 个 0，b 初始化为 1 个 0，都需要梯度
# @blank reqgrad mask=requires_grad=True
# @hint: 哪个参数让张量“需要梯度”？
# @option: grad=True
# @option: requires_grad=False
# @option: autograd=True
# @accept: requires_grad = True
# @bad: w = torch.zeros(3)
# @why: requires_grad=True 告诉 autograd 追踪这个张量；没有它，backward 之后 w.grad 是 None。
w = torch.zeros(3, requires_grad=True)
# @endblank
b = torch.zeros(1, requires_grad=True)
# @check: assert w.requires_grad and b.requires_grad, "w 和 b 都要 requires_grad=True"
# @check: assert tuple(w.shape) == (3,) and tuple(b.shape) == (1,), "w 应该有 3 个元素，b 有 1 个"

# @step loss 前向传播与均方误差
# @explain
# 前向传播：$\hat y = Xw + b$。回归问题常用**均方误差**：
#
# $$L = \frac{1}{N}\sum_i (\hat y_i - y_i)^2$$
#
# 结果必须是一个**标量**（0 维张量），因为 `backward()` 默认从一个标量出发求导。
# @end
pred = X @ w + b
# @todo: 预测值减真实值、平方、再取平均
# @blank mse mask=((pred - y) ** 2).mean()
# @hint: 先平方再平均，结果是一个数
# @option: (pred - y).mean()
# @option: (pred - y) ** 2
# @option: (pred - y).abs().sum()
# @accept: ((pred-y)**2).mean()
# @accept: torch.mean((pred - y) ** 2)
# @bad: loss = (pred - y).mean()
# @why: 不平方的话正负误差会抵消；不取平均得到的是 200 个数，backward 需要标量。
loss = ((pred - y) ** 2).mean()
# @endblank
loss_before = loss.item()
print("初始损失：", round(loss_before, 4))
# @check: assert loss.dim() == 0, "loss 应该是一个标量（0 维张量），记得最后取 mean()"
# @check: assert abs(loss.item() - ((X @ w.detach() + b.detach() - y) ** 2).mean().item()) < 1e-5, "loss 应该是均方误差 ((pred - y) ** 2).mean()"

# @step backward 反向传播：梯度存到哪里了？
# @explain
# `loss.backward()` 沿着刚才记录的运算反向求导，把 $\partial L / \partial w$ 存进 **`w.grad`**（`b` 同理）。它只**计算**梯度，不会修改 w 本身。
#
# 对均方误差，可以手算出梯度 $\frac{2}{N}X^\top(\hat y - y)$，评测会用它来核对。
# @end
# @todo: 从 loss 出发反向传播
# @blank backward2 mask=backward()
# @hint: 哪个方法从损失出发计算所有参数的梯度？
# @option: grad()
# @option: step()
# @option: detach()
# @why: backward() 计算梯度并累加到 .grad；detach() 是切断计算图，step() 是优化器的方法。
loss.backward()
# @endblank
print("w 的梯度：", w.grad)
# @check: assert w.grad is not None, "w.grad 还是 None：忘了调用 loss.backward()？"
# @check: _g = 2 * X.T @ (X @ w.detach() + b.detach() - y) / len(y); assert torch.allclose(w.grad, _g, atol=1e-5), "w.grad 和手算的梯度 2/N·Xᵀ(ŷ−y) 不一致"

# @step step 手写一步梯度下降，然后清空梯度
# @explain
# `optimizer.step()` 对普通 SGD 做的事就是：
#
# $$w \leftarrow w - \text{lr} \times \frac{\partial L}{\partial w}$$
#
# 两个要点：
#
# 1. 更新要放在 `torch.no_grad()` 里——这一步只是改数值，**不能**被记进计算图
# 2. 更新完要把梯度清零（`zero_grad()` 做的就是这个），因为 PyTorch 的梯度会**累加**
# @end
lr = 0.1
# @todo: 在不记录梯度的上下文里，沿负梯度方向更新 w 和 b
# @blank nograd2 mask=no_grad
# @hint: 修改参数本身的这一步不应该被 autograd 记录
# @option: enable_grad
# @option: inference_mode_off
# @option: set_grad_enabled(True)
# @why: 在 no_grad 外直接 w -= ... 会报错：“a leaf Variable that requires grad is being used in an in-place operation”。
with torch.no_grad():
# @endblank
    # @blank update mask=w -= lr * w.grad
    # @hint: 沿着梯度的反方向走一小步
    # @option: w += lr * w.grad
    # @option: w = lr * w.grad
    # @option: w -= w.grad / lr
    # @accept: w-=lr*w.grad
    # @accept: w.sub_(lr * w.grad)
    # @bad: w += lr * w.grad
    # @why: 梯度指向损失增大的方向，所以要减去；w = lr * w.grad 会把 w 整个替换掉，也就不再是参数了。
    w -= lr * w.grad
    # @endblank
    b -= lr * b.grad
# @todo: 把累加的梯度清零，为下一步做准备
# @blank zero mask=zero_()
# @hint: 带下划线的方法表示“原地修改”，哪个把张量置零？
# @option: zero()
# @option: zeros()
# @option: clear()
# @why: Tensor.zero_() 原地置零；optimizer.zero_grad() 内部做的就是对每个参数的 .grad 置零（或设为 None）。
w.grad.zero_()
b.grad.zero_()
# @endblank
loss_after = ((X @ w + b - y) ** 2).mean().item()
print(f"走了一步：损失 {loss_before:.4f} → {loss_after:.4f}")
# @check: assert ((X @ w.detach() + b.detach() - y) ** 2).mean().item() < loss_before, "更新一步后损失应该变小：检查是不是减去了梯度"
# @check: assert float(w.grad.abs().sum()) == 0.0 and float(b.grad.abs().sum()) == 0.0, "更新后要把 w.grad 和 b.grad 清零"

# @step loop 把这一步重复 100 次
# @explain
# 训练就是把“前向 → 损失 → 反向 → 更新 → 清零”重复很多次。注意每一轮都要**重新做前向传播**，因为参数变了，预测也跟着变。
# @end
for i in range(100):
    pred = X @ w + b
    loss = ((pred - y) ** 2).mean()
    # @todo: 反向传播、更新、清零——和上一步完全一样
    # @blank loop_bw mask=loss.backward()
    # @hint: 每一轮先算出梯度
    # @option: loss.step()
    # @option: w.backward()
    # @option: loss.grad()
    # @bad: pass
    # @why: 每一轮的损失都是新算出来的，要重新 backward 才有新的梯度。
    loss.backward()
    # @endblank
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
print("学到的 w：", w.detach().numpy().round(3), " b：", round(b.item(), 3), "（真实值 [2, -3, 0.5] 和 1）")
# @check: assert torch.allclose(w.detach(), true_w, atol=0.1), f"100 步后 w 应该接近 [2, -3, 0.5]，实际 {w.detach().tolist()}"
# @check: assert abs(b.item() - 1.0) < 0.1, f"b 应该接近 1，实际 {b.item():.3f}"

# @step optim 换成 torch.optim.SGD：结果完全一样
# @explain
# 现在用 PyTorch 的优化器重做一遍。从同样的初值出发、同样的学习率、同样的步数，结果应该和手写版本**分毫不差**——这就证明了：
#
# - `optimizer.zero_grad()` = 把每个参数的 `.grad` 清零
# - `optimizer.step()` = 在 no_grad 下做 `参数 -= lr × 梯度`
#
# 优化器要知道它负责哪些参数，所以创建时把参数列表传进去。
# @end
w2 = torch.zeros(3, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
# @todo: 让 SGD 负责更新 w2 和 b2，学习率与手写版相同
# @blank optlist mask=[w2, b2]
# @hint: 优化器的第一个参数是“要更新的张量”组成的列表
# @option: w2, b2
# @option: [w2.grad, b2.grad]
# @option: [w, b]
# @accept: (w2, b2)
# @why: SGD 接收一个可迭代的参数集合；传 w、b 会去更新已经训练好的那组参数。
opt = torch.optim.SGD([w2, b2], lr=lr)
# @endblank
for i in range(101):          # 手写版本一共更新了 1 + 100 步
    loss2 = ((X @ w2 + b2 - y) ** 2).mean()
    opt.zero_grad()
    loss2.backward()
    opt.step()
print("手写:", w.detach().numpy().round(5), " SGD:", w2.detach().numpy().round(5))
# @check: assert torch.allclose(w2.detach(), w.detach(), atol=1e-5) and torch.allclose(b2.detach(), b.detach(), atol=1e-5), "torch.optim.SGD 的结果应该和手写版本完全一致"
