"""读懂 PyTorch 的自动微分：用 Python 类与运算符重载实现一个 40 行的标量自动微分引擎。"""
import math


class Value:
    def __init__(self, data, children=(), op=""):
        self.data, self.grad = data, 0.0
        self._prev, self._op, self._backward = children, op, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad; other.grad += out.grad
        out._backward = _backward; return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad; other.grad += self.data * out.grad
        out._backward = _backward; return out

    __radd__ = __add__
    __rmul__ = __mul__

    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), "tanh")
        def _backward(): self.grad += (1 - t * t) * out.grad
        out._backward = _backward; return out

    def backward(self):                        # 拓扑排序后按逆序应用链式法则
        order, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v); [build(c) for c in v._prev]; order.append(v)
        build(self); self.grad = 1.0
        for v in reversed(order): v._backward()

    def __repr__(self): return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# 一个神经元: y = tanh(w1*x1 + w2*x2 + b)
x1, x2, w1, w2, b = Value(2.0), Value(0.0), Value(-3.0), Value(1.0), Value(6.8814)
y = (x1 * w1 + x2 * w2 + b).tanh()
y.backward()
print("y  =", y)
print("w1 =", w1, " (dy/dw1 = x1 * (1 - y^2))")
print("x1 =", x1)
eps = 1e-6                                     # 数值梯度验证
f = lambda w: math.tanh(2.0 * w + 0.0 * 1.0 + 6.8814)
print(f"数值梯度 dy/dw1 = {(f(-3 + eps) - f(-3 - eps)) / (2 * eps):.4f}")
