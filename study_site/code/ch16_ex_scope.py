"""例题：预测输出。"""
x = 10
def f():
    x = 20
    def g():
        return x
    return g()
print("f() =", f(), " 全局 x =", x)

nums = [1, 2, 3]
squares = (n * n for n in nums)                # 生成器：惰性求值
nums.append(4)
print("生成器在迭代时才计算:", list(squares))
print("再次迭代同一个生成器:", list(squares), "<- 生成器只能遍历一次")
