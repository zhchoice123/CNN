"""Python 对象模型的三个常见陷阱：共享引用、可变默认参数、作用域（LEGB）。"""

# 1. 变量只是名字，赋值让名字指向对象；两个名字可以指向同一个列表
a = [1, 2, 3]
b = a                       # 不是复制！
b.append(4)
print("共享引用:", a, " a is b:", a is b)
c = a.copy(); c.append(5)
print("复制之后:", a, c)


# 2. 默认参数只在函数定义时求值一次
def add_item(x, bucket=[]):
    bucket.append(x); return bucket


print("可变默认参数:", add_item(1), add_item(2))      # 第二次调用看到了第一次的结果


def add_item_ok(x, bucket=None):
    bucket = [] if bucket is None else bucket
    bucket.append(x); return bucket


print("正确写法:", add_item_ok(1), add_item_ok(2))

# 3. LEGB：Local -> Enclosing -> Global -> Built-in；修改外层变量需要 nonlocal
count = 0


def make_counter():
    count = 0
    def inc():
        nonlocal count      # 没有这一行会抛出 UnboundLocalError
        count += 1; return count
    return inc


inc = make_counter()
print("闭包计数器:", inc(), inc(), inc(), " 全局 count 仍为", count)
s = "abc"; t = s; s += "d"                     # 字符串不可变：+= 创建了新对象，t 不受影响
print("不可变对象:", s, t)
