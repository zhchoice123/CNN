"""例题：12 个球称重问题的信息论下界。"""
import math
outcomes = 12 * 2                     # 哪个球异常 × 偏重还是偏轻
need = math.log2(outcomes)
per_weigh = math.log2(3)              # 天平三种结果
print(f"需要的信息量 log2(24) = {need:.3f} 比特")
print(f"每次称重最多获得 log2(3) = {per_weigh:.3f} 比特")
print(f"最少称重次数 >= {need/per_weigh:.3f} -> {math.ceil(need/per_weigh)} 次")
print(f"3 次称重最多区分 3^3 = {3**3} 种结果 >= 24")
