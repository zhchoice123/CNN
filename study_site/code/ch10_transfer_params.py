"""迁移学习：VGG16 各部分的参数量，以及“冻结卷积基、只训练新分类头”能少训练多少参数。"""


def conv(cin, cout, k=3): return (k * k * cin + 1) * cout
def fc(n_in, n_out): return (n_in + 1) * n_out


blocks = [[(3, 64), (64, 64)], [(64, 128), (128, 128)], [(128, 256), (256, 256), (256, 256)],
          [(256, 512), (512, 512), (512, 512)], [(512, 512), (512, 512), (512, 512)]]
conv_base = sum(conv(a, b) for blk in blocks for a, b in blk)
top = fc(7 * 7 * 512, 4096) + fc(4096, 4096) + fc(4096, 1000)
print(f"VGG16 卷积基参数: {conv_base:,}")
print(f"VGG16 原分类头 (3 个全连接层) 参数: {top:,}")
print(f"合计: {conv_base + top:,}  (约 1.38 亿，大部分在全连接层)")

new_head = fc(512, 5)                       # 全局平均池化 -> 5 类新任务
last_block = sum(conv(a, b) for a, b in blocks[-1])
print(f"\n新任务 (5 类，用全局平均池化+一个全连接层):")
print(f"  特征提取 (冻结全部卷积基): 需训练 {new_head:,} 个参数")
print(f"  微调最后一个卷积块 + 新头:   需训练 {last_block + new_head:,} 个参数")
print(f"  全部微调:                   需训练 {conv_base + new_head:,} 个参数")
