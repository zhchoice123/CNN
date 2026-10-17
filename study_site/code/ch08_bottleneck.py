"""例题：Inception 中 1x1 卷积“瓶颈层”节省的乘法次数。"""
H = W = 28
cin, cout = 192, 32

direct = H * W * cout * 5 * 5 * cin                 # 直接 5x5 卷积
reduce_ = H * W * 16 * 1 * 1 * cin                  # 先 1x1 降到 16 通道
then_5 = H * W * cout * 5 * 5 * 16                  # 再 5x5 卷积
print(f"直接 5x5 卷积乘法次数: {direct:,}")
print(f"1x1 降维: {reduce_:,} + 5x5: {then_5:,} = {reduce_ + then_5:,}")
print(f"计算量约为原来的 {(reduce_ + then_5) / direct:.1%}")
