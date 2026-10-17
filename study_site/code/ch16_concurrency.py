"""操作系统视角：I/O 密集任务适合多线程，CPU 密集任务受 GIL 限制，需要多进程。"""
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def io_task(_):
    time.sleep(0.2)                            # 模拟读磁盘/网络：等待期间释放 GIL
    return 1


def cpu_task(n):
    return sum(i * i for i in range(n))        # 纯 Python 计算：持有 GIL


def timed(fn):
    t = time.perf_counter(); fn(); return time.perf_counter() - t


if __name__ == "__main__":
    print(f"8 个 I/O 任务  串行 {timed(lambda: [io_task(i) for i in range(8)]):.2f}s", end="  ")
    with ThreadPoolExecutor(8) as ex:
        print(f"8 线程 {timed(lambda: list(ex.map(io_task, range(8)))):.2f}s")
    N = [2_000_000] * 4
    serial = timed(lambda: [cpu_task(n) for n in N])
    with ThreadPoolExecutor(4) as ex:
        threads = timed(lambda: list(ex.map(cpu_task, N)))
    with ProcessPoolExecutor(4) as ex:
        ex.submit(cpu_task, 10).result()       # 预热，排除进程启动时间
        procs = timed(lambda: list(ex.map(cpu_task, N)))
    print(f"4 个 CPU 任务  串行 {serial:.2f}s  4 线程 {threads:.2f}s  4 进程 {procs:.2f}s")
    print("结论: 线程适合等待型任务；CPU 密集的纯 Python 代码要用多进程（如 PyTorch DataLoader 的 num_workers）")
