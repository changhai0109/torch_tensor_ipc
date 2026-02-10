import torch
import torch.multiprocessing as mp
import torch_tensor_ipc.cuda_ipc_ext as ext  # 导入刚才写的封装模块
import time
import os


def consumer_process(rank, queue):
    """
    消费者进程：接收 Handle -> 重建 Tensor -> 验证数据 -> 修改数据
    """
    print(f"[Consumer] Started (PID: {os.getpid()})")

    # 1. 从队列获取元数据
    # meta: (handle_bytes, shape, dtype, device_index)
    try:
        handle_bytes, shape, dtype, device_idx = queue.get(timeout=10)
    except Exception as e:
        print(f"[Consumer] Queue empty or error: {e}")
        return

    print(f"[Consumer] Received handle. Reconstructing tensor...")

    # 2. 重建 Tensor (Zero-copy)
    # 注意：这个 tensor 指向的是 Producer 的显存
    t_remote = ext.tensor_from_ipc_bytes(handle_bytes, shape, dtype, device_idx)

    print(f"[Consumer] Tensor info: shape={t_remote.shape}, device={t_remote.device}")
    print(f"[Consumer] Data sample: {t_remote[0, :5]}")

    # 3. 修改数据 (证明是共享内存)
    # 给所有元素 +100
    t_remote.add_(100.0)
    print(f"[Consumer] Added 100 to tensor. Synced.")

    # 4. 保持进程存活一小会儿，确保 Producer 还没释放内存
    # 实际上只要 Producer 不释放，Consumer 就可以一直读
    del t_remote  # 触发 deleter
    print(f"[Consumer] Done.")


def run_test():
    # 必须使用 spawn 启动方式
    mp.set_start_method("spawn", force=True)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available.")
        return

    # 1. Producer (主进程) 创建 Tensor
    device_id = 0
    t_original = torch.randn(10, 10, device=f"cuda:{device_id}")
    print(f"[Producer] Original tensor sample: {t_original[0, :5]}")

    # 2. 导出 Handle
    handle_bytes, nbytes = ext.export_tensor_ipc(t_original)
    print(f"[Producer] Exported handle ({len(handle_bytes)} bytes)")

    # 3. 启动消费者进程
    queue = mp.Queue()
    p = mp.Process(target=consumer_process, args=(1, queue))
    p.start()

    # 4. 发送元数据给消费者
    # 注意：我们必须把 shape 和 dtype 也传过去，因为 handle 里不包含这些
    queue.put((handle_bytes, list(t_original.shape), t_original.dtype, device_id))

    # 5. 等待消费者完成
    p.join()

    # 6. 验证修改结果
    print(f"[Producer] Checking tensor after consumer modification...")
    print(f"[Producer] Tensor sample: {t_original[0, :5]}")

    # 如果 Consumer 成功执行了 add_(100)，这里的数值应该变大了
    # 我们简单检查一下第一个元素是否大概增加了 100
    # 注意：原始是 randn，均值 0，+100 后应该在 100 附近
    if t_original[0, 0] > 90:
        print("\nSUCCESS: IPC shared memory modification verified!")
    else:
        print("\nFAILURE: Data did not change.")


if __name__ == "__main__":
    import os

    run_test()
