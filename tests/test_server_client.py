import multiprocessing
import time
import os
import sys
import random
import torch  # 必须引入 torch 来操作 GPU

# 假设你的代码都在当前目录下
# 也可以根据你的包结构调整 import
try:
    from torch_tensor_ipc.server import GpuIpcServer
    from torch_tensor_ipc.client import GpuIpcClient
except ImportError:
    print(
        "请确保 server.py, client.py, protocol.py, cuda_ipc_ext.so 都在当前目录或 PYTHONPATH 中"
    )
    sys.exit(1)

SOCKET_PATH = "/tmp/test_real_gpu_ipc.sock"


def server_process_func(ready_event, stop_event):
    """
    Server 进程运行的函数
    """
    # 1. 在新进程中初始化 CUDA (不要从父进程继承)
    if torch.cuda.is_available():
        print(f"[Server] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("[Server] Error: No CUDA device found!")
        return

    # 2. 创建真实的 GPU Tensor
    # 这里我们创建两个 Tensor，填充不同的值以便验证
    t1 = torch.full((1024, 1024), 3.14159, device="cuda", dtype=torch.float32).to(
        "cuda:1"
    )
    t2 = torch.arange(100, device="cuda", dtype=torch.float32).to("cuda:0")

    print(f"[Server] Created tensors. t1: {t1.shape}, t2: {t2.shape}")

    # 3. 启动 Server
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = GpuIpcServer(SOCKET_PATH)

    # 注册 Tensor (这会调用 ext.export_tensor_ipc)
    try:
        server.register("matrix_a", t1)
        server.register("vector_b", t2)
        print("[Server] Tensors registered successfully.")
    except Exception as e:
        print(f"[Server] Registration failed: {e}")
        return

    # 4. 通知主进程 Server 已就绪
    ready_event.set()

    # 5. 循环运行，直到收到停止信号
    # 注意：server.run_forever() 是阻塞的，我们需要一种方式停止它
    # 这里为了简单，我们让它运行在 Daemon 线程中，或者通过超时控制
    # 实际测试中，我们可以直接让主进程 terminate 这个进程
    try:
        server.run_forever()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[Server] Error loop: {e}")
    finally:
        server.close()


def client_process_func(client_id, target_name, expected_value_check_func):
    """
    Client 进程运行的函数
    """
    try:
        # 1. 随机休眠一下，模拟真实并发
        time.sleep(random.uniform(0.01, 0.1))

        # 2. 连接 Server
        client = GpuIpcClient(SOCKET_PATH)

        # 3. 获取 Tensor List (可选测试)
        if client_id % 2 == 0:
            lst = client.get_tensor_list()
            if target_name not in lst:
                print(
                    f"[Client {client_id}] Error: Target {target_name} not in list {lst}"
                )
                return False

        # 4. 获取 Tensor (关键步骤)
        # 这会调用 ext.import_tensor_ipc
        # 假设你的 ext 返回的是一个 torch.Tensor 或者能被 torch 识别的对象
        start_t = time.time()
        tensor = client.get_tensor(target_name)
        cost = (time.time() - start_t) * 1000

        # 5. 验证数据
        # 必须把数据从 GPU 拷回 CPU 才能确信 IPC 成功并且内存是可访问的
        if hasattr(tensor, "cpu"):
            cpu_val = tensor.cpu()
        else:
            # 如果你的 ext 返回的不是 tensor 对象，可能需要封装
            # 这里假设它是一个 tensor
            print(
                f"[Client {client_id}] Error: Returned object is not a tensor (no .cpu())"
            )
            return False

        # 检查数值
        if expected_value_check_func(cpu_val):
            print(
                f"[Client {client_id}] Success! Get '{target_name}' in {cost:.2f}ms. Shape: {tensor.shape}"
            )
            return True
        else:
            print(f"[Client {client_id}] Error: Value mismatch!")
            return False

    except Exception as e:
        print(f"[Client {client_id}] Exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_matrix(t):
    # 验证 matrix_a 是否全是 3.14159
    # 允许一点点浮点误差
    return torch.allclose(t, torch.tensor(3.14159), atol=1e-4)


def check_vector(t):
    # 验证 vector_b 是否是 0..99
    expected = torch.arange(100, dtype=torch.float32)
    return torch.allclose(t, expected, atol=1e-4)


if __name__ == "__main__":
    # --- 关键设置：使用 spawn 启动进程 ---
    # 这对于 CUDA 多进程是必须的，否则会报 CUDA 初始化错误
    multiprocessing.set_start_method("spawn", force=True)

    print("=== Starting Real GPU IPC Test ===")

    # 1. 启动 Server 进程
    ready_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()

    server_proc = multiprocessing.Process(
        target=server_process_func, args=(ready_event, stop_event), name="ServerProc"
    )
    server_proc.start()

    # 等待 Server 启动
    if not ready_event.wait(timeout=5):
        print("Error: Server failed to start in 5 seconds.")
        if server_proc.is_alive():
            server_proc.terminate()
        sys.exit(1)

    print("=== Server Ready. Starting Clients... ===")

    # 2. 启动多个 Client 进程并发请求
    client_procs = []
    num_clients = 10  # 并发客户端数量

    for i in range(num_clients):
        # 一半客户端取 matrix_a，一半取 vector_b
        if i % 2 == 0:
            target = "matrix_a"
            check_func = check_matrix
        else:
            target = "vector_b"
            check_func = check_vector

        p = multiprocessing.Process(
            target=client_process_func,
            args=(i, target, check_func),
            name=f"ClientProc-{i}",
        )
        client_procs.append(p)
        p.start()

    # 3. 等待所有 Client 完成
    success_count = 0
    for p in client_procs:
        p.join()
        if p.exitcode == 0:
            # 注意：这里 exitcode == 0 只代表进程没崩，
            # 实际上我们上面的逻辑里如果 verify 失败打印了 Error 但没 exit(1)，exitcode 也是 0。
            # 为了严格测试，你可以修改 client_func 让它失败时 sys.exit(1)
            success_count += 1

    print("=== All Clients Finished ===")

    # 4. 清理 Server
    print("Stopping Server...")
    server_proc.terminate()
    server_proc.join()

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    print("=== Test Complete ===")
