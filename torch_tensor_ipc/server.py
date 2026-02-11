import os
import socket
import json
import struct
import sys
import threading
from . import cuda_ipc_ext as ext
from . import protocol as proto


class GpuIpcConnection:
    def __init__(self, conn: socket.socket):
        self.conn = conn

    def recv_packet(self):
        header_data = self._recvn(proto.PACKET_SIZE)
        msg_type, payload_len = proto.unpack_packet_header(header_data)

        payload = self._recvn(payload_len)
        return msg_type, payload

    def send_json(self, obj):
        payload = json.dumps(obj).encode("utf-8")
        header = proto.pack_packet_header(proto.MSG_TYPE_JSON, len(payload))
        self.conn.sendall(header + payload)

    def send_error(self, msg):
        payload = json.dumps({"error": msg}).encode("utf-8")
        header = proto.pack_packet_header(proto.MSG_TYPE_ERROR, len(payload))
        self.conn.sendall(header + payload)

    def send_tensor(self, tensor):
        handle_bytes, device_uuid, nbytes = ext.export_tensor_ipc(tensor)
        if len(handle_bytes) != proto.HANDLE_SIZE:
            raise ValueError("Handle size mismatch")

        meta_bytes = proto.pack_tensor_meta(tensor, nbytes, device_uuid)
        payload = meta_bytes + handle_bytes

        header = proto.pack_packet_header(proto.MSG_TYPE_TENSOR, len(payload))
        self.conn.sendall(header + payload)

    def _recvn(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed")
            buf.extend(chunk)
        return bytes(buf)

    def close(self):
        self.conn.close()


class GpuIpcServer:
    def __init__(self, path: str):
        self.path = path
        self.registry = {}  # {name: tensor}

        if os.path.exists(path):
            os.unlink(path)

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.sock.listen(5)
        print(f"[Server] Listening on {path}")

    def register(self, name: str, tensor):
        self.registry[name] = tensor
        print(f"[Server] Registered '{name}' shape={tensor.shape}")

    def run_forever(self):
        try:
            while True:
                conn_sock, _ = self.sock.accept()
                client_thread = threading.Thread(
                    target=self._handle_client, args=(conn_sock,)
                )
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            self.close()

    def _handle_client(self, conn_sock):
        conn = GpuIpcConnection(conn_sock)
        try:
            msg_type, payload = conn.recv_packet()

            if msg_type != proto.MSG_TYPE_JSON:
                conn.send_error("Expected JSON command")
                return

            cmd_data = json.loads(payload)
            cmd = cmd_data.get("cmd")
            name = cmd_data.get("name")

            if cmd == "GET":
                if name in self.registry:
                    conn.send_tensor(self.registry[name])
                else:
                    conn.send_error(f"Tensor '{name}' not found")
            elif cmd == "LIST":
                conn.send_json({"tensors": list(self.registry.keys())})
            else:
                conn.send_error(f"Unknown command: {cmd}")

        except Exception as e:
            print(f"[Server] Error: {e}")
        finally:
            conn.close()

    def close(self):
        self.sock.close()
        if os.path.exists(self.path):
            os.unlink(self.path)


if __name__ == "__main__":
    import torch
    import time
    import random
    import multiprocessing

    SOCKET_PATH = "/tmp/gpu_ipc_socket"
    ready_event = multiprocessing.Event()

    # 1. 创建 Tensor (在 Server 进程中创建，确保它们在 GPU 上)
    t1 = torch.full((1024, 1024), 3.14159, device="cuda", dtype=torch.float32).to(
        "cuda:1"
    )
    t2 = torch.arange(100, device="cuda", dtype=torch.float32).to("cuda:0")

    # 2. 启动 Server 进程
    def server_process_func(ready_event):
        server = GpuIpcServer(SOCKET_PATH)
        server.register("matrix_a", t1)
        server.register("vector_b", t2)
        ready_event.set()  # 通知主进程 Server 已就绪
        server.run_forever()

    server_process = multiprocessing.Process(
        target=server_process_func, args=(ready_event,)
    )
    server_process.start()

    # 等待 Server 启动
    if not ready_event.wait(timeout=5):
        print("Error: Server failed to start in 5 seconds.")
        if server_process.is_alive():
            server_process.terminate()
        sys.exit(1)
    print("Server is ready. You can run the client now.")
