# server.py
import os
import socket
import json
import struct
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
        handle_bytes, nbytes = ext.export_tensor_ipc(tensor)
        if len(handle_bytes) != proto.HANDLE_SIZE:
            raise ValueError("Handle size mismatch")

        meta_bytes = proto.pack_tensor_meta(tensor, nbytes)
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
                self._handle_client(conn_sock)
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
