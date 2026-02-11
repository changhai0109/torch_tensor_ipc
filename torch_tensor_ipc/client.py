# client.py
import socket
import json
import struct
from . import cuda_ipc_ext as ext
from . import protocol as proto


class GpuIpcClient:
    def __init__(self, path: str, device_index: int = 0):
        self.path = path
        self.device_index = device_index

    def get_tensor(self, name: str):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.path)

        try:
            req = json.dumps({"cmd": "GET", "name": name}).encode("utf-8")
            header = proto.pack_packet_header(proto.MSG_TYPE_JSON, len(req))
            sock.sendall(header + req)

            head_data = self._recvn(sock, proto.PACKET_SIZE)
            msg_type, payload_len = proto.unpack_packet_header(head_data)

            payload = self._recvn(sock, payload_len)

            if msg_type == proto.MSG_TYPE_TENSOR:
                return self._parse_tensor_payload(payload)

            elif msg_type == proto.MSG_TYPE_ERROR or msg_type == proto.MSG_TYPE_JSON:
                err_msg = json.loads(payload)
                raise RuntimeError(f"Server error: {err_msg}")

            else:
                raise RuntimeError(f"Unknown message type: {msg_type}")

        finally:
            sock.close()

    def get_tensor_list(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.path)

        try:
            req = json.dumps({"cmd": "LIST"}).encode("utf-8")
            header = proto.pack_packet_header(proto.MSG_TYPE_JSON, len(req))
            sock.sendall(header + req)

            head_data = self._recvn(sock, proto.PACKET_SIZE)
            msg_type, payload_len = proto.unpack_packet_header(head_data)

            payload = self._recvn(sock, payload_len)

            if msg_type == proto.MSG_TYPE_JSON:
                resp = json.loads(payload)
                return resp.get("tensors", [])

            elif msg_type == proto.MSG_TYPE_ERROR:
                err_msg = json.loads(payload)
                raise RuntimeError(f"Server error: {err_msg}")

            else:
                raise RuntimeError(f"Unknown message type: {msg_type}")

        finally:
            sock.close()

    def _parse_tensor_payload(self, payload: bytes):
        meta_bytes = payload[: proto.TENSOR_HEADER_SIZE]
        handle_bytes = payload[proto.TENSOR_HEADER_SIZE :]

        if len(handle_bytes) != proto.HANDLE_SIZE:
            raise ValueError("Invalid handle size in payload")

        meta = proto.unpack_tensor_meta(meta_bytes)

        return ext.tensor_from_ipc_bytes(
            handle_bytes, meta["shape"], meta["dtype"], meta["device_uuid"]
        )

    def _recvn(self, sock, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            buf.extend(chunk)
        return bytes(buf)
