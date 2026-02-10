import struct
import json
import torch

MSG_TYPE_JSON = 1
MSG_TYPE_TENSOR = 2
MSG_TYPE_ERROR = 3

PACKET_FMT = "!B3sI"
PACKET_SIZE = struct.calcsize(PACKET_FMT)

MAGIC = b"TORCH_TENSOR_IPC"
VERSION = 1
MAX_DIMS = 8
HANDLE_SIZE = 64

DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float: 1,
    torch.float64: 2,
    torch.double: 3,
    torch.float16: 4,
    torch.bfloat16: 5,
    torch.float8_e4m3fn: 6,
    torch.float8_e4m3fnuz: 7,
    torch.float8_e5m2: 8,
    torch.float8_e5m2fnuz: 9,
    torch.half: 10,
    torch.uint8: 11,
    torch.uint16: 12,
    torch.uint32: 13,
    torch.uint64: 14,
    torch.int8: 15,
    torch.int16: 16,
    torch.short: 17,
    torch.int32: 18,
    torch.int: 19,
    torch.int64: 20,
    torch.long: 21,
    torch.complex32: 22,
    torch.complex64: 23,
    torch.chalf: 24,
    torch.cfloat: 25,
    torch.complex128: 26,
    torch.cdouble: 27,
    torch.quint8: 28,
    torch.qint8: 29,
    torch.qint32: 30,
    torch.bool: 31,
    torch.quint4x2: 32,
    torch.quint2x4: 33,
    torch.bits1x8: 34,
    torch.bits2x4: 35,
    torch.bits4x2: 36,
    torch.bits8: 37,
    torch.bits16: 38,
}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}

TENSOR_HEADER_FMT = "!10sBBBBQ" + "Q" * MAX_DIMS
TENSOR_HEADER_SIZE = struct.calcsize(TENSOR_HEADER_FMT)


def pack_packet_header(msg_type: int, payload_len: int) -> bytes:
    return struct.pack(PACKET_FMT, msg_type, b"\x00\x00\x00", payload_len)


def unpack_packet_header(header_bytes: bytes):
    msg_type, _, payload_len = struct.unpack(PACKET_FMT, header_bytes)
    return msg_type, payload_len


def pack_tensor_meta(tensor, nbytes):
    shape = list(tensor.shape)
    ndim = len(shape)
    if ndim > MAX_DIMS:
        raise ValueError(f"ndim {ndim} > MAX_DIMS {MAX_DIMS}")

    dtype_code = DTYPE_TO_CODE.get(tensor.dtype, 255)  # 255 unknown

    shape_padded = shape + [0] * (MAX_DIMS - ndim)

    meta = struct.pack(
        TENSOR_HEADER_FMT,
        MAGIC,
        VERSION,
        dtype_code,
        ndim,
        0,  # flags
        nbytes,
        *shape_padded,
    )
    return meta


def unpack_tensor_meta(meta_bytes):
    unpacked = struct.unpack(TENSOR_HEADER_FMT, meta_bytes)
    magic = unpacked[0]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic}")

    dtype_code = unpacked[2]
    ndim = unpacked[3]
    nbytes = unpacked[5]
    shape = unpacked[6 : 6 + ndim]

    dtype = CODE_TO_DTYPE.get(dtype_code, torch.float32)

    return {"dtype": dtype, "shape": list(shape), "nbytes": nbytes}
