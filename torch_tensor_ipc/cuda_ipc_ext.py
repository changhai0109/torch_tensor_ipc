import os
import torch
from torch.utils.cpp_extension import load

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_FILE = os.path.join(_CURRENT_DIR, "_cuda_ipc_ext.cpp")

CUDA_HOME = os.getenv("CUDA_HOME", "/usr/local/cuda")

_ext = load(
    name="cuda_ipc_ext_module",
    sources=[_SOURCE_FILE],
    extra_cflags=["-O3", f"-I{os.path.join(CUDA_HOME, 'include')}"],
    extra_cuda_cflags=["-O3", f"-I{os.path.join(CUDA_HOME, 'include')}"],
    extra_ldflags=[f"-L{os.path.join(CUDA_HOME, 'lib64')}", "-lcuda"],
    verbose=True,
)


def export_tensor_ipc(tensor: torch.Tensor):
    """
    导出 Tensor 的 IPC 句柄。
    Returns:
        (handle_bytes, nbytes)
    """
    return _ext.export_tensor_ipc(tensor)


def tensor_from_ipc_bytes(
    handle_bytes: bytes, shape: list, dtype: torch.dtype, device_index: int = 0
):
    """
    从 IPC 句柄重建 Tensor。
    """
    return _ext.tensor_from_ipc_bytes(handle_bytes, shape, dtype, device_index)
