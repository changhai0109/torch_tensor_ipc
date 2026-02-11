import os
import torch
import importlib
from torch.utils.cpp_extension import load

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_FILE = os.path.join(_CURRENT_DIR, "_cuda_ipc_ext.cpp")

CUDA_HOME = os.getenv("CUDA_HOME", "/usr/local/cuda")

EXT_NAME = "torch_tensor_ipc._cuda_ipc_ext"


def _load_extension():
    try:
        return importlib.import_module(EXT_NAME)
    except ImportError:
        pass

    return load(
        name=EXT_NAME,
        sources=[_SOURCE_FILE],
        extra_cflags=["-O3", f"-I{os.path.join(CUDA_HOME, 'include')}"],
        extra_cuda_cflags=["-O3", f"-I{os.path.join(CUDA_HOME, 'include')}"],
        extra_ldflags=[f"-L{os.path.join(CUDA_HOME, 'lib64')}", "-lcuda"],
        verbose=True,
    )


_ext = _load_extension()


def export_tensor_ipc(tensor: torch.Tensor):
    """
    Returns:
        (handle_bytes, device_uuid, nbytes):
    """
    ret = _ext.export_tensor_ipc(tensor)
    return ret


def tensor_from_ipc_bytes(handle_bytes, shape, dtype, device_uuid):
    return _ext.tensor_from_ipc_bytes(handle_bytes, shape, dtype, device_uuid)
