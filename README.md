# torch_tensor_ipc
A small utility to perform zero-copy PyTorch tensor transfer on CUDA between processes on the same machine and same GPU device using CUDA IPC handles.

## Key Idea:
`torch.multiprocessing.Queue` has the ability to transfer `Tensors` between processes, however, only limited to those are created through `spawn` or `fork`. In this tool, we extend this into sharing the same CUDA Tensors in a zero-copy manner across completely non-related processes, as long as they are accessing to the same devices.

## Example use
Check `tests/test_server_clinet.py`


