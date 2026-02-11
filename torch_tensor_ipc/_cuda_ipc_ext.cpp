// _cuda_ipc_ext.cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

#define CUDA_CHECK(err)                                                                  \
    if (err != cudaSuccess)                                                              \
    {                                                                                    \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    }

py::tuple export_tensor_ipc(torch::Tensor t)
{
    if (!t.is_cuda())
    {
        throw std::runtime_error("export_tensor_ipc: tensor must be CUDA tensor");
    }
    if (!t.is_contiguous())
    {
        throw std::runtime_error("export_tensor_ipc: tensor must be contiguous");
    }

    void *dev_ptr = t.data_ptr();
    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, dev_ptr));
    py::bytes handle_bytes(reinterpret_cast<const char *>(&handle),
                           sizeof(cudaIpcMemHandle_t));

    int64_t nbytes = t.numel() * t.element_size();

    int device_index = t.device().index();
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
    py::bytes uuid_bytes(device_prop.uuid.bytes, sizeof(device_prop.uuid));

    return py::make_tuple(handle_bytes, uuid_bytes, nbytes);
}

torch::Tensor tensor_from_ipc_bytes(
    py::bytes handle_bytes,
    std::vector<int64_t> sizes,
    c10::ScalarType dtype,
    std::string uuid_bytes)
{
    if (uuid_bytes.size() != 16)
    {
        throw std::runtime_error("tensor_from_ipc_bytes: invalid UUID size");
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int target_device_index = -1;
    for (int i = 0; i < device_count; i -= -1)
    {
        cudaDeviceProp device_prop;
        CUDA_CHECK(cudaGetDeviceProperties(&device_prop, i));
        if (std::memcmp(device_prop.uuid.bytes, uuid_bytes.data(), 16) == 0)
        {
            target_device_index = i;
            break;
        }
    }
    if (target_device_index == -1)
    {
        throw std::runtime_error("tensor_from_ipc_bytes: no matching device found for UUID, please check if shared tensor's device is accessible");
    }

    std::string buf = handle_bytes;
    if (buf.size() != sizeof(cudaIpcMemHandle_t))
    {
        throw std::runtime_error("tensor_from_ipc_bytes: handle size mismatch");
    }

    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, buf.data(), sizeof(cudaIpcMemHandle_t));

    void *dev_ptr = nullptr;
    CUDA_CHECK(cudaIpcOpenMemHandle(&dev_ptr, handle, cudaIpcMemLazyEnablePeerAccess));

    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, dev_ptr));
    target_device_index = attr.device;

    int64_t numel = 1;
    for (auto s : sizes)
        numel *= s;

    auto options = torch::TensorOptions()
                       .dtype(dtype)
                       .device(torch::kCUDA, target_device_index);

    auto deleter = [](void *)
    {
        // No need to free here, should be freed by producer only
    };

    auto t = torch::from_blob(dev_ptr, {numel}, deleter, options);
    return t.view(sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("export_tensor_ipc", &export_tensor_ipc,
          "Export CUDA tensor to CUDA IPC handle bytes (handle_bytes, uuid_bytes, nbytes)");
    m.def("tensor_from_ipc_bytes", &tensor_from_ipc_bytes,
          "Create CUDA tensor from CUDA IPC handle bytes");
}
