import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

CUDA_HOME = os.getenv("CUDA_HOME", "/usr/local/cuda")

ext_modules = [
    CUDAExtension(
        name="torch_tensor_ipc.cuda_ipc_ext_module",  # Python import path
        sources=[
            os.path.join("torch_tensor_ipc", "_cuda_ipc_ext.cpp"),
        ],
        extra_compile_args={
            "cxx": ["-O3", f"-I{os.path.join(CUDA_HOME, 'include')}"],
            "nvcc": ["-O3", f"-I{os.path.join(CUDA_HOME, 'include')}"],
        },
        libraries=["cuda"],
        library_dirs=[os.path.join(CUDA_HOME, "lib64")],
        # optional: extra_link_args=[]
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
