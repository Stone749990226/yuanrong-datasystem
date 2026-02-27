# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hetero client python interface test (GPU / CUDA version).
"""

from __future__ import absolute_import

import argparse
import ctypes
import ctypes.util

from yr.datasystem.hetero_client import (
    HeteroClient,
    Blob,
    DeviceBlobList,
)

# ---- CUDA Runtime helpers via ctypes ----
_cudart_path = ctypes.util.find_library("cudart")
if _cudart_path is None:
    raise RuntimeError(
        "libcudart.so not found. Please install CUDA Toolkit and ensure "
        "LD_LIBRARY_PATH includes the CUDA lib directory (e.g. /usr/local/cuda/lib64)."
    )
_cudart = ctypes.CDLL(_cudart_path)


def _check_cuda(err, func_name):
    """Check CUDA return code; raise on failure."""
    if err != 0:
        raise RuntimeError(f"{func_name} failed with error code {err}")


def cuda_set_device(device_id):
    _check_cuda(_cudart.cudaSetDevice(device_id), "cudaSetDevice")


def cuda_malloc(size):
    ptr = ctypes.c_void_p()
    _check_cuda(_cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size)), "cudaMalloc")
    return ptr


def cuda_free(ptr):
    _check_cuda(_cudart.cudaFree(ptr), "cudaFree")


def cuda_memcpy(dst, src, size, kind):
    """kind: 1 = H2D, 2 = D2H, 3 = D2D"""
    _check_cuda(_cudart.cudaMemcpy(dst, src, ctypes.c_size_t(size), ctypes.c_int(kind)), "cudaMemcpy")


class HeteroClientExampleGpu():
    """This class shows the SDK usage example of the HeteroClient on GPU."""

    def __init__(self):
        parser = argparse.ArgumentParser(description="Hetero client (GPU) python interface Test")
        parser.add_argument("--host", required=True, help="The IP of worker service")
        parser.add_argument("--port", required=True, type=int, help="The port of worker service")
        parser.add_argument("--device_id", type=int, default=0, help="The GPU device id")
        args = parser.parse_args()
        self._host = args.host
        self._port = args.port
        self._device_id = args.device_id

    def dev_mset_and_dev_mget_example(self):
        """test dev_mset and dev_mget on GPU device memory"""
        cuda_set_device(self._device_id)

        client = HeteroClient(self._host, self._port)
        client.init()

        key = "gpu_key"
        value = bytes("val", encoding='utf8')
        size = len(value)

        # ---- dev_mset: copy host data to device, then store ----
        in_dev_ptr = cuda_malloc(size)
        host_buf = ctypes.create_string_buffer(value)
        cuda_memcpy(in_dev_ptr, host_buf, size, 1)  # H2D

        in_blob = Blob(in_dev_ptr.value, size)
        in_blob_list = [DeviceBlobList(self._device_id, [in_blob])]
        failed_keys = client.dev_mset([key], in_blob_list)
        if failed_keys:
            raise RuntimeError(f"dev_mset failed, failed keys: {failed_keys}")

        # ---- dev_mget: read back into device memory ----
        out_dev_ptr = cuda_malloc(size)
        out_blob = Blob(out_dev_ptr.value, size)
        out_blob_list = [DeviceBlobList(self._device_id, [out_blob])]
        sub_timeout_ms = 30_000
        failed_keys = client.dev_mget([key], out_blob_list, sub_timeout_ms)
        if failed_keys:
            raise RuntimeError(f"dev_mget failed, failed keys: {failed_keys}")

        # ---- verify: copy device data back to host and compare ----
        result_buf = ctypes.create_string_buffer(size)
        cuda_memcpy(result_buf, out_dev_ptr, size, 2)  # D2H
        if result_buf.raw != value:
            raise RuntimeError(
                f"Data verification failed! Expected {value!r}, but got {result_buf.raw!r}"
            )

        cuda_free(in_dev_ptr)
        cuda_free(out_dev_ptr)
        print("HeteroClient (GPU) dev_mset_and_dev_mget_example: PASS")


if __name__ == '__main__':
    example = HeteroClientExampleGpu()
    example.dev_mset_and_dev_mget_example()
