/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Description: CUDA resource manager for GPU support.
 */
#ifndef DATASYSTEM_COMMON_DEVICE_NVIDIA_CUDA_RESOURCE_MANAGER_H
#define DATASYSTEM_COMMON_DEVICE_NVIDIA_CUDA_RESOURCE_MANAGER_H

#include "datasystem/common/device/device_resource_manager.h"

namespace datasystem {

class CudaResourceManager : public DeviceResourceManager {
public:
    CudaResourceManager();
    ~CudaResourceManager() = default;

    Status MemcpyBatchD2H(const std::vector<DeviceBlobList> &devBlobList, std::vector<Buffer *> &bufferList) override;
    Status MemcpyBatchH2D(const std::vector<DeviceBlobList> &devBlobList, std::vector<Buffer *> &bufferList) override;
    void SetPolicyByHugeTlb(bool enableHugeTlb) override;

private:
    /**
     * @brief Batch memory copy between host and device using CUDA streams.
     * @param[in] devBlobList List of device blob lists (blobs on the same GPU card).
     * @param[in,out] bufferList List of host-side buffers.
     * @param[in] copyKind The direction of the copy (HOST_TO_DEVICE or DEVICE_TO_HOST).
     * @return Status of the batch copy operation.
     */
    Status CudaMemcpyBatch(const std::vector<DeviceBlobList> &devBlobList, std::vector<Buffer *> &bufferList,
                           MemcpyKind copyKind);

    DeviceManagerBase *devManager_ = nullptr;
};
}  // namespace datasystem

#endif  // DATASYSTEM_COMMON_DEVICE_NVIDIA_CUDA_RESOURCE_MANAGER_H
