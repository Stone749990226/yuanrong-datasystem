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

#include "datasystem/common/device/nvidia/cuda_resource_manager.h"

#include "datasystem/common/device/device_manager_factory.h"
#include "datasystem/common/util/format.h"
#include "datasystem/common/util/status_helper.h"

namespace datasystem {

CudaResourceManager::CudaResourceManager()
{
    // CUDA does not support FFTS/HUGE_FFTS (NPU-specific), force DIRECT policy
    policyD2H = MemcopyPolicy::DIRECT;
    policyH2D = MemcopyPolicy::DIRECT;
    devManager_ = DeviceManagerFactory::GetDeviceManager();
}

Status CudaResourceManager::MemcpyBatchD2H(const std::vector<DeviceBlobList> &devBlobList,
                                           std::vector<Buffer *> &bufferList)
{
    return CudaMemcpyBatch(devBlobList, bufferList, MemcpyKind::DEVICE_TO_HOST);
}

Status CudaResourceManager::MemcpyBatchH2D(const std::vector<DeviceBlobList> &devBlobList,
                                           std::vector<Buffer *> &bufferList)
{
    return CudaMemcpyBatch(devBlobList, bufferList, MemcpyKind::HOST_TO_DEVICE);
}

void CudaResourceManager::SetPolicyByHugeTlb(bool enableHugeTlb)
{
    // CUDA does not support FFTS/HUGE_FFTS, no policy change needed
    (void)enableHugeTlb;
}

Status CudaResourceManager::CudaMemcpyBatch(const std::vector<DeviceBlobList> &devBlobList,
                                            std::vector<Buffer *> &bufferList, MemcpyKind copyKind)
{
    CHECK_FAIL_RETURN_STATUS(!devBlobList.empty(), K_INVALID, "The devBlobList is empty.");
    CHECK_FAIL_RETURN_STATUS(!bufferList.empty(), K_INVALID, "The bufferList is empty.");
    CHECK_FAIL_RETURN_STATUS(devBlobList.size() == bufferList.size(), K_INVALID,
                             FormatString("The devBlobList size %zu is not equal to bufferList size %zu",
                                          devBlobList.size(), bufferList.size()));
    CHECK_FAIL_RETURN_STATUS(devManager_ != nullptr, K_RUNTIME_ERROR, "Failed to get device manager.");

    auto deviceId = devBlobList[0].deviceIdx;
    RETURN_IF_NOT_OK(devManager_->SetDevice(deviceId));

    Status lastErr = Status::OK();
    for (size_t i = 0; i < devBlobList.size(); i++) {
        CHECK_FAIL_RETURN_STATUS(devBlobList[i].deviceIdx == deviceId, K_INVALID,
                                 FormatString("Device index mismatch in batch: expect %d, actual %d, index %zu",
                                              deviceId, devBlobList[i].deviceIdx, i));
        if (bufferList[i] == nullptr) {
            continue;
        }
        auto &blobs = devBlobList[i].blobs;
        auto *buffer = bufferList[i];
        auto *hostRawPointer = reinterpret_cast<uint8_t *>(buffer->MutableData());
        auto *offsetArrPtr = reinterpret_cast<uint64_t *>(buffer->MutableData());
        auto sz = *offsetArrPtr;
        auto *offsets = offsetArrPtr + 1;

        CHECK_FAIL_RETURN_STATUS(
            sz == blobs.size() && sz > 0, K_INVALID,
            FormatString("Blobs count mismatch in devBlobList between sender and receiver, sender count is: %ld, "
                         "receiver count is: %ld, mismatch devBlobList index: %zu",
                         sz, blobs.size(), i));

        for (size_t j = 0; j < blobs.size(); j++) {
            auto hostDataSize = offsets[j + 1] - offsets[j];
            auto *devicePointer = blobs[j].pointer;
            auto deviceDataSize = blobs[j].size;
            auto *hostPointer = hostRawPointer + offsets[j];

            CHECK_FAIL_RETURN_STATUS(static_cast<size_t>(hostDataSize) == deviceDataSize, K_RUNTIME_ERROR,
                                     FormatString("Data size mismatch: host size %lu, device size %lu, key index %zu, "
                                                  "blob index %zu",
                                                  hostDataSize, deviceDataSize, i, j));

            Status rc;
            if (copyKind == MemcpyKind::DEVICE_TO_HOST) {
                rc = devManager_->MemCopyD2H(hostPointer, hostDataSize, devicePointer, hostDataSize);
            } else {
                rc = devManager_->MemCopyH2D(devicePointer, hostDataSize, hostPointer, hostDataSize);
            }
            if (rc.IsError()) {
                lastErr = rc;
                LOG(ERROR) << FormatString("CudaMemCopy failed for key %zu, blob %zu: %s", i, j, rc.ToString().c_str());
            }
        }
    }

    return lastErr.IsError() ? lastErr : Status::OK();
}

}  // namespace datasystem
