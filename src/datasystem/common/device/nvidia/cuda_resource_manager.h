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

#include <cstddef>
#include <vector>

#include "datasystem/common/device/device_batch_copy_helper.h"
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
    Status CudaMemcpyBatch(const std::vector<DeviceBlobList> &devBlobList, std::vector<Buffer *> &bufferList,
                           MemcpyKind copyKind);
    Status CudaMemcpyBatchSync(DeviceBatchCopyHelper &helper, MemcpyKind copyKind);
    Status SubmitBatchH2D(DeviceBatchCopyHelper &helper, int32_t deviceId);
    Status SubmitBatchD2H(DeviceBatchCopyHelper &helper, int32_t deviceId);
    Status SubmitBatchDirectAsync(DeviceBatchCopyHelper &helper, int32_t deviceId, MemcpyKind copyKind);
    Status SubmitObjectDirectAsync(const DeviceBatchCopyHelper &helper, size_t objectIndex, void *stream,
                                   MemcpyKind copyKind);

    struct GpuPinnedSlotBase {
        GpuPinnedSlotBase() : hostPinnedPool(1)
        {
        }

        std::vector<ShmUnit> hostPinnedPool;
        void *stream = nullptr;
        void *doneEvent = nullptr;
        uint64_t capacity = 0;
        size_t lastObjectIndex = 0;
    };

    struct GpuPinnedPipelineStateBase {
        int32_t deviceId = -1;
        void *directStream = nullptr;
        size_t nextVictim = 0;
    };

    struct GpuH2DSlot : GpuPinnedSlotBase {
        bool inFlight = false;
    };

    struct GpuH2DPipelineState : GpuPinnedPipelineStateBase {
        std::vector<GpuH2DSlot> slots;
    };

    enum class GpuD2HSlotPhase {
        AVAILABLE,
        IN_FLIGHT_D2H,
        READY_FOR_DRAIN,
    };

    struct GpuD2HSlot : GpuPinnedSlotBase {
        GpuD2HSlotPhase phase = GpuD2HSlotPhase::AVAILABLE;
    };

    struct GpuD2HPipelineState : GpuPinnedPipelineStateBase {
        std::vector<GpuD2HSlot> slots;
    };

    Status InitPinnedPipelineResources(const std::vector<GpuPinnedSlotBase *> &slots, void **directStream);
    void CleanupPinnedPipelineResources(const std::vector<GpuPinnedSlotBase *> &slots, void *directStream);
    Status EnsurePinnedSlotCapacity(GpuPinnedSlotBase &slot, uint64_t objectSize, bool &shouldFallbackDirect);

    Status InitH2DPipelineState(int32_t deviceId, GpuH2DPipelineState &state);
    void CleanupH2DPipelineState(GpuH2DPipelineState &state);
    Status TailWaitH2D(GpuH2DPipelineState &state);
    Status AcquireH2DSlot(GpuH2DPipelineState &state, GpuH2DSlot *&slot);
    Status SubmitObjectWithH2DPipeline(const DeviceBatchCopyHelper &helper, size_t objectIndex,
                                       GpuH2DPipelineState &state, bool &shouldFallbackDirect);

    Status InitD2HPipelineState(int32_t deviceId, GpuD2HPipelineState &state);
    void CleanupD2HPipelineState(GpuD2HPipelineState &state);
    Status TailWaitD2H(const DeviceBatchCopyHelper &helper, GpuD2HPipelineState &state);
    Status AcquireD2HSlot(const DeviceBatchCopyHelper &helper, GpuD2HPipelineState &state, GpuD2HSlot *&slot);
    Status DrainD2HSlotToHost(const DeviceBatchCopyHelper &helper, GpuD2HSlot &slot);
    Status SubmitObjectWithD2HPipeline(const DeviceBatchCopyHelper &helper, size_t objectIndex,
                                       GpuD2HPipelineState &state, bool &shouldFallbackDirect);
    bool EnableH2DPipeline() const;
    size_t GetH2DPipelineDepth() const;
    bool EnableD2HPipeline() const;
    size_t GetD2HPipelineDepth() const;

    DeviceManagerBase *devManager_ = nullptr;
    bool enableH2DPipeline_ = true;
    size_t h2dPipelineDepth_ = 4;
    bool enableD2HPipeline_ = true;
    size_t d2hPipelineDepth_ = 4;
};
}  // namespace datasystem

#endif  // DATASYSTEM_COMMON_DEVICE_NVIDIA_CUDA_RESOURCE_MANAGER_H
