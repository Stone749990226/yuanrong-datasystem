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
#include "datasystem/common/flags/flags.h"
#include "datasystem/common/util/format.h"
#include "datasystem/common/util/status_helper.h"

namespace datasystem {
namespace {
constexpr size_t DEFAULT_GPU_PIPELINE_DEPTH = 4;
constexpr size_t MIN_GPU_PIPELINE_DEPTH = 2;
constexpr size_t MAX_GPU_PIPELINE_DEPTH = 8;
}

CudaResourceManager::CudaResourceManager()
{
    // CUDA does not support FFTS/HUGE_FFTS (NPU-specific), force DIRECT policy
    policyD2H = MemcopyPolicy::DIRECT;
    policyH2D = MemcopyPolicy::DIRECT;
    devManager_ = DeviceManagerFactory::GetDeviceManager();
    enableH2DPipeline_ = GetBoolFromEnv("DS_GPU_H2D_ENABLE_PIPELINE", true);
    uint64_t pipelineDepth = DEFAULT_GPU_PIPELINE_DEPTH;
    LOG_IF_ERROR(GetNumberFromEnv("DS_GPU_H2D_PIPELINE_DEPTH", pipelineDepth), "GetNumberFromEnv failed");
    h2dPipelineDepth_ =
        std::clamp(static_cast<size_t>(pipelineDepth), MIN_GPU_PIPELINE_DEPTH, MAX_GPU_PIPELINE_DEPTH);
    enableD2HPipeline_ = GetBoolFromEnv("DS_GPU_D2H_ENABLE_PIPELINE", true);
    pipelineDepth = DEFAULT_GPU_PIPELINE_DEPTH;
    LOG_IF_ERROR(GetNumberFromEnv("DS_GPU_D2H_PIPELINE_DEPTH", pipelineDepth), "GetNumberFromEnv failed");
    d2hPipelineDepth_ =
        std::clamp(static_cast<size_t>(pipelineDepth), MIN_GPU_PIPELINE_DEPTH, MAX_GPU_PIPELINE_DEPTH);
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
    for (size_t i = 0; i < devBlobList.size(); i++) {
        CHECK_FAIL_RETURN_STATUS(devBlobList[i].deviceIdx == deviceId, K_INVALID,
                                 FormatString("Device index mismatch in batch: expect %d, actual %d, index %zu",
                                              deviceId, devBlobList[i].deviceIdx, i));
    }

    DeviceBatchCopyHelper helper;
    RETURN_IF_NOT_OK(helper.Prepare(devBlobList, bufferList, copyKind));
    RETURN_OK_IF_TRUE(helper.batchSize == 0);

    if (copyKind == MemcpyKind::HOST_TO_DEVICE) {
        return SubmitBatchH2D(helper, deviceId);
    }
    return SubmitBatchD2H(helper, deviceId);
}

Status CudaResourceManager::CudaMemcpyBatchSync(DeviceBatchCopyHelper &helper, MemcpyKind copyKind)
{
    Status lastErr = Status::OK();
    for (size_t index = 0; index < helper.batchSize; index++) {
        auto *dst = helper.dstList[index];
        auto *src = helper.srcList[index];
        auto size = helper.dataSizeList[index];
        Status rc;
        if (copyKind == MemcpyKind::DEVICE_TO_HOST) {
            rc = devManager_->MemCopyD2H(dst, size, src, size);
        } else {
            rc = devManager_->MemCopyH2D(dst, size, src, size);
        }
        if (rc.IsError()) {
            lastErr = rc;
            LOG(ERROR) << FormatString("CudaMemCopy failed for flat index %zu: %s", index, rc.ToString().c_str());
        }
    }
    return lastErr.IsError() ? lastErr : Status::OK();
}

Status CudaResourceManager::SubmitBatchH2D(DeviceBatchCopyHelper &helper, int32_t deviceId)
{
    if (!EnableH2DPipeline()) {
        return SubmitBatchDirectAsync(helper, deviceId, MemcpyKind::HOST_TO_DEVICE);
    }

    auto initRc = EnsureInitialized();
    if (initRc.IsError()) {
        LOG(WARNING) << "GPU H2D pipeline init failed, fallback to direct async: " << initRc;
        return SubmitBatchDirectAsync(helper, deviceId, MemcpyKind::HOST_TO_DEVICE);
    }

    GpuH2DPipelineState state;
    auto rc = InitH2DPipelineState(deviceId, state);
    if (rc.IsError()) {
        LOG(WARNING) << "GPU H2D pipeline state init failed, fallback to direct async: " << rc;
        return SubmitBatchDirectAsync(helper, deviceId, MemcpyKind::HOST_TO_DEVICE);
    }

    Status lastRc = Status::OK();
    for (size_t objectIndex = 0; objectIndex < helper.bufferMetas.size(); objectIndex++) {
        const auto &meta = helper.bufferMetas[objectIndex];
        if (meta.blobCount <= 1) {
            rc = SubmitObjectDirectAsync(helper, objectIndex, state.directStream, MemcpyKind::HOST_TO_DEVICE);
        } else {
            bool shouldFallbackDirect = false;
            rc = SubmitObjectWithH2DPipeline(helper, objectIndex, state, shouldFallbackDirect);
            if (rc.IsError() && shouldFallbackDirect) {
                LOG(WARNING) << FormatString("GPU H2D pipeline fallback to direct async for object %zu: %s",
                                             objectIndex, rc.ToString().c_str());
                rc = SubmitObjectDirectAsync(helper, objectIndex, state.directStream, MemcpyKind::HOST_TO_DEVICE);
            }
        }
        if (rc.IsError()) {
            lastRc = rc;
            break;
        }
    }

    auto tailRc = TailWaitH2D(state);
    CleanupH2DPipelineState(state);
    return lastRc.IsError() ? lastRc : tailRc;
}

Status CudaResourceManager::SubmitBatchD2H(DeviceBatchCopyHelper &helper, int32_t deviceId)
{
    if (!EnableD2HPipeline()) {
        return SubmitBatchDirectAsync(helper, deviceId, MemcpyKind::DEVICE_TO_HOST);
    }

    auto initRc = EnsureInitialized();
    if (initRc.IsError()) {
        LOG(WARNING) << "GPU D2H pipeline init failed, fallback to direct async: " << initRc;
        return SubmitBatchDirectAsync(helper, deviceId, MemcpyKind::DEVICE_TO_HOST);
    }

    GpuD2HPipelineState state;
    auto rc = InitD2HPipelineState(deviceId, state);
    if (rc.IsError()) {
        LOG(WARNING) << "GPU D2H pipeline state init failed, fallback to direct async: " << rc;
        return SubmitBatchDirectAsync(helper, deviceId, MemcpyKind::DEVICE_TO_HOST);
    }

    Status lastRc = Status::OK();
    for (size_t objectIndex = 0; objectIndex < helper.bufferMetas.size(); objectIndex++) {
        const auto &meta = helper.bufferMetas[objectIndex];
        if (meta.blobCount <= 1) {
            rc = SubmitObjectDirectAsync(helper, objectIndex, state.directStream, MemcpyKind::DEVICE_TO_HOST);
        } else {
            bool shouldFallbackDirect = false;
            rc = SubmitObjectWithD2HPipeline(helper, objectIndex, state, shouldFallbackDirect);
            if (rc.IsError() && shouldFallbackDirect) {
                LOG(WARNING) << FormatString("GPU D2H pipeline fallback to direct async for object %zu: %s",
                                             objectIndex, rc.ToString().c_str());
                rc = SubmitObjectDirectAsync(helper, objectIndex, state.directStream, MemcpyKind::DEVICE_TO_HOST);
            }
        }
        if (rc.IsError()) {
            lastRc = rc;
            break;
        }
    }

    auto tailRc = TailWaitD2H(helper, state);
    CleanupD2HPipelineState(state);
    return lastRc.IsError() ? lastRc : tailRc;
}

Status CudaResourceManager::SubmitBatchDirectAsync(DeviceBatchCopyHelper &helper, int32_t deviceId,
                                                   MemcpyKind copyKind)
{
    (void)deviceId;
    void *stream = nullptr;
    RETURN_IF_NOT_OK(devManager_->CreateStream(&stream));

    Status lastRc = Status::OK();
    for (size_t objectIndex = 0; objectIndex < helper.bufferMetas.size(); objectIndex++) {
        auto rc = SubmitObjectDirectAsync(helper, objectIndex, stream, copyKind);
        if (rc.IsError()) {
            lastRc = rc;
            break;
        }
    }

    auto syncRc = devManager_->SynchronizeStream(stream);
    LOG_IF_ERROR(devManager_->DestroyStream(stream), "Destroy direct stream failed.");
    return lastRc.IsError() ? lastRc : syncRc;
}

Status CudaResourceManager::SubmitObjectDirectAsync(const DeviceBatchCopyHelper &helper, size_t objectIndex,
                                                    void *stream, MemcpyKind copyKind)
{
    CHECK_FAIL_RETURN_STATUS(objectIndex < helper.bufferMetas.size(), K_INVALID,
                             FormatString("Invalid object index %zu", objectIndex));
    auto &meta = helper.bufferMetas[objectIndex];
    if (copyKind == MemcpyKind::HOST_TO_DEVICE) {
        auto &srcBuffer = helper.srcBuffers[objectIndex];
        size_t offset = 0;
        for (size_t blobIndex = 0; blobIndex < meta.blobCount; blobIndex++) {
            size_t flatBlobIndex = meta.firstBlobOffset + blobIndex;
            CHECK_FAIL_RETURN_STATUS(
                flatBlobIndex < helper.dstBuffers.size(), K_RUNTIME_ERROR,
                FormatString("Invalid flat blob index %zu for object %zu", flatBlobIndex, objectIndex));
            auto &dstBuffer = helper.dstBuffers[flatBlobIndex];
            auto rc = devManager_->MemcpyAsync(static_cast<void *>(dstBuffer.ptr), dstBuffer.size,
                                               static_cast<void *>(static_cast<uint8_t *>(srcBuffer.ptr) + offset),
                                               dstBuffer.size, copyKind, stream);
            if (rc.IsError()) {
                LOG(ERROR) << FormatString("DirectAsync H2D submit failed for object %zu, blob %zu: %s", objectIndex,
                                           blobIndex, rc.ToString().c_str());
                return rc;
            }
            offset += dstBuffer.size;
        }
        return Status::OK();
    }

    CHECK_FAIL_RETURN_STATUS(copyKind == MemcpyKind::DEVICE_TO_HOST, K_INVALID, "Invalid memcpy kind");
    CHECK_FAIL_RETURN_STATUS(objectIndex < helper.dstBuffers.size(), K_RUNTIME_ERROR,
                             FormatString("Invalid destination object index %zu", objectIndex));
    auto &dstBuffer = helper.dstBuffers[objectIndex];
    size_t offset = 0;
    for (size_t blobIndex = 0; blobIndex < meta.blobCount; blobIndex++) {
        size_t flatBlobIndex = meta.firstBlobOffset + blobIndex;
        CHECK_FAIL_RETURN_STATUS(
            flatBlobIndex < helper.srcBuffers.size(), K_RUNTIME_ERROR,
            FormatString("Invalid flat blob index %zu for object %zu", flatBlobIndex, objectIndex));
        auto &srcBuffer = helper.srcBuffers[flatBlobIndex];
        CHECK_FAIL_RETURN_STATUS(offset + srcBuffer.size <= dstBuffer.size, K_RUNTIME_ERROR,
                                 FormatString("DirectAsync D2H offset overflow for object %zu, blob %zu", objectIndex,
                                              blobIndex));
        auto *dstPtr = static_cast<void *>(static_cast<uint8_t *>(dstBuffer.ptr) + offset);
        auto rc = devManager_->MemcpyAsync(dstPtr, srcBuffer.size, srcBuffer.ptr, srcBuffer.size, copyKind, stream);
        if (rc.IsError()) {
            LOG(ERROR) << FormatString("DirectAsync D2H submit failed for object %zu, blob %zu: %s", objectIndex,
                                       blobIndex, rc.ToString().c_str());
            return rc;
        }
        offset += srcBuffer.size;
    }
    return Status::OK();
}

Status CudaResourceManager::InitPinnedPipelineResources(const std::vector<GpuPinnedSlotBase *> &slots,
                                                        void **directStream)
{
    RETURN_IF_NOT_OK(devManager_->CreateStream(directStream));
    for (auto *slot : slots) {
        CHECK_FAIL_RETURN_STATUS(slot != nullptr, K_INVALID, "Null GPU pinned slot");
        auto rc = devManager_->CreateStream(&slot->stream);
        if (rc.IsError()) {
            CleanupPinnedPipelineResources(slots, *directStream);
            return rc;
        }
        rc = devManager_->CreateEvent(&slot->doneEvent);
        if (rc.IsError()) {
            CleanupPinnedPipelineResources(slots, *directStream);
            return rc;
        }
    }
    return Status::OK();
}

void CudaResourceManager::CleanupPinnedPipelineResources(const std::vector<GpuPinnedSlotBase *> &slots,
                                                         void *directStream)
{
    if (directStream != nullptr) {
        LOG_IF_ERROR(devManager_->SynchronizeStream(directStream), "Synchronize direct stream failed.");
        LOG_IF_ERROR(devManager_->DestroyStream(directStream), "Destroy direct stream failed.");
    }
    for (auto *slot : slots) {
        if (slot == nullptr) {
            continue;
        }
        if (slot->doneEvent != nullptr) {
            LOG_IF_ERROR(devManager_->DestroyEvent(slot->doneEvent), "Destroy slot event failed.");
            slot->doneEvent = nullptr;
        }
        if (slot->stream != nullptr) {
            LOG_IF_ERROR(devManager_->DestroyStream(slot->stream), "Destroy slot stream failed.");
            slot->stream = nullptr;
        }
        if (Host() != nullptr) {
            LOG_IF_ERROR(Host()->Free(slot->hostPinnedPool), "Free slot host pinned buffer failed.");
        }
        slot->capacity = 0;
        slot->lastObjectIndex = 0;
    }
}

Status CudaResourceManager::EnsurePinnedSlotCapacity(GpuPinnedSlotBase &slot, uint64_t objectSize,
                                                     bool &shouldFallbackDirect)
{
    shouldFallbackDirect = false;
    if (objectSize <= slot.capacity) {
        return Status::OK();
    }
    if (Host() == nullptr) {
        shouldFallbackDirect = true;
        return Status(K_RUNTIME_ERROR, "HostMemMgr is not initialized for GPU pinned pipeline");
    }
    if (slot.capacity > 0) {
        auto freeRc = Host()->Free(slot.hostPinnedPool);
        if (freeRc.IsError()) {
            return freeRc;
        }
        slot.capacity = 0;
    }

    std::vector<BufferMetaInfo> bufferMetas{
        BufferMetaInfo{ .blobCount = 1, .firstBlobOffset = 0, .size = objectSize }
    };
    auto allocRc = Host()->Allocate(bufferMetas, slot.hostPinnedPool, true);
    if (allocRc.IsError()) {
        shouldFallbackDirect = true;
        return allocRc;
    }
    slot.capacity = slot.hostPinnedPool[0].size;
    return Status::OK();
}

Status CudaResourceManager::InitH2DPipelineState(int32_t deviceId, GpuH2DPipelineState &state)
{
    state.deviceId = deviceId;
    state.slots.resize(GetH2DPipelineDepth());
    std::vector<GpuPinnedSlotBase *> slotBases;
    slotBases.reserve(state.slots.size());
    for (auto &slot : state.slots) {
        slotBases.emplace_back(&slot);
    }
    return InitPinnedPipelineResources(slotBases, &state.directStream);
}

void CudaResourceManager::CleanupH2DPipelineState(GpuH2DPipelineState &state)
{
    for (auto &slot : state.slots) {
        if (slot.inFlight && slot.doneEvent != nullptr) {
            LOG_IF_ERROR(devManager_->SynchronizeEvent(slot.doneEvent), "Synchronize H2D slot event failed.");
            slot.inFlight = false;
        }
    }
    std::vector<GpuPinnedSlotBase *> slotBases;
    slotBases.reserve(state.slots.size());
    for (auto &slot : state.slots) {
        slotBases.emplace_back(&slot);
    }
    CleanupPinnedPipelineResources(slotBases, state.directStream);
    state.directStream = nullptr;
    state.slots.clear();
}

Status CudaResourceManager::TailWaitH2D(GpuH2DPipelineState &state)
{
    RETURN_IF_NOT_OK(devManager_->SynchronizeStream(state.directStream));
    for (auto &slot : state.slots) {
        if (!slot.inFlight) {
            continue;
        }
        RETURN_IF_NOT_OK(devManager_->SynchronizeEvent(slot.doneEvent));
        slot.inFlight = false;
    }
    return Status::OK();
}

Status CudaResourceManager::AcquireH2DSlot(GpuH2DPipelineState &state, GpuH2DSlot *&slot)
{
    CHECK_FAIL_RETURN_STATUS(!state.slots.empty(), K_RUNTIME_ERROR, "GPU H2D pipeline slots are empty.");
    for (size_t attempt = 0; attempt < state.slots.size(); attempt++) {
        size_t index = (state.nextVictim + attempt) % state.slots.size();
        auto &candidate = state.slots[index];
        if (!candidate.inFlight) {
            state.nextVictim = (index + 1) % state.slots.size();
            slot = &candidate;
            return Status::OK();
        }
        if (devManager_->QueryEventStatus(candidate.doneEvent).IsOk()) {
            candidate.inFlight = false;
            state.nextVictim = (index + 1) % state.slots.size();
            slot = &candidate;
            return Status::OK();
        }
    }

    auto &candidate = state.slots[state.nextVictim];
    RETURN_IF_NOT_OK(devManager_->SynchronizeEvent(candidate.doneEvent));
    candidate.inFlight = false;
    slot = &candidate;
    state.nextVictim = (state.nextVictim + 1) % state.slots.size();
    return Status::OK();
}

Status CudaResourceManager::SubmitObjectWithH2DPipeline(const DeviceBatchCopyHelper &helper, size_t objectIndex,
                                                        GpuH2DPipelineState &state, bool &shouldFallbackDirect)
{
    shouldFallbackDirect = false;
    CHECK_FAIL_RETURN_STATUS(objectIndex < helper.bufferMetas.size(), K_INVALID,
                             FormatString("Invalid object index %zu", objectIndex));
    GpuH2DSlot *slot = nullptr;
    RETURN_IF_NOT_OK(AcquireH2DSlot(state, slot));

    auto &meta = helper.bufferMetas[objectIndex];
    auto &srcBuffer = helper.srcBuffers[objectIndex];
    auto rc = EnsurePinnedSlotCapacity(*slot, meta.size, shouldFallbackDirect);
    if (rc.IsError()) {
        return rc;
    }

    auto *hostPinnedPtr = slot->hostPinnedPool[0].pointer;
    RETURN_IF_NOT_OK(Host()->HostMemoryCopy(hostPinnedPtr, meta.size, srcBuffer.ptr, srcBuffer.size));

    size_t offset = 0;
    for (size_t blobIndex = 0; blobIndex < meta.blobCount; blobIndex++) {
        size_t flatBlobIndex = meta.firstBlobOffset + blobIndex;
        CHECK_FAIL_RETURN_STATUS(flatBlobIndex < helper.dstBuffers.size(), K_RUNTIME_ERROR,
                                 FormatString("Invalid flat blob index %zu for object %zu", flatBlobIndex, objectIndex));
        auto &dstBuffer = helper.dstBuffers[flatBlobIndex];
        rc = devManager_->MemcpyAsync(dstBuffer.ptr, dstBuffer.size,
                                      static_cast<void *>(static_cast<uint8_t *>(hostPinnedPtr) + offset),
                                      dstBuffer.size, MemcpyKind::HOST_TO_DEVICE, slot->stream);
        if (rc.IsError()) {
            LOG(ERROR) << FormatString("Pipeline H2D submit failed for object %zu, blob %zu: %s", objectIndex,
                                       blobIndex, rc.ToString().c_str());
            return rc;
        }
        offset += dstBuffer.size;
    }

    RETURN_IF_NOT_OK(devManager_->RecordEvent(slot->doneEvent, slot->stream));
    slot->inFlight = true;
    slot->lastObjectIndex = objectIndex;
    return Status::OK();
}

Status CudaResourceManager::InitD2HPipelineState(int32_t deviceId, GpuD2HPipelineState &state)
{
    state.deviceId = deviceId;
    state.slots.resize(GetD2HPipelineDepth());
    std::vector<GpuPinnedSlotBase *> slotBases;
    slotBases.reserve(state.slots.size());
    for (auto &slot : state.slots) {
        slotBases.emplace_back(&slot);
    }
    return InitPinnedPipelineResources(slotBases, &state.directStream);
}

void CudaResourceManager::CleanupD2HPipelineState(GpuD2HPipelineState &state)
{
    for (auto &slot : state.slots) {
        if (slot.phase == GpuD2HSlotPhase::IN_FLIGHT_D2H && slot.doneEvent != nullptr) {
            LOG_IF_ERROR(devManager_->SynchronizeEvent(slot.doneEvent), "Synchronize D2H slot event failed.");
        }
        slot.phase = GpuD2HSlotPhase::AVAILABLE;
    }
    std::vector<GpuPinnedSlotBase *> slotBases;
    slotBases.reserve(state.slots.size());
    for (auto &slot : state.slots) {
        slotBases.emplace_back(&slot);
    }
    CleanupPinnedPipelineResources(slotBases, state.directStream);
    state.directStream = nullptr;
    state.slots.clear();
}

Status CudaResourceManager::DrainD2HSlotToHost(const DeviceBatchCopyHelper &helper, GpuD2HSlot &slot)
{
    CHECK_FAIL_RETURN_STATUS(Host() != nullptr, K_RUNTIME_ERROR,
                             "HostMemMgr is not initialized for GPU D2H pipeline");
    CHECK_FAIL_RETURN_STATUS(slot.lastObjectIndex < helper.bufferMetas.size(), K_RUNTIME_ERROR,
                             FormatString("Invalid D2H slot object index %zu", slot.lastObjectIndex));
    CHECK_FAIL_RETURN_STATUS(slot.lastObjectIndex < helper.dstBuffers.size(), K_RUNTIME_ERROR,
                             FormatString("Invalid D2H destination object index %zu", slot.lastObjectIndex));

    const auto &meta = helper.bufferMetas[slot.lastObjectIndex];
    const auto &dstBuffer = helper.dstBuffers[slot.lastObjectIndex];
    auto *hostPinnedPtr = slot.hostPinnedPool[0].pointer;
    CHECK_FAIL_RETURN_STATUS(hostPinnedPtr != nullptr, K_RUNTIME_ERROR, "D2H pinned host buffer is null");
    CHECK_FAIL_RETURN_STATUS(meta.size <= slot.capacity, K_RUNTIME_ERROR,
                             FormatString("D2H slot capacity %llu too small for object size %zu",
                                          static_cast<unsigned long long>(slot.capacity), meta.size));
    CHECK_FAIL_RETURN_STATUS(meta.size <= dstBuffer.size, K_RUNTIME_ERROR,
                             FormatString("D2H destination size %zu too small for object size %zu", dstBuffer.size,
                                          meta.size));
    RETURN_IF_NOT_OK(Host()->HostMemoryCopy(dstBuffer.ptr, dstBuffer.size, hostPinnedPtr, meta.size));
    return Status::OK();
}

Status CudaResourceManager::AcquireD2HSlot(const DeviceBatchCopyHelper &helper, GpuD2HPipelineState &state,
                                           GpuD2HSlot *&slot)
{
    CHECK_FAIL_RETURN_STATUS(!state.slots.empty(), K_RUNTIME_ERROR, "GPU D2H pipeline slots are empty.");
    for (size_t attempt = 0; attempt < state.slots.size(); attempt++) {
        size_t index = (state.nextVictim + attempt) % state.slots.size();
        auto &candidate = state.slots[index];
        if (candidate.phase == GpuD2HSlotPhase::AVAILABLE) {
            state.nextVictim = (index + 1) % state.slots.size();
            slot = &candidate;
            return Status::OK();
        }
        if (candidate.phase == GpuD2HSlotPhase::IN_FLIGHT_D2H
            && devManager_->QueryEventStatus(candidate.doneEvent).IsOk()) {
            candidate.phase = GpuD2HSlotPhase::READY_FOR_DRAIN;
        }
        if (candidate.phase == GpuD2HSlotPhase::READY_FOR_DRAIN) {
            RETURN_IF_NOT_OK(DrainD2HSlotToHost(helper, candidate));
            candidate.phase = GpuD2HSlotPhase::AVAILABLE;
            state.nextVictim = (index + 1) % state.slots.size();
            slot = &candidate;
            return Status::OK();
        }
    }

    auto &candidate = state.slots[state.nextVictim];
    if (candidate.phase == GpuD2HSlotPhase::IN_FLIGHT_D2H) {
        RETURN_IF_NOT_OK(devManager_->SynchronizeEvent(candidate.doneEvent));
        candidate.phase = GpuD2HSlotPhase::READY_FOR_DRAIN;
    }
    if (candidate.phase == GpuD2HSlotPhase::READY_FOR_DRAIN) {
        RETURN_IF_NOT_OK(DrainD2HSlotToHost(helper, candidate));
        candidate.phase = GpuD2HSlotPhase::AVAILABLE;
    }
    slot = &candidate;
    state.nextVictim = (state.nextVictim + 1) % state.slots.size();
    return Status::OK();
}

Status CudaResourceManager::SubmitObjectWithD2HPipeline(const DeviceBatchCopyHelper &helper, size_t objectIndex,
                                                        GpuD2HPipelineState &state, bool &shouldFallbackDirect)
{
    shouldFallbackDirect = false;
    CHECK_FAIL_RETURN_STATUS(objectIndex < helper.bufferMetas.size(), K_INVALID,
                             FormatString("Invalid object index %zu", objectIndex));
    GpuD2HSlot *slot = nullptr;
    RETURN_IF_NOT_OK(AcquireD2HSlot(helper, state, slot));

    const auto &meta = helper.bufferMetas[objectIndex];
    auto rc = EnsurePinnedSlotCapacity(*slot, meta.size, shouldFallbackDirect);
    if (rc.IsError()) {
        return rc;
    }

    auto *hostPinnedPtr = slot->hostPinnedPool[0].pointer;
    CHECK_FAIL_RETURN_STATUS(hostPinnedPtr != nullptr, K_RUNTIME_ERROR, "D2H pinned host buffer is null");

    size_t offset = 0;
    for (size_t blobIndex = 0; blobIndex < meta.blobCount; blobIndex++) {
        size_t flatBlobIndex = meta.firstBlobOffset + blobIndex;
        CHECK_FAIL_RETURN_STATUS(
            flatBlobIndex < helper.srcBuffers.size(), K_RUNTIME_ERROR,
            FormatString("Invalid flat blob index %zu for object %zu", flatBlobIndex, objectIndex));
        const auto &srcBuffer = helper.srcBuffers[flatBlobIndex];
        CHECK_FAIL_RETURN_STATUS(offset + srcBuffer.size <= slot->capacity, K_RUNTIME_ERROR,
                                 FormatString("Pipeline D2H offset overflow for object %zu, blob %zu", objectIndex,
                                              blobIndex));
        auto *dstPtr = static_cast<void *>(static_cast<uint8_t *>(hostPinnedPtr) + offset);
        rc = devManager_->MemcpyAsync(dstPtr, slot->capacity - offset, srcBuffer.ptr, srcBuffer.size,
                                      MemcpyKind::DEVICE_TO_HOST, slot->stream);
        if (rc.IsError()) {
            LOG(ERROR) << FormatString("Pipeline D2H submit failed for object %zu, blob %zu: %s", objectIndex,
                                       blobIndex, rc.ToString().c_str());
            return rc;
        }
        offset += srcBuffer.size;
    }

    RETURN_IF_NOT_OK(devManager_->RecordEvent(slot->doneEvent, slot->stream));
    slot->phase = GpuD2HSlotPhase::IN_FLIGHT_D2H;
    slot->lastObjectIndex = objectIndex;
    return Status::OK();
}

Status CudaResourceManager::TailWaitD2H(const DeviceBatchCopyHelper &helper, GpuD2HPipelineState &state)
{
    RETURN_IF_NOT_OK(devManager_->SynchronizeStream(state.directStream));
    for (auto &slot : state.slots) {
        if (slot.phase == GpuD2HSlotPhase::IN_FLIGHT_D2H) {
            RETURN_IF_NOT_OK(devManager_->SynchronizeEvent(slot.doneEvent));
            slot.phase = GpuD2HSlotPhase::READY_FOR_DRAIN;
        }
        if (slot.phase == GpuD2HSlotPhase::READY_FOR_DRAIN) {
            RETURN_IF_NOT_OK(DrainD2HSlotToHost(helper, slot));
            slot.phase = GpuD2HSlotPhase::AVAILABLE;
        }
    }
    return Status::OK();
}

bool CudaResourceManager::EnableH2DPipeline() const
{
    return enableH2DPipeline_;
}

size_t CudaResourceManager::GetH2DPipelineDepth() const
{
    return h2dPipelineDepth_;
}

bool CudaResourceManager::EnableD2HPipeline() const
{
    return enableD2HPipeline_;
}

size_t CudaResourceManager::GetD2HPipelineDepth() const
{
    return d2hPipelineDepth_;
}

}  // namespace datasystem
