

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * Description: Defines the ascend memory copy pool.
 */
#include "datasystem/common/device/ascend/acl_memcpy_pool.h"
#include "datasystem/common/device/resource_mgr.h"
#include "datasystem/common/perf/perf_manager.h"
namespace datasystem {

AsyncAclMemCopyPool::AsyncAclMemCopyPool(ResourceMgr *resourceMgr) : resourceMgr_(resourceMgr)
{
    const size_t pipeLineNums = 2;
    copyPool_ = std::make_unique<ThreadPool>(1);
    fftsCopyPool_ = std::make_unique<ThreadPool>(1);
    const int h2hThreadCount = 2;
    h2hCopyPool_ = std::make_unique<ThreadPool>(h2hThreadCount);
    devInterImpl_ = acl::AclDeviceManager::Instance();
    for (size_t i = 0; i < pipeLineNums; i++) {
        aclrtStream copyStream = nullptr;
        copyStreams_.emplace_back(copyStream);
    }
}

Status AsyncAclMemCopyPool::MemcpyBatchD2H(uint32_t deviceId, DeviceBatchCopyHelper &helper, MemcopyPolicy policy)
{
    PerfPoint point(PerfKey::CLIENT_D2H_MEMCPY_INIT);
    if (policy == MemcopyPolicy::FFTS || policy == MemcopyPolicy::HUGE_FFTS) {
        if (deviceNow_ != static_cast<int32_t>(deviceId)) {
            RETURN_IF_NOT_OK_PRINT_ERROR_MSG(devInterImpl_->SetDevice(deviceId), "Failed to init.");
        }
        RETURN_IF_NOT_OK(resourceMgr_->Init());
        point.RecordAndReset(PerfKey::CLIENT_D2H_MEMCPY_RUN);
        bool skipH2HMemcpy = resourceMgr_->GetD2HPolicy() == resourceMgr_->GetH2DPolicy()
                             && resourceMgr_->GetD2HPolicy() == MemcopyPolicy::HUGE_FFTS;
        CHECK_FAIL_RETURN_STATUS(
            deviceId < MAX_DEVICE_COUNT, K_INVALID,
            FormatString("Invalid device id %zu, exceed max device id %zu", deviceId, MAX_DEVICE_COUNT));
        AclDeviceResource &deviceResource = *resourceMgr_->DeviceResources()[deviceId];
        FftsPipelineD2HCopier copier(deviceId, deviceResource, resourceMgr_->Host(), resourceMgr_->Device(),
                                     helper.bufferMetas, h2hCopyPool_.get(), fftsCopyPool_.get(), skipH2HMemcpy);
        return copier.ExecuteMemcpy(helper.dstBuffers, helper.srcBuffers);
    } else {
        PerfPoint point(PerfKey::TOTAL_D2H_BATCH_MEMCPY);
        return AclMemcpyBatch(deviceId, helper, MemcpyKind::DEVICE_TO_HOST);
    }
}

Status AsyncAclMemCopyPool::MemcpyBatchH2D(uint32_t deviceId, DeviceBatchCopyHelper &helper, MemcopyPolicy policy)
{
    PerfPoint point(PerfKey::CLIENT_H2D_MEMCPY_INIT);
    if (policy == MemcopyPolicy::FFTS || policy == MemcopyPolicy::HUGE_FFTS) {
        if (deviceNow_ != static_cast<int32_t>(deviceId)) {
            RETURN_IF_NOT_OK_PRINT_ERROR_MSG(devInterImpl_->SetDevice(deviceId), "Failed to init.");
        }
        RETURN_IF_NOT_OK(resourceMgr_->Init());
        point.RecordAndReset(PerfKey::CLIENT_H2D_MEMCPY_RUN);
        bool skipH2HMemcpy = resourceMgr_->GetD2HPolicy() == resourceMgr_->GetH2DPolicy()
                             && resourceMgr_->GetH2DPolicy() == MemcopyPolicy::HUGE_FFTS;
        CHECK_FAIL_RETURN_STATUS(
            deviceId < MAX_DEVICE_COUNT, K_INVALID,
            FormatString("Invalid device id %zu, exceed max device id %zu", deviceId, MAX_DEVICE_COUNT));
        AclDeviceResource &deviceResource = *resourceMgr_->DeviceResources()[deviceId];
        FftsPipelineH2DCopier copier(deviceId, deviceResource, resourceMgr_->Host(), resourceMgr_->Device(),
                                     helper.bufferMetas, h2hCopyPool_.get(), fftsCopyPool_.get(), skipH2HMemcpy);
        return copier.ExecuteMemcpy(helper.dstBuffers, helper.srcBuffers);
    } else {
        PerfPoint point(PerfKey::TOTAL_H2D_BATCH_MEMCPY);
        return AclMemcpyBatch(deviceId, helper, MemcpyKind::HOST_TO_DEVICE);
    }
}

Status AsyncAclMemCopyPool::AclMemcpyBatch(uint32_t deviceId, DeviceBatchCopyHelper &helper, MemcpyKind copyKind)
{
    size_t leftNum = helper.dataSizeList.size();
    size_t startIndex = 0;
    while (leftNum > 0) {
        auto maxBatchSize = 4096UL;
        auto batchNum = std::min(leftNum, maxBatchSize);
        size_t failedIdx = 0;
        auto res =
            devInterImpl_->MemcpyBatch(helper.dstList.data() + startIndex, helper.dataSizeList.data() + startIndex,
                                       helper.srcList.data() + startIndex, helper.dataSizeList.data() + startIndex,
                                       batchNum, copyKind, deviceId, &failedIdx);
        if (res.IsError()) {
            LOG(ERROR) << FormatString("AclMemcpyBatch return error , failed index:%lu", failedIdx) << "," << res;
            return res;
        }
        leftNum -= batchNum;
        startIndex += batchNum;
    }
    return Status::OK();
}

AsyncAclMemCopyPool::~AsyncAclMemCopyPool()
{
    for (auto &stream : copyStreams_) {
        if (stream != nullptr) {
            LOG_IF_ERROR(devInterImpl_->DestroyStream(stream), "Destory stream failed.");
        }
    }
}
}  // namespace datasystem
