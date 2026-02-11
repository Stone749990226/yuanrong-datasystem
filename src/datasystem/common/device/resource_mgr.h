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
 * Description: Data system AclMemMgr.
 */

#ifndef DATASYSTEM_CLIENT_OBJECT_CACHE_DEVICE_RESOURCE_MGR_H
#define DATASYSTEM_CLIENT_OBJECT_CACHE_DEVICE_RESOURCE_MGR_H

#include <memory>
#include <vector>

#include "datasystem/common/shared_memory/allocator.h"
#include "datasystem/common/shared_memory/arena_group_key.h"
#include "datasystem/common/shared_memory/shm_unit.h"
#include "datasystem/common/util/wait_post.h"
#include "datasystem/common/device/device_helper.h"
#include "datasystem/utils/status.h"
#include "datasystem/object/buffer.h"
#include "datasystem/hetero/device_common.h"
// ifdef USE_NPU
#include "datasystem/common/device/ascend/cann_types.h"
#include "datasystem/common/device/ascend/acl_device_resource.h"
#include "datasystem/common/device/ascend/acl_memcpy_pool.h"
using DeviceResourceList = datasystem::AclDeviceResourceList;
using AsyncMemCopyPool = datasystem::AsyncAclMemCopyPool;
namespace datasystem {

struct DeviceBatchCopyHelper {
    bool is64BitAligned(void *ptr)
    {
        constexpr uintptr_t alignmentMask = 0x7;
        uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
        return (address & alignmentMask) == 0;
    }

    Status Prepare(const std::vector<DeviceBlobList> &devBlobList, std::vector<Buffer *> &bufferList,
                   MemcpyKind copyKind)
    {
        std::vector<void *> hostPointerList;
        std::vector<void *> devPointerList;
        std::vector<BufferView> hostBuffers;
        std::vector<BufferView> deviceBuffers;
        hostBuffers.reserve(devBlobList.size());
        deviceBuffers.reserve(devBlobList.size());
        CHECK_FAIL_RETURN_STATUS(!devBlobList.empty(), K_INVALID, "The devBlobList is empty.");
        CHECK_FAIL_RETURN_STATUS(!bufferList.empty(), K_INVALID, "The bufferList is empty.");
        size_t keyStartInBlobs = 0;
        for (size_t i = 0; i < devBlobList.size(); i++) {
            auto &blobs = devBlobList[i].blobs;
            if (bufferList[i] == nullptr) {
                continue;
            }
            auto &buffer = bufferList[i];
            auto offsetArrPtr = reinterpret_cast<uint64_t *>(buffer->MutableData());
            auto hostRawPointer = reinterpret_cast<uint8_t *>(buffer->MutableData());
            auto sz = *offsetArrPtr;
            auto offsets = offsetArrPtr + 1;
            CHECK_FAIL_RETURN_STATUS(
                sz == blobs.size() && sz > 0, K_INVALID,
                FormatString("Blobs count mismatch in devBlobList between sender and receiver, sender count is: %ld, "
                             "receiver count is: %ld, mismatch devBlobList index: %zu, mismatch key index: %zu",
                             sz, blobs.size(), i, i));
            size_t dataSize = buffer->GetSize() - offsets[0];
            bufferMetas.emplace_back(
                BufferMetaInfo{ .blobCount = blobs.size(), .firstBlobOffset = keyStartInBlobs, .size = dataSize });
            hostBuffers.emplace_back(BufferView{ .ptr = hostRawPointer + offsets[0], .size = dataSize });
            for (size_t j = 0; j < blobs.size(); j++) {
                auto hostDataSize = offsets[j + 1] - offsets[j];
                auto devicePointer = blobs[j].pointer;
                auto deviceDataSize = blobs[j].size;
                auto hostPointer = hostRawPointer + offsets[j];
                if (!is64BitAligned(hostPointer)) {
                    LOG(WARNING) << "host memory is not 64 aligned: " << hostRawPointer;
                }
                if (!is64BitAligned(devicePointer)) {
                    LOG(WARNING) << "deivce memory is not 64 aligned: " << devicePointer;
                }
                CHECK_FAIL_RETURN_STATUS(static_cast<size_t>(hostDataSize) == deviceDataSize, K_RUNTIME_ERROR,
                                         "The data size of device and host is not equal.");
                deviceBuffers.emplace_back(BufferView{ .ptr = devicePointer, .size = hostDataSize });
                hostPointerList.emplace_back(hostPointer);
                devPointerList.emplace_back(devicePointer);
                dataSizeList.emplace_back(hostDataSize);
            }
            keyStartInBlobs += blobs.size();
        }
        if (copyKind == MemcpyKind::HOST_TO_DEVICE) {
            srcBuffers = std::move(hostBuffers);
            dstBuffers = std::move(deviceBuffers);

            srcList = std::move(hostPointerList);
            dstList = std::move(devPointerList);
        } else if (copyKind == MemcpyKind::DEVICE_TO_HOST) {
            srcBuffers = std::move(deviceBuffers);
            dstBuffers = std::move(hostBuffers);

            srcList = std::move(devPointerList);
            dstList = std::move(hostPointerList);
        } else {
            RETURN_STATUS(K_INVALID, "Invalid MemcpyKind");
        }
        return Status::OK();
    }
    std::vector<size_t> dataSizeList;
    std::vector<void *> srcList;
    std::vector<void *> dstList;

    std::vector<BufferView> srcBuffers;
    std::vector<BufferView> dstBuffers;
    std::vector<BufferMetaInfo> bufferMetas;
};

enum class MemcopyPolicy : int {
    DIRECT,
    FFTS,
    HUGE_FFTS,
};

struct MemcopyConfig {
    void Init();
    std::string ToString();
    Status GetNumberFromEnv(const char *key, uint64_t &value);
    Status GetPolicyFromEnv(const char *key, MemcopyPolicy &policy);

    const uint64_t defaultHostMemSize = 2684354560;   // 2.5G
    const uint64_t defaultDeviceMemSize = 104857600;  // 100MB
    const uint64_t defaultBlockSize = 10485760;       // 10MB
    MemcopyPolicy policyD2H = MemcopyPolicy::FFTS;
    MemcopyPolicy policyH2D = MemcopyPolicy::FFTS;
    uint64_t deviceMemSize = defaultDeviceMemSize;
    uint64_t hostMemSize = defaultHostMemSize;
};

using datasystem::memory::Allocator;
using datasystem::memory::DevMemFuncRegister;
using AllocateType = datasystem::memory::CacheType;

class MemMgrBase {
public:
    MemMgrBase(Allocator *allocator) : allocator_(allocator)
    {
    }

    virtual ~MemMgrBase() = default;

    Status Init();

    Status Allocate(const std::vector<BufferMetaInfo> &bMeta, std::vector<ShmUnit> &memoryPool, bool skipRetry = false);

    Status Free(std::vector<ShmUnit> &memoryPool);

protected:
    std::mutex memPoolLock_;

    void *ptr_ = nullptr;
    Allocator *allocator_ = nullptr;
    const std::string DEFAULT_TENANTID = "";
    AllocateType type_ = AllocateType::DEV_HOST;
    std::unique_ptr<WaitPost> waitPost_{ nullptr };  // wait for some second to check memory is free
};

class HostMemMgr : public MemMgrBase {
public:
    HostMemMgr(Allocator *allocator);
    ~HostMemMgr() = default;

    Status HostMemoryCopy(void *dstData, uint64_t dstLength, void *srcData, uint64_t srcLength);

protected:
    std::shared_ptr<ThreadPool> memoryCopyThreadPool_;
};

class DeviceMemMgr : public MemMgrBase {
public:
    DeviceMemMgr(Allocator *allocator) : MemMgrBase(allocator)
    {
        type_ = AllocateType::DEV_DEVICE;
    }
    ~DeviceMemMgr() = default;
};

class ResourceMgr {
public:
    ResourceMgr()
    {
        deviceResources_.reserve(MAX_DEVICE_COUNT);
        for (size_t deviceId = 0; deviceId < MAX_DEVICE_COUNT; deviceId++) {
            deviceResources_.emplace_back(std::make_unique<AclDeviceResource>(deviceId));
        }
        config.Init();
        swapOutPool_ = std::make_unique<AsyncAclMemCopyPool>(this);
        swapInPool_ = std::make_unique<AsyncAclMemCopyPool>(this);
    };
    ~ResourceMgr();
    Status Init();
    // Should be called after Init, to make sure the mem mgr is ready.
    HostMemMgr *Host()
    {
        return hostMemMgr_.get();
    }
    DeviceMemMgr *Device()
    {
        return deviceMemMgr_.get();
    }
    DeviceResourceList &DeviceResources()
    {
        return deviceResources_;
    }

    void SetPolicyDirect()
    {
        config.policyD2H = MemcopyPolicy::DIRECT;
        config.policyH2D = MemcopyPolicy::DIRECT;
    }

    void SetPolicyByHugeTlb(bool enableHugeTlb)
    {
        if (enableHugeTlb && config.policyD2H == MemcopyPolicy::FFTS) {
            config.policyD2H = MemcopyPolicy::HUGE_FFTS;
            config.hostMemSize = 0;
        }
        if (enableHugeTlb && config.policyH2D == MemcopyPolicy::FFTS) {
            config.policyH2D = MemcopyPolicy::HUGE_FFTS;
            config.hostMemSize = 0;
        }
    }

    MemcopyPolicy GetD2HPolicy()
    {
        return config.policyD2H;
    }

    MemcopyPolicy GetH2DPolicy()
    {
        return config.policyH2D;
    }

    Status MemcpyBatchD2H(const std::vector<DeviceBlobList> &devBlobList, std::vector<Buffer *> &bufferList)
    {
        auto deviceId = devBlobList[0].deviceIdx;
        DeviceBatchCopyHelper helper;
        RETURN_IF_NOT_OK(helper.Prepare(devBlobList, bufferList, MemcpyKind::DEVICE_TO_HOST));
        return swapOutPool_->MemcpyBatchD2H(deviceId, helper, config.policyD2H);
    }

    Status MemcpyBatchH2D(const std::vector<DeviceBlobList> &devBlobList, std::vector<Buffer *> &bufferList)
    {
        auto deviceId = devBlobList[0].deviceIdx;
        DeviceBatchCopyHelper helper;
        RETURN_IF_NOT_OK(helper.Prepare(devBlobList, bufferList, MemcpyKind::HOST_TO_DEVICE));
        return swapInPool_->MemcpyBatchH2D(deviceId, helper, config.policyH2D);
    }

    // #ifdef USE_NPU

    // #endif
private:
    MemcopyConfig config;
    const size_t CACHE_SIZE = 8;
    std::shared_timed_mutex mutex_;
    std::unique_ptr<HostMemMgr> hostMemMgr_;
    std::unique_ptr<DeviceMemMgr> deviceMemMgr_;

    DeviceResourceList deviceResources_;

    std::unique_ptr<AsyncMemCopyPool> swapOutPool_;
    std::unique_ptr<AsyncMemCopyPool> swapInPool_;
};

}  // namespace datasystem
#endif
