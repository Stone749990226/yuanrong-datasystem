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

#include <securec.h>
#include <cstring>
#include <sstream>
#include <string>

#include "datasystem/common/device/resource_mgr.h"
#include "datasystem/common/device/device_manager_factory.h"
#include "datasystem/common/device/ascend/acl_device_resource.h"
#include "datasystem/common/device/ascend/acl_memcpy_pool.h"
#include "datasystem/common/device/ascend/ffts_dispatcher.h"
#include "datasystem/common/inject/inject_point.h"
#include "datasystem/common/perf/perf_manager.h"
#include "datasystem/common/util/format.h"
#include "datasystem/common/util/memory.h"
#include "datasystem/common/util/status_helper.h"
#include "datasystem/common/util/strings_util.h"
#include "datasystem/common/util/timer.h"
#include "datasystem/utils/status.h"

namespace datasystem {

void MemcopyConfig::Init()
{
    LOG_IF_ERROR(GetNumberFromEnv("DS_DEVICE_ACL_SIZE", deviceMemSize), "GetNumberFromEnv failed");
    LOG_IF_ERROR(GetNumberFromEnv("DS_HOST_ACL_SIZE", hostMemSize), "GetNumberFromEnv failed");
    LOG_IF_ERROR(GetPolicyFromEnv("DS_D2H_MEMCPY_POLICY", policyD2H), "GetPolicyFromEnv failed");
    LOG_IF_ERROR(GetPolicyFromEnv("DS_H2D_MEMCPY_POLICY", policyH2D), "GetPolicyFromEnv failed");
}

std::string MemcopyConfig::ToString()
{
    std::stringstream ss;
    ss << "MemcopyConfig { policyD2H:" << static_cast<int>(policyD2H);
    ss << ", policyH2D:" << static_cast<int>(policyH2D);
    ss << ", deviceMemSize:" << deviceMemSize;
    ss << ", hostMemSize:" << hostMemSize;
    ss << "}";
    return ss.str();
}

Status MemcopyConfig::GetNumberFromEnv(const char *key, uint64_t &value)
{
    auto strValue = std::getenv(key);
    RETURN_OK_IF_TRUE(strValue == nullptr);
    try {
        uint64_t ret = StrToUnsignedLong(strValue);
        if (ret == 0) {
            throw std::out_of_range("Memory should not be set to zero.");
        }
        value = ret;
    } catch (std::invalid_argument &invalidArgument) {
        RETURN_STATUS(StatusCode::K_RUNTIME_ERROR,
                      FormatString("Env %s value %s parse to number failed, invalid argument", key, strValue));
    } catch (std::out_of_range &outOfRange) {
        RETURN_STATUS(StatusCode::K_RUNTIME_ERROR,
                      FormatString("Env %s value %s parse to number failed, out of range", key, strValue));
    }
    return Status::OK();
}

Status MemcopyConfig::GetPolicyFromEnv(const char *key, MemcopyPolicy &policy)
{
    auto strValue = std::getenv(key);
    RETURN_OK_IF_TRUE(strValue == nullptr);
    std::string str = strValue;
    if (str == "ffts") {
        policy = MemcopyPolicy::FFTS;
    } else if (str == "direct") {
        policy = MemcopyPolicy::DIRECT;
    } else if (str == "huge_ffts") {
        policy = MemcopyPolicy::HUGE_FFTS;
    } else {
        RETURN_STATUS(K_INVALID, FormatString("Unknown memcopy policy %s from env %s", str, key));
    }
    return Status::OK();
}

Status MemMgrBase::Init()
{
    int index;
    auto rc = acl::AclDeviceManager::Instance()->GetDeviceIdx(index);
    if (rc.IsError()) {
        LOG(WARNING) << "Not set device idx yet, return warning!";
        return Status::OK();
    }
    waitPost_ = std::make_unique<WaitPost>();
    return Status::OK();
}

Status MemMgrBase::Allocate(const std::vector<BufferMetaInfo> &bMeta, std::vector<ShmUnit> &memoryPool, bool skipRetry)
{
    std::lock_guard<std::mutex> lock(memPoolLock_);
    uint64_t batchSize = bMeta.size();
    uint64_t maxAllocateSize = 0;
    if (type_ == AllocateType::DEV_DEVICE) {
        for (const auto &meta : bMeta) {
            maxAllocateSize = std::max(maxAllocateSize, meta.size);
        }
        batchSize = memoryPool.size();
    }
    uint32_t intervalMs = 100;
    for (uint64_t i = 0; i < batchSize; i++) {
        int retryNums = 10;
        if (skipRetry) {
            retryNums = 0;
        }
        Status rc = Status::OK();
        do {
            if (rc.IsError()) {
                waitPost_->WaitFor(intervalMs);
            }

            size_t allocSize = (type_ == AllocateType::DEV_DEVICE) ? maxAllocateSize : bMeta[i].size;
            rc = memoryPool[i].AllocateMemory(DEFAULT_TENANTID, allocSize, false, ServiceType::OBJECT, type_);
            if (retryNums <= 0) {
                break;
            }
            retryNums--;
        } while (rc.IsError() && rc.GetCode() == K_OUT_OF_MEMORY);
        if (skipRetry) {
            RETURN_IF_NOT_OK(rc);
        } else {
            RETURN_IF_NOT_OK_PRINT_ERROR_MSG(rc, FormatString("Failed to allocate memory with size %d", bMeta[i].size));
        }
    }
    return Status::OK();
}

Status MemMgrBase::Free(std::vector<ShmUnit> &memoryPool)
{
    std::lock_guard<std::mutex> lock(memPoolLock_);
    uint64_t size = memoryPool.size();
    for (uint64_t i = 0; i < size; i++) {
        RETURN_IF_NOT_OK_PRINT_ERROR_MSG(memoryPool[i].FreeMemory(), "Failed to free memory");
    }
    return Status::OK();
}

HostMemMgr::HostMemMgr(Allocator *allocator) : MemMgrBase(allocator)
{
    type_ = AllocateType::DEV_HOST;
    memoryCopyThreadPool_ = std::make_shared<ThreadPool>(0, GetRecommendedMemoryCopyThreadsNum());
}

Status HostMemMgr::HostMemoryCopy(void *dstData, uint64_t dstLength, void *srcData, uint64_t srcLength)
{
    CHECK_FAIL_RETURN_STATUS_PRINT_ERROR(dstData != nullptr, K_INVALID, "Can't put null dst ptr!");
    CHECK_FAIL_RETURN_STATUS_PRINT_ERROR(srcData != nullptr, K_INVALID, "Can't put null src ptr!");
    CHECK_FAIL_RETURN_STATUS_PRINT_ERROR(
        dstLength > 0 && srcLength > 0, K_INVALID,
        FormatString("length must greater than 0! dstLength : %lld, srcLength : %lld", dstLength, srcLength));
    Status status = ::datasystem::MemoryCopy(static_cast<uint8_t *>(dstData), dstLength,
                                             static_cast<const uint8_t *>(srcData), srcLength, memoryCopyThreadPool_);
    CHECK_FAIL_RETURN_STATUS_PRINT_ERROR(status.IsOk(), K_RUNTIME_ERROR,
                                         FormatString("Copy data to buffer failed, err: %s", status.ToString()));
    return Status::OK();
}

ResourceMgr::~ResourceMgr() = default;

Status ResourceMgr::Init()
{
    {
        std::shared_lock<std::shared_timed_mutex> rlocker(mutex_);
        if (hostMemMgr_ != nullptr && deviceMemMgr_ != nullptr) {
            return Status::OK();
        }
    }
    INJECT_POINT_NO_RETURN("ResourceMgr.Init");

    auto devInterImpl = DeviceManagerFactory::GetDeviceManager();
    struct DevMemFuncRegister regFunc;

    auto hostAllocFunc = [devInterImpl](void **ptr, size_t maxSize) -> Status {
        return devInterImpl->MallocHost(&(*ptr), maxSize);
    };
    auto hostdestroyFunc = [](void *ptr, size_t destroySize) -> Status {
        (void)destroySize;
        if (ptr) {
            // do not free in instance yet (destroy func : devInterImpl->aclrtFreeHost(ptr))
            // destroy may cause interrupt with acl static data
            return Status::OK();
        }
        return Status::OK();
    };
    auto devAllocFunc = [devInterImpl](void **ptr, size_t maxSize) -> Status {
        return devInterImpl->Malloc(&(*ptr), maxSize, MemMallocPolicy::HUGE_FIRST);
    };
    auto devdestroyFunc = [](void *ptr, size_t destroySize) -> Status {
        (void)destroySize;
        if (ptr) {
            // do not free in instance yet (destroy func : devInterImpl->aclrtFree(ptr)
            // destroy may cause interrupt with acl static data
            return Status::OK();
        }
        return Status::OK();
    };
    regFunc.devDeviceCreateFunc = devAllocFunc;
    regFunc.devDeviceDestroyFunc = devdestroyFunc;
    regFunc.devHostCreateFunc = hostAllocFunc;
    regFunc.devHostDestroyFunc = hostdestroyFunc;

    std::lock_guard<std::shared_timed_mutex> wlocker(mutex_);
    auto *allocator = Allocator::Instance();
    LOG(INFO) << config.ToString();
    allocator->InitWithoutShm(config.deviceMemSize, config.hostMemSize, regFunc);

    if (!hostMemMgr_) {
        hostMemMgr_ = std::make_unique<HostMemMgr>(allocator);
        auto rc = hostMemMgr_->Init();
        if (rc.IsError()) {
            hostMemMgr_.reset();
        }
        RETURN_IF_NOT_OK_PRINT_ERROR_MSG(rc, "Failed to init dev host mem.");
    }

    if (!deviceMemMgr_) {
        deviceMemMgr_ = std::make_unique<DeviceMemMgr>(allocator);
        auto rc = deviceMemMgr_->Init();
        if (rc.IsError()) {
            deviceMemMgr_.reset();
        }
        RETURN_IF_NOT_OK_PRINT_ERROR_MSG(rc, "Failed to init dev device mem.");
    }
    return Status::OK();
}

}  // namespace datasystem
