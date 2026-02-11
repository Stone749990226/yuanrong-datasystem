

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
#ifndef DATASYSTEM_COMMON_DEVICE_ACL_MEMCPY_POOL_H
#define DATASYSTEM_COMMON_DEVICE_ACL_MEMCPY_POOL_H

#include "datasystem/common/device/ascend/acl_device_resource.h"
#include "datasystem/common/device/device_manager_base.h"
#include "datasystem/common/util/thread_pool.h"

namespace datasystem {
class ResourceMgr;
class DeviceBatchCopyHelper;
enum class MemcopyPolicy : int;
class AsyncAclMemCopyPool {
public:
    AsyncAclMemCopyPool(ResourceMgr *resourceMgr);
    /**
     * @brief Perform a batch memory copy operation on the NPU.
     *
     * @param[in] copyKind   Type of memory copy.
     * @param[in] helper     Helper object that holds the batch copy tasks.
     * @param[in] deviceId   Target device ID.
     * @return Status K_OK on success; an error code otherwise.
     */
    Status MemcpyBatchD2H(uint32_t deviceId, DeviceBatchCopyHelper &helper, MemcopyPolicy policy);

    Status MemcpyBatchH2D(uint32_t deviceId, DeviceBatchCopyHelper &helper, MemcopyPolicy policy);

    ~AsyncAclMemCopyPool();

private:
    Status AclMemcpyBatch(uint32_t deviceId, DeviceBatchCopyHelper &helper, MemcpyKind copyKind);
    std::unique_ptr<ThreadPool> copyPool_;
    std::unique_ptr<ThreadPool> h2hCopyPool_;
    std::unique_ptr<ThreadPool> fftsCopyPool_;
    std::vector<void *> copyStreams_;
    int32_t deviceNow_ = -1;
    DeviceManagerBase *devInterImpl_;
    ResourceMgr *resourceMgr_;
};
}  // namespace datasystem
#endif  // DATASYSTEM_COMMON_DEVICE_ACL_MEMCPY_POOL_H
