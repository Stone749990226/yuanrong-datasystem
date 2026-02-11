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
 * Description: Defines the ascend device resource.
 */

#ifndef DATASYSTEM_COMMON_DEVICE_ACL_DEVICE_RESOURCE_H
#define DATASYSTEM_COMMON_DEVICE_ACL_DEVICE_RESOURCE_H

#include <shared_mutex>
#include <vector>
#include "datasystem/utils/status.h"
#include "datasystem/common/device/ascend/ffts_dispatcher.h"
#include "datasystem/common/device/ascend/callback_thread.h"
#include "datasystem/common/device/ascend/cann_types.h"
#include "datasystem/hetero/device_common.h"
#include "datasystem/common/util/thread_pool.h"
#include "datasystem/common/shared_memory/shm_unit.h"
#define CHECK_ACL_RESULT(aclRet, apiName)                                                             \
    do {                                                                                              \
        int _aclRet = (aclRet);                                                                       \
        if (_aclRet != 0) {                                                                           \
            std::string errMsg = FormatString("%s api failed with error code %d ", apiName, _aclRet); \
            return Status(StatusCode::K_ACL_ERROR, __LINE__, __FILE__, errMsg);                       \
        }                                                                                             \
    } while (false)

namespace datasystem {
class ResourceMgr;
class HostMemMgr;
class DeviceMemMgr;
const size_t FFTS_PIPELINE = 2;
const size_t MAX_DEVICE_COUNT = 64;

class AclDeviceResource {
public:
    AclDeviceResource(uint32_t deviceId) : deviceId_(deviceId)
    {
    }
    Status CreateAclRtStream(aclrtStream &stream, bool subscribeReport);
    Status FreeAclRtStream(aclrtStream stream, bool subscribeReport);
    Status CreateRtNotify(rtNotify_t &notify);
    Status FreeRtNotify(rtNotify_t notify);

private:
    Status InitCallbackThread();
    const size_t CACHE_SIZE = 8;
    uint32_t deviceId_;
    std::shared_timed_mutex mutex_;
    std::unique_ptr<ffts::FftsDispatcher> fftsDispatcher_;
    std::unique_ptr<acl::CallbackThread> callbackThread_;
    std::deque<aclrtStream> streamQueue_;
    std::deque<aclrtStream> subscribeReportStreamQueue_;
    std::deque<rtNotify_t> notifyQueue_;
};

using AclDeviceResourceList = std::vector<std::unique_ptr<AclDeviceResource>>;

struct AclResource {
    void *primaryStream;
    void *secondaryStream;
    void *toDestDone[FFTS_PIPELINE];
    void *toPinDone[FFTS_PIPELINE];
    bool subscribeReport;
};

struct PipelineH2DTasks {
    std::vector<BufferView> srcBuffers;
    std::vector<BufferView> destBuffers;
    std::vector<BufferMetaInfo> bufferMetas;
    bool IsEmpty()
    {
        return srcBuffers.empty();
    }
};

class FftsPipelineCopierBase {
public:
    FftsPipelineCopierBase(uint32_t deviceId, AclDeviceResource &deviceResource, HostMemMgr *hostMemMgr,
                           DeviceMemMgr *deviceMemMgr, const std::vector<BufferMetaInfo> &bufferMetas,
                           ThreadPool *h2hCopyPool, ThreadPool *fftsCopyPool, bool skipH2HMemcpy);
    ~FftsPipelineCopierBase();

    FftsPipelineCopierBase(const FftsPipelineCopierBase &) = delete;
    FftsPipelineCopierBase &operator=(const FftsPipelineCopierBase &) = delete;

protected:
    Status GetBufferViews(size_t count, const std::vector<ShmUnit> &memoryPool, std::vector<BufferView> &buffers);

    Status AllocAndInitTransferBuffers(const std::vector<BufferView> &hostBuffer);

    bool IsFinish()
    {
        return finishCount_ >= bufferMetas_.size();
    }

    Status InitAclResource(bool subscribeReport);

    Status NotifyStart();
    Status WaitFinish();

    acl::AclDeviceManager *aclDeviceManager_;
    const int32_t deviceId_;
    AclDeviceResource &deviceResource_;
    HostMemMgr *hostMemMgr_;
    DeviceMemMgr *deviceMemMgr_;
    const std::vector<BufferMetaInfo> &bufferMetas_;
    AclResource resource_;
    std::unique_ptr<ffts::FftsDispatcher> fftsDispatcher_;
    ThreadPool *h2hCopyPool_;
    ThreadPool *fftsCopyPool_;
    bool skipH2HMemcpy_;
    std::vector<BufferView> transferHostBuffers_;
    std::vector<BufferView> transferDeviceBuffers_;
    std::vector<ShmUnit> transferHostPool_;
    std::vector<ShmUnit> transferDevicePool_;

    std::mutex mutex_;
    std::condition_variable cv_;
    size_t finishCount_;
};

class FftsPipelineH2DCopier : public FftsPipelineCopierBase {
public:
    FftsPipelineH2DCopier(uint32_t deviceId, AclDeviceResource &deviceResource, HostMemMgr *hostMemMgr,
                          DeviceMemMgr *deviceMemMgr, const std::vector<BufferMetaInfo> &bufferMetas,
                          ThreadPool *h2hCopyPool, ThreadPool *fftsCopyPool, bool skipH2HMemcpy);
    ~FftsPipelineH2DCopier() = default;
    FftsPipelineH2DCopier(const FftsPipelineH2DCopier &) = delete;
    FftsPipelineH2DCopier &operator=(const FftsPipelineH2DCopier &) = delete;

    Status ExecuteMemcpy(const std::vector<BufferView> &deviceBuffers, const std::vector<BufferView> &hostBuffers);

    Status AddFftsNotifyTask(size_t index, const std::vector<BufferView> &deviceBuffers, bool addTask = true);

private:
    void AddTask(size_t index, const std::vector<BufferView> &deviceBuffers);
    Status SubmitToStream(const std::vector<BufferView> &srcBuffers, const std::vector<BufferView> &destBuffers,
                          const std::vector<BufferMetaInfo> &bufferMetas);

    PipelineH2DTasks tasks_;
    size_t blobOffset_;
    std::atomic<size_t> submitCount_;
};

class FftsPipelineD2HCopier;
struct NotifyH2HCallbackData {
    FftsPipelineD2HCopier *copier;
    size_t index;
};

struct PipelineH2HTasks {
    std::vector<size_t> indexes;
    bool IsEmpty()
    {
        return indexes.empty();
    }
};

class FftsPipelineD2HCopier : public FftsPipelineCopierBase {
public:
    FftsPipelineD2HCopier(uint32_t deviceId, AclDeviceResource &deviceResource, HostMemMgr *hostMemMgr,
                          DeviceMemMgr *deviceMemMgr, const std::vector<BufferMetaInfo> &bufferMetas,
                          ThreadPool *h2hCopyPool, ThreadPool *fftsCopyPool, bool skipH2HMemcpy);
    ~FftsPipelineD2HCopier() = default;
    FftsPipelineD2HCopier(const FftsPipelineH2DCopier &) = delete;
    FftsPipelineD2HCopier &operator=(const FftsPipelineH2DCopier &) = delete;

    Status ExecuteMemcpy(const std::vector<BufferView> &hostBuffers, const std::vector<BufferView> &deviceBuffers);

private:
    Status SubmitToStream(const std::vector<BufferView> &srcBuffers, const std::vector<BufferView> &transferBuffers,
                          const std::vector<BufferView> &destBuffers, const std::vector<BufferMetaInfo> &bufferMetas,
                          std::vector<NotifyH2HCallbackData> &callbackDatas);
    static void NotifyH2HCallback(void *userData);
    void ForceFinish();

    PipelineH2HTasks tasks_;
};
}  // namespace datasystem
#endif  // DATASYSTEM_COMMON_DEVICE_ACL_DEVICE_RESOURCE_H
