
#ifndef DATASYSTEM_COMMON_DEVICE_NVIDIA_CUDA_RESOURCE_MANAGER_H
#define DATASYSTEM_COMMON_DEVICE_NVIDIA_CUDA_RESOURCE_MANAGER_H

#include "datasystem/common/device/device_resource_manager.h"

namespace datasystem {

class CudaResourceManager : public DeviceResourceManager {
public:
    CudaResourceManager() = default;
    ~CudaResourceManager() = default;

    void SetD2HPolicyByHugeTlb(bool enableHugeTlb)
    {
        (void)enableHugeTlb;
    }
    Status MemcpyBatchD2H(const std::vector<DeviceBlobList> &devBlobList, std::vector<Buffer *> &bufferList) override
    {
        (void)devBlobList;
        (void)bufferList;
        return Status::OK();
    }
    Status MemcpyBatchH2D(const std::vector<DeviceBlobList> &devBlobList, std::vector<Buffer *> &bufferList) override
    {
        (void)devBlobList;
        (void)bufferList;
        return Status::OK();
    }
    void SetPolicyByHugeTlb(bool enableHugeTlb) override
    {
        (void)enableHugeTlb;
    }
};
}  // namespace datasystem

#endif  // DATASYSTEM_COMMON_DEVICE_NVIDIA_CUDA_RESOURCE_MANAGER_H
