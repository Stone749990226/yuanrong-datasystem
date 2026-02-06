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
 * Description: Simple test for HeteroClient MGetH2D and MSetD2H functions.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "datasystem/hetero_client.h"
#include "datasystem/hetero/device_common.h"
#include "datasystem/utils/status.h"

#ifdef BUILD_HETERO
#include <acl/acl.h>
#endif

namespace datasystem {
namespace ut {

// Simple test for MSetD2H and MGetH2D
TEST(HeteroClientH2DD2HTest, BasicRoundTripTest)
{
#ifndef BUILD_HETERO
    GTEST_SKIP() << "Test requires BUILD_HETERO to be enabled";
#else
    // Setup connection options
    ConnectOptions connectOptions = { .host = "127.0.0.1", .port = 31501 };
    auto client = std::make_shared<HeteroClient>(connectOptions);

    // Initialize client
    Status status = client->Init();
    if (status.IsError()) {
        GTEST_SKIP() << "Cannot initialize client, worker may not be running: " << status.GetMsg();
    }

    // Initialize ACL
    int deviceId = 0;
    int ret = aclInit(nullptr);
    if (ret != 0) {
        GTEST_SKIP() << "ACL initialization failed, NPU may not be available";
    }
    ret = aclrtSetDevice(deviceId);
    ASSERT_EQ(ret, 0) << "Failed to set device";

    // Prepare test data
    std::vector<std::string> keys = { "test-key1" };
    std::vector<uint64_t> blobSize = { 1024, 2048 };
    int blobNum = blobSize.size();

    // Allocate HBM memory for MSetD2H (Device to Host)
    std::vector<DeviceBlobList> swapOutBlobList;
    swapOutBlobList.resize(keys.size());
    for (size_t i = 0; i < swapOutBlobList.size(); i++) {
        swapOutBlobList[i].deviceIdx = deviceId;
        for (int j = 0; j < blobNum; j++) {
            void *devPtr = nullptr;
            int code = aclrtMalloc(&devPtr, blobSize[j], ACL_MEM_MALLOC_HUGE_FIRST);
            ASSERT_EQ(code, 0) << "Failed to allocate device memory";

            // Fill device memory with test pattern
            std::string testData(blobSize[j], 'A' + j);
            aclrtMemcpy(devPtr, blobSize[j], testData.data(), blobSize[j], ACL_MEMCPY_HOST_TO_DEVICE);

            Blob blob = { .pointer = devPtr, .size = blobSize[j] };
            swapOutBlobList[i].blobs.emplace_back(std::move(blob));
        }
    }

    // Test MSetD2H: Transfer data from device to host
    status = client->MSetD2H(keys, swapOutBlobList);
    ASSERT_TRUE(status.IsOk()) << "MSetD2H failed: " << status.GetMsg();

    // Allocate HBM memory for MGetH2D (Host to Device)
    std::vector<DeviceBlobList> swapInBlobList;
    swapInBlobList.resize(keys.size());
    for (size_t i = 0; i < swapInBlobList.size(); i++) {
        swapInBlobList[i].deviceIdx = deviceId;
        for (int j = 0; j < blobNum; j++) {
            void *devPtr = nullptr;
            int code = aclrtMalloc(&devPtr, blobSize[j], ACL_MEM_MALLOC_HUGE_FIRST);
            ASSERT_EQ(code, 0) << "Failed to allocate device memory";
            Blob blob = { .pointer = devPtr, .size = blobSize[j] };
            swapInBlobList[i].blobs.emplace_back(std::move(blob));
        }
    }

    // Test MGetH2D: Transfer data from host to device
    std::vector<std::string> failedList;
    status = client->MGetH2D(keys, swapInBlobList, failedList, 5000);
    ASSERT_TRUE(status.IsOk()) << "MGetH2D failed: " << status.GetMsg();
    EXPECT_TRUE(failedList.empty()) << "Some keys failed to get";

    // Verify data integrity
    for (size_t i = 0; i < swapInBlobList.size(); i++) {
        for (size_t j = 0; j < swapInBlobList[i].blobs.size(); j++) {
            std::vector<char> retrievedData(blobSize[j]);
            aclrtMemcpy(retrievedData.data(), blobSize[j],
                       swapInBlobList[i].blobs[j].pointer, blobSize[j],
                       ACL_MEMCPY_DEVICE_TO_HOST);

            std::string expectedData(blobSize[j], 'A' + j);
            EXPECT_EQ(std::string(retrievedData.data(), blobSize[j]), expectedData)
                << "Data mismatch for blob " << j;
        }
    }

    // Cleanup: Free device memory
    for (auto &blobList : swapOutBlobList) {
        for (auto &blob : blobList.blobs) {
            if (blob.pointer != nullptr) {
                aclrtFree(blob.pointer);
            }
        }
    }
    for (auto &blobList : swapInBlobList) {
        for (auto &blob : blobList.blobs) {
            if (blob.pointer != nullptr) {
                aclrtFree(blob.pointer);
            }
        }
    }

    // Delete keys from cache
    std::vector<std::string> deleteFailedKeys;
    client->Delete(keys, deleteFailedKeys);

    // Cleanup ACL
    aclrtResetDevice(deviceId);
    aclFinalize();

    // Shutdown client
    client->ShutDown();
#endif
}

}  // namespace ut
}  // namespace datasystem
