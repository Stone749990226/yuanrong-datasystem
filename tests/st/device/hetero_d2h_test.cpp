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
 * Description: hetero d2h test.
 */
#include "device/dev_test_helper.h"

namespace datasystem {
using namespace acl;
namespace st {
class HeteroD2HTest : public DevTestHelper {
    void SetClusterSetupOptions(ExternalClusterOptions &opts) override
    {
        opts.numWorkers = 1;
        opts.workerGflagParams =
            " -v=0 -authorization_enable=true -shared_memory_size_mb=4096 -enable_fallocate=false -arena_per_tenant=2 ";
        opts.enableDistributedMaster = "false";
        opts.numEtcd = 1;
        FLAGS_v = 0;
    }

    void SetUp() override
    {
        const char *ascend_root = std::getenv("ASCEND_HOME_PATH");
        if (ascend_root == nullptr) {
            DS_ASSERT_OK(datasystem::inject::Set("NO_USE_FFTS", "call()"));
            DS_ASSERT_OK(datasystem::inject::Set("client.GetOrCreateHcclComm.setIsSameNode", "call(0)"));
            BINEXPECT_CALL(AclDeviceManager::Instance, ()).WillRepeatedly(Return(&managerMock_));
        }
        ExternalClusterTest::SetUp();
    }

    void TearDown() override
    {
        ExternalClusterTest::TearDown();
    }
};

class HeteroD2HTestEvcit : public HeteroD2HTest {
    void SetClusterSetupOptions(ExternalClusterOptions &opts) override
    {
        opts.numWorkers = 1;
        opts.workerGflagParams =
            "-log_monitor_interval_ms=1000 -v=2 -authorization_enable=true -shared_memory_size_mb=2096 "
            "-enable_fallocate=false -arena_per_tenant=2 ";
        opts.enableDistributedMaster = "false";
        opts.numEtcd = 1;
        opts.enableSpill = true;
        FLAGS_v = 0;
    }
};

class HeteroD2HTestEvictThreshold : public HeteroD2HTest {
    void SetClusterSetupOptions(ExternalClusterOptions &opts) override
    {
        opts.numWorkers = 1;
        opts.workerGflagParams =
            "-log_monitor_interval_ms=1000 -v=2 -authorization_enable=true -shared_memory_size_mb=560 "
            "-enable_fallocate=false -arena_per_tenant=2 -eviction_reserve_mem_threshold_mb=100 ";
        opts.enableDistributedMaster = "true";  // Match Python test default (distributed mode)
        opts.numEtcd = 1;
        opts.enableSpill = false;  // No spill directory, match Python test
        FLAGS_v = 0;
    }
};

class HeteroD2HThroughTcpTest : public DevTestHelper {
    void SetClusterSetupOptions(ExternalClusterOptions &opts) override
    {
        auto workerNum = 2;
        opts.numWorkers = workerNum;
        opts.workerGflagParams =
            " -ipc_through_shared_memory=false -log_monitor=true -v=1 -authorization_enable=true "
            "-shared_memory_size_mb=4096 "
            "-enable_fallocate=false -arena_per_tenant=2 ";
        opts.enableDistributedMaster = "false";
        opts.numEtcd = 1;
        FLAGS_v = 1;
    }

    void SetUp() override
    {
        const char *ascend_root = std::getenv("ASCEND_HOME_PATH");
        if (ascend_root == nullptr) {
            DS_ASSERT_OK(datasystem::inject::Set("NO_USE_FFTS", "call()"));
            DS_ASSERT_OK(datasystem::inject::Set("client.GetOrCreateHcclComm.setIsSameNode", "call(0)"));
            BINEXPECT_CALL(AclDeviceManager::Instance, ()).WillRepeatedly(Return(&managerMock_));
        }
        ExternalClusterTest::SetUp();
    }

    void TearDown() override
    {
        ExternalClusterTest::TearDown();
    }
};

TEST_F(HeteroD2HThroughTcpTest, SetGet)
{
    std::shared_ptr<HeteroClient> c0, c1;
    size_t numOfObjs = 10, blksPerObj = 1024, blkSz = 1024;
    std::vector<std::string> inObjectKeys, failedKeys;
    std::vector<DeviceBlobList> devGetBlobList, devSetBlobList;
    for (auto j = 0ul; j < numOfObjs; j++) {
        inObjectKeys.emplace_back(FormatString("key_%s", j));
    }
    auto retcode = -1;
    auto hook = [&retcode]() { exit(retcode); };
    auto setPid = fork();
    if (setPid == 0) {
        Raii raii(hook);
        InitAcl(0);
        InitTestHeteroClient(0, c0);
        PrePareDevData(numOfObjs, blksPerObj, blkSz, devGetBlobList, devSetBlobList, 0);
        DS_ASSERT_OK(c0->MSetD2H(inObjectKeys, devSetBlobList));
        c0.reset();
        retcode = 0;
    }
    ASSERT_FALSE(WaitForChildFork(setPid));
    auto getPid = fork();
    if (getPid == 0) {
        Raii raii(hook);
        InitTestHeteroClient(1, c1);
        InitAcl(1);
        PrePareDevData(numOfObjs, blksPerObj, blkSz, devGetBlobList, devSetBlobList, 1);
        DS_ASSERT_OK(c1->MGetH2D(inObjectKeys, devGetBlobList, failedKeys, 0));
        ASSERT_TRUE(failedKeys.empty());
        DS_ASSERT_OK(IsSameContent(devGetBlobList, devSetBlobList, 'b'));
        c1.reset();
        retcode = 0;
    }
    ASSERT_FALSE(WaitForChildFork(getPid));
}

TEST_F(HeteroD2HTest, Perf)
{
    std::vector<uint64_t> keyNums{ 1, 10, 450, 500, 1000 };
    std::vector<std::vector<double>> costs(keyNums.size());
    std::shared_ptr<HeteroClient> client;
    InitAcl(0);
    InitTestHeteroClient(0, client);
    auto repeatNum = 100u, testIdx = 0u, blkSz = 1024u, blksPerObj = 1u;
    for (auto numOfObjs : keyNums) {
        auto idx = 0u;
        for (auto i = 0u; i < repeatNum; i++) {
            LOG(INFO) << FormatString("start test MSetD2H Test ##### %lu KB, Round:%lu", numOfObjs, idx);

            std::vector<std::string> inObjectKeys;
            std::vector<DeviceBlobList> devGetBlobList, devSetBlobList;
            for (auto j = 0ul; j < numOfObjs; j++) {
                inObjectKeys.emplace_back(FormatString("round_%lu_key_%s", idx, j));
            }
            PrePareDevData(numOfObjs, blksPerObj, blkSz, devGetBlobList, devSetBlobList, 0);
            Timer t;
            DS_ASSERT_OK(client->MSetD2H(inObjectKeys, devSetBlobList));
            costs[testIdx].push_back(t.ElapsedMicroSecond());
            idx++;
        }
        testIdx++;
    }
    for (auto i = 0u; i < keyNums.size(); i++) {
        LOG(INFO) << FormatString(
            "#### MSetD2H Test #####  %4lu KB, Key Size: %4lu B, Key num:%4lu  Round:%4lu, --- result(ms): %s",
            keyNums[i], blkSz * blksPerObj, keyNums[i], repeatNum, GetTimeCostUsState(costs[i]));
    }
}

TEST_F(HeteroD2HTest, TestNoExist)
{
    std::vector<uint64_t> keyNums{ 450, 500 };
    std::vector<std::vector<double>> costs(keyNums.size());
    std::shared_ptr<HeteroClient> client;
    InitAcl(0);
    InitTestHeteroClient(0, client);
    auto testIdx = 0u, blkSz = 1024u, blksPerObj = 1u;
    for (auto numOfObjs : keyNums) {
        LOG(INFO) << FormatString("start test MSetD2H Test ##### %lu KB", numOfObjs);
        std::vector<std::string> inObjectKeys, failedKeys;
        std::vector<DeviceBlobList> devGetBlobList, devSetBlobList;
        for (auto j = 0ul; j < numOfObjs; j++) {
            inObjectKeys.emplace_back(FormatString("key_%s", j));
        }
        PrePareDevData(numOfObjs, blksPerObj, blkSz, devGetBlobList, devSetBlobList, 0);
        DS_ASSERT_OK(client->MSetD2H(inObjectKeys, devSetBlobList));
        DS_ASSERT_OK(client->MGetH2D(inObjectKeys, devGetBlobList, failedKeys, MIN_RPC_TIMEOUT_MS));
        DS_ASSERT_TRUE(failedKeys.empty(), true);
        DS_ASSERT_OK(IsSameContent(devGetBlobList, devGetBlobList, 'b'));
        testIdx++;
    }
}

TEST_F(HeteroD2HTest, TestAllExist)
{
    std::vector<uint64_t> keyNums{ 450, 500 };
    std::vector<std::vector<double>> costs(keyNums.size());
    std::shared_ptr<HeteroClient> client;
    InitAcl(0);
    InitTestHeteroClient(0, client);
    auto blkSz = 1024u, blksPerObj = 1u;
    for (auto numOfObjs : keyNums) {
        LOG(INFO) << FormatString("start test MSetD2H Test ##### %lu KB", numOfObjs);
        std::vector<std::string> inObjectKeys, failedKeys;
        std::vector<DeviceBlobList> devGetBlobList, devSetBlobList;
        for (auto j = 0ul; j < numOfObjs; j++) {
            inObjectKeys.emplace_back(FormatString("key_%s", j));
        }
        PrePareDevData(numOfObjs, blksPerObj, blkSz, devGetBlobList, devSetBlobList, 0);

        if (numOfObjs == 450) {
            // First iteration: set 450 keys with 'b' data
            DS_ASSERT_OK(client->MSetD2H(inObjectKeys, devSetBlobList));
            DS_ASSERT_OK(client->MGetH2D(inObjectKeys, devGetBlobList, failedKeys, MIN_RPC_TIMEOUT_MS));
            DS_ASSERT_TRUE(failedKeys.empty(), true);
            DS_ASSERT_OK(IsSameContent(devGetBlobList, devSetBlobList, 'b'));
        } else {
            // Second iteration: try to set 500 keys with 'c' data
            // Change device memory to 'c'
            for (auto &blobList : devSetBlobList) {
                for (auto &blob : blobList.blobs) {
                    auto data = std::string(blob.size, 'c');
                    DS_ASSERT_OK(
                        AclDeviceManager::Instance()->MemCopyH2D(blob.pointer, blob.size, data.data(), blob.size));
                }
            }

            // MSetD2H semantics: if key exists, it will NOT update
            // Keys 0-449 already exist from first iteration (stay 'b')
            // Keys 450-499 are new and will be set with 'c'
            DS_ASSERT_OK(client->MSetD2H(inObjectKeys, devSetBlobList));

            // Verify the results
            DS_ASSERT_OK(client->MGetH2D(inObjectKeys, devGetBlobList, failedKeys, MIN_RPC_TIMEOUT_MS));
            DS_ASSERT_TRUE(failedKeys.empty(), true);

            // Keys 0-449 should still be 'b' (from first iteration, not updated)
            std::vector<DeviceBlobList> firstPartGet(devGetBlobList.begin(), devGetBlobList.begin() + 450);
            std::vector<DeviceBlobList> firstPartSet(devSetBlobList.begin(), devSetBlobList.begin() + 450);
            DS_ASSERT_OK(IsSameContent(firstPartGet, firstPartSet, 'b'));

            // Keys 450-499 should be 'c' (newly set in this iteration)
            std::vector<DeviceBlobList> secondPartGet(devGetBlobList.begin() + 450, devGetBlobList.end());
            std::vector<DeviceBlobList> secondPartSet(devSetBlobList.begin() + 450, devSetBlobList.end());
            DS_ASSERT_OK(IsSameContent(secondPartGet, secondPartSet, 'c'));
        }
    }
}

TEST_F(HeteroD2HTest, TestPartExist)
{
    std::vector<uint64_t> keyNums{ 450, 500 };
    std::vector<std::vector<double>> costs(keyNums.size());
    std::shared_ptr<HeteroClient> client;
    InitAcl(0);
    InitTestHeteroClient(0, client);
    auto testIdx = 0u, blkSz = 1024u, blksPerObj = 1u;
    for (auto numOfObjs : keyNums) {
        LOG(INFO) << FormatString("start test MSetD2H Test ##### %lu KB", numOfObjs);
        std::vector<std::string> inObjectKeys, failedKeys, partKeys;
        std::vector<DeviceBlobList> devGetBlobList, devSetBlobList, partDevBlobList;
        for (auto j = 0ul; j < numOfObjs; j++) {
            inObjectKeys.emplace_back(FormatString("key_%s", j));
        }
        PrePareDevData(numOfObjs, blksPerObj, blkSz, devGetBlobList, devSetBlobList, 0);
        std::vector<uint64_t> randIdxs = { 1, 5, 6, 7, 10, 33 };
        for (auto randIdx : randIdxs) {
            partKeys.push_back(inObjectKeys[randIdx]);
            partDevBlobList.push_back(devSetBlobList[randIdx]);
        }
        DS_ASSERT_OK(client->MSetD2H(partKeys, partDevBlobList));
        DS_ASSERT_OK(client->MSetD2H(inObjectKeys, devSetBlobList));
        DS_ASSERT_OK(client->MGetH2D(inObjectKeys, devGetBlobList, failedKeys, MIN_RPC_TIMEOUT_MS));
        DS_ASSERT_TRUE(failedKeys.empty(), true);
        DS_ASSERT_OK(IsSameContent(devGetBlobList, devGetBlobList, 'b'));
        testIdx++;
    }
}

TEST_F(HeteroD2HTestEvcit, SpillTest)
{
    inject::Set("worker.SubmitSpillTask", "sleep(10000)");
    std::shared_ptr<HeteroClient> c0, c1;
    size_t numOfObjs = 2200, blksPerObj = 1, blkSz = 1024000;
    std::vector<std::string> inObjectKeys, failedKeys;
    std::vector<DeviceBlobList> devGetBlobList, devSetBlobList;
    for (auto j = 0ul; j < numOfObjs; j++) {
        inObjectKeys.emplace_back(FormatString("key_%s", j));
    }
    auto retcode = -1;
    auto hook = [&retcode]() { exit(retcode); };
    Raii raii(hook);
    auto npu = 1;
    InitAcl(npu);
    InitTestHeteroClient(0, c0);
    PrePareDevData(numOfObjs, blksPerObj, blkSz, devGetBlobList, devSetBlobList, npu);
    for (auto i = 0uL; i < numOfObjs; i++) {
        DS_ASSERT_OK(c0->MSetD2H({ inObjectKeys[i] }, { devSetBlobList[i] },
                                 SetParam{ .writeMode = WriteMode::NONE_L2_CACHE_EVICT }));
    }
    c0.reset();
    retcode = 0;
    sleep(MINI_WAIT_TIME);
}

TEST_F(HeteroD2HTest, TestMSetD2HMsgWithInvalidDeviceId)
{
    const int32_t deviceId = 0;
    InitAcl(deviceId);
    std::shared_ptr<HeteroClient> client;
    InitTestHeteroClient(deviceId, client);

    const size_t numOfObjs = 1;
    const size_t blksPerObj = 1;
    const size_t blkSz = 1024;
    std::vector<DeviceBlobList> swapOutBlobList;
    std::vector<DeviceBlobList> swapInBlobList;
    PrePareDevData(numOfObjs, blksPerObj, blkSz, swapOutBlobList, swapInBlobList, deviceId);

    std::vector<std::string> keys{ GetStringUuid() };
    const int32_t invalidDeviceId = -1;
    for (auto &it : swapInBlobList) {
        it.deviceIdx = invalidDeviceId;
    }

    std::vector<std::string> failedKeys;
    Status s1 = client->MSetD2H(keys, swapInBlobList);
    LOG(INFO) << "MSetD2H Status: " << s1.ToString();
    ASSERT_TRUE(s1.IsError());
}

// Test for device memory leak when running sequential MSetD2H with NONE_L2_CACHE_EVICT
// This reproduces the issue where device memory is not released between operations
TEST_F(HeteroD2HTestEvcit, TestDeviceMemoryLeakSequential)
{
    std::shared_ptr<HeteroClient> client;
    InitAcl(0);
    InitTestHeteroClient(0, client);

    // Use large blob size close to device memory limit (400MB)
    // Default device memory is 500MB, so 400MB should work once but fail on second attempt if not released
    const size_t numOfObjs = 1;
    const size_t blksPerObj = 1;
    const size_t blkSz = 400 * 1024 * 1024;  // 400MB per blob

    LOG(INFO) << "Starting sequential MSetD2H test with large blobs (400MB each)";

    // First iteration - should succeed
    {
        std::vector<std::string> keys1;
        std::vector<DeviceBlobList> devGetBlobList1, devSetBlobList1;
        for (auto j = 0ul; j < numOfObjs; j++) {
            keys1.emplace_back(FormatString("test_seq_key1_%lu", j));
        }
        PrePareDevData(numOfObjs, blksPerObj, blkSz, devGetBlobList1, devSetBlobList1, 0);

        LOG(INFO) << "First MSetD2H call with 400MB blob";
        DS_ASSERT_OK(client->MSetD2H(keys1, devSetBlobList1, SetParam{ .writeMode = WriteMode::NONE_L2_CACHE_EVICT }));
        LOG(INFO) << "First MSetD2H succeeded";
    }

    // Small delay to allow cleanup
    sleep(1);

    // Second iteration - should also succeed if device memory is properly released
    // If device memory leaks, this will fail with "Out of memory"
    {
        std::vector<std::string> keys2;
        std::vector<DeviceBlobList> devGetBlobList2, devSetBlobList2;
        for (auto j = 0ul; j < numOfObjs; j++) {
            keys2.emplace_back(FormatString("test_seq_key2_%lu", j));
        }
        PrePareDevData(numOfObjs, blksPerObj, blkSz, devGetBlobList2, devSetBlobList2, 0);

        LOG(INFO) << "Second MSetD2H call with 400MB blob";
        Status s = client->MSetD2H(keys2, devSetBlobList2, SetParam{ .writeMode = WriteMode::NONE_L2_CACHE_EVICT });

        if (s.IsError()) {
            LOG(ERROR) << "Second MSetD2H failed - device memory leak detected: " << s.ToString();
            ASSERT_TRUE(false) << "Device memory not released between MSetD2H calls. Error: " << s.ToString();
        }
        LOG(INFO) << "Second MSetD2H succeeded - device memory properly released";
    }
}


// Test MSetD2H eviction behavior when memory usage reaches threshold
// This test replicates the Python test: test_func_ds_hertero_config_07
// Configuration: 560MB shared memory, 100MB eviction reserve threshold
// High water mark: max(560*0.8, 560-100) = max(448, 460) = 460MB
// Expected behavior: When total memory exceeds 460MB, eviction triggers
TEST_F(HeteroD2HTestEvictThreshold, TestEvictionAtThreshold)
{
    std::shared_ptr<HeteroClient> client;
    InitAcl(0);
    InitTestHeteroClient(0, client);

    const size_t deviceId = 0;
    const size_t blksPerObj = 1;

    // Step 1: Write 400MB data (keys_list1)
    const size_t blobSize1 = 400 * 1024 * 1024;  // 400MB
    const size_t numOfObjs1 = 1;
    std::vector<std::string> keys1;
    std::vector<DeviceBlobList> devGetBlobList1, devSetBlobList1;
    for (auto j = 0ul; j < numOfObjs1; j++) {
        keys1.emplace_back(FormatString("evict_test_key1_%lu", j));
    }
    PrePareDevData(numOfObjs1, blksPerObj, blobSize1, devGetBlobList1, devSetBlobList1, deviceId);

    LOG(INFO) << "Step 1: Writing 400MB data (keys_list1)";
    DS_ASSERT_OK(client->MSetD2H(keys1, devSetBlobList1,
                                 SetParam{ .writeMode = WriteMode::NONE_L2_CACHE_EVICT }));

    // Step 2: Write 10MB data (keys_list2)
    const size_t blobSize2 = 10 * 1024 * 1024;  // 10MB
    const size_t numOfObjs2 = 1;
    std::vector<std::string> keys2;
    std::vector<DeviceBlobList> devGetBlobList2, devSetBlobList2;
    for (auto j = 0ul; j < numOfObjs2; j++) {
        keys2.emplace_back(FormatString("evict_test_key2_%lu", j));
    }
    PrePareDevData(numOfObjs2, blksPerObj, blobSize2, devGetBlobList2, devSetBlobList2, deviceId);

    LOG(INFO) << "Step 2: Writing 10MB data (keys_list2)";
    DS_ASSERT_OK(client->MSetD2H(keys2, devSetBlobList2,
                                 SetParam{ .writeMode = WriteMode::NONE_L2_CACHE_EVICT }));

    // Step 3: Verify keys_list1 exists
    LOG(INFO) << "Step 3: Verifying keys_list1 exists";
    std::vector<bool> existResults1;
    DS_ASSERT_OK(client->Exist(keys1, existResults1));
    ASSERT_TRUE(existResults1[0]) << "keys_list1 should exist before eviction";

    // Step 4: Delete keys_list2
    LOG(INFO) << "Step 4: Deleting keys_list2";
    std::vector<std::string> failedKeys;
    DS_ASSERT_OK(client->Delete(keys2, failedKeys));

    // Step 5: Write 20MB data (keys_list3) - this should trigger eviction
    const size_t blobSize3 = 20 * 1024 * 1024;  // 20MB
    const size_t numOfObjs3 = 1;
    std::vector<std::string> keys3;
    std::vector<DeviceBlobList> devGetBlobList3, devSetBlobList3;
    for (auto j = 0ul; j < numOfObjs3; j++) {
        keys3.emplace_back(FormatString("evict_test_key3_%lu", j));
    }
    PrePareDevData(numOfObjs3, blksPerObj, blobSize3, devGetBlobList3, devSetBlobList3, deviceId);

    LOG(INFO) << "Step 5: Writing 20MB data (keys_list3) - should trigger eviction";
    DS_ASSERT_OK(client->MSetD2H(keys3, devSetBlobList3,
                                 SetParam{ .writeMode = WriteMode::NONE_L2_CACHE_EVICT }));

    // Small delay to allow eviction to complete
    sleep(2);

    // Step 6: Verify keys_list1 was evicted (no longer exists)
    LOG(INFO) << "Step 6: Verifying keys_list1 was evicted";
    std::vector<bool> existResults2;
    DS_ASSERT_OK(client->Exist(keys1, existResults2));
    ASSERT_FALSE(existResults2[0]) << "keys_list1 should be evicted after memory threshold exceeded";

    LOG(INFO) << "Test completed successfully - eviction triggered as expected";
}
}  // namespace st
}  // namespace datasystem
