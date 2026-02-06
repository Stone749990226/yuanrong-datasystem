# HeteroClient MGetH2D and MSetD2H 测试

## 概述

这个测试文件 ([hetero_client_h2d_d2h_test.cpp](hetero_client_h2d_d2h_test.cpp)) 提供了一个简单的测试，用于测试 HeteroClient 的 `MGetH2D` 和 `MSetD2H` 函数，这些函数用于在主机内存和昇腾 NPU 设备内存之间传输数据。

## 测试描述

测试执行完整的往返操作：

1. **初始化**: 设置 HeteroClient 和 ACL (Ascend Computing Language)
2. **MSetD2H (设备到主机)**:
   - 在 NPU 上分配设备内存
   - 用测试数据填充（模式 'A', 'B' 等）
   - 将数据从设备传输到主机缓存
3. **MGetH2D (主机到设备)**:
   - 分配新的设备内存
   - 从主机缓存检索数据回到设备
4. **验证**: 检查检索的数据是否与原始数据匹配
5. **清理**: 释放所有资源

## 构建和运行

### 前提条件

- 使用 hetero 支持构建 (`-X on`)
- 运行中的 datasystem worker
- 昇腾 NPU 硬件（或模拟环境）
- **重要**: 需要设置 `ASCEND_HOME_PATH` 或 `ASCEND_CUSTOM_PATH` 环境变量

### 启动基础设施

```bash
# 启动 ETCD
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://localhost:2379 &

# 启动 worker
dscli start -w --worker_address "127.0.0.1:31501" --etcd_address "127.0.0.1:2379"
```

### 构建和运行测试

```bash
# 使用 hetero 支持构建
cd /home/stone/learn/pytorch/yuanrong-datasystem
bash build.sh -X on -j 8

# 运行特定测试
cd build
ctest --timeout 80 -R "HeteroClientH2DD2HTest" --output-on-failure -V
```

## 解决 "undefined reference to aclInit" 错误

### 问题原因

如果您在编译时遇到 `undefined reference to aclInit` 错误，这是因为：

1. ACL 库（libascendcl.so）没有被正确链接
2. 需要确保 `BUILD_HETERO` 选项已启用
3. 需要正确设置 Ascend 环境变量

### 解决方案

#### 1. 确保环境变量设置正确

```bash
# 设置 Ascend 安装路径（选择其中一个）
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
# 或
export ASCEND_CUSTOM_PATH=/usr/local/Ascend/ascend-toolkit

# 验证 ACL 库存在
ls -l $ASCEND_HOME_PATH/lib64/libascendcl.so
ls -l $ASCEND_HOME_PATH/lib64/libhccl.so
```

#### 2. 使用正确的构建选项

```bash
# 必须使用 -X on 启用 hetero 支持
bash build.sh -X on -j 8
```

#### 3. CMakeLists.txt 修改（已完成）

我们已经修改了 `tests/ut/CMakeLists.txt` 文件，在第 34-37 行添加了：

```cmake
# Add ACL libraries for hetero tests
if (BUILD_HETERO)
    list(APPEND DS_UT_DEPEND_LIBS ${ASCEND_LIBRARY} ${HCCL_LIBRARY})
endif()
```

这确保了当 `BUILD_HETERO` 启用时，ACL 库会被链接到 UT 测试中。

### 验证链接

构建后，您可以验证 ACL 库是否正确链接：

```bash
cd build
ldd tests/ut/ds_ut | grep ascend
# 应该看到类似输出：
# libascendcl.so => /usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so
# libhccl.so => /usr/local/Ascend/ascend-toolkit/latest/lib64/libhccl.so
```

## 测试行为

### 有 NPU 硬件和运行中的 Worker
- 测试将完全执行
- 真实的设备内存分配和传输
- 数据完整性验证

### 没有运行中的 Worker
- 测试将跳过，消息："Cannot initialize client, worker may not be running"

### 没有 NPU 硬件
- 测试将跳过，消息："ACL initialization failed, NPU may not be available"

### 没有 BUILD_HETERO 标志
- 测试将跳过，消息："Test requires BUILD_HETERO to be enabled"

## 测试的关键操作

### 1. MSetD2H (设备到主机)
```cpp
Status MSetD2H(const std::vector<std::string> &keys,
               const std::vector<DeviceBlobList> &devBlobList,
               const SetParam &setParam = {});
```
- 将数据从 NPU HBM 传输到主机缓存
- 用于将 NPU 数据交换到主机内存

### 2. MGetH2D (主机到设备)
```cpp
Status MGetH2D(const std::vector<std::string> &keys,
               const std::vector<DeviceBlobList> &devBlobList,
               std::vector<std::string> &failKeys,
               int32_t subTimeoutMs);
```
- 将数据从主机缓存传输到 NPU HBM
- 用于将数据从主机交换到 NPU

## 测试数据

- **Key**: "test-key1"
- **Blobs**: 每个 key 2 个 blob
  - Blob 0: 1024 字节，填充 'A'
  - Blob 1: 2048 字节，填充 'B'
- **Device**: NPU 设备 0

## 预期输出

测试通过时：
```
[ RUN      ] HeteroClientH2DD2HTest.BasicRoundTripTest
[       OK ] HeteroClientH2DD2HTest.BasicRoundTripTest (XXX ms)
```

测试跳过时（无 worker/NPU）：
```
[ RUN      ] HeteroClientH2DD2HTest.BasicRoundTripTest
[  SKIPPED ] HeteroClientH2DD2HTest.BasicRoundTripTest (X ms)
```

## 常见问题

### Q: 为什么我遇到 "undefined reference to aclInit" 错误？
**A**: 这是链接错误，不是编译错误。确保：
1. 使用 `-X on` 构建
2. 设置了 `ASCEND_HOME_PATH` 环境变量
3. ACL 库文件存在于 `$ASCEND_HOME_PATH/lib64/`
4. `tests/ut/CMakeLists.txt` 已包含我们的修改

### Q: 测试文件应该放在哪里？
**A**: 放在 `tests/ut/client/hetero_client_h2d_d2h_test.cpp`，它会被编译到 `ds_ut` 可执行文件中。

### Q: 命名空间应该是什么？
**A**: 使用 `namespace datasystem { namespace ut { ... } }`，与其他 UT 测试保持一致。

## 注意事项

1. 这是一个简化的测试，专注于基本功能
2. 测试使用直接的 ACL API 调用（aclrtMalloc, aclrtMemcpy 等）
3. 测试后会正确清理内存
4. 如果不满足前提条件，测试会优雅地跳过
5. 通过比较检索的数据与原始模式来验证数据完整性
