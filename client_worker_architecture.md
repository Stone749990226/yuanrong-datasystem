# Client-Worker架构详解

## 进程部署模型

### 典型部署场景

```
机器1 (训练节点)                    机器2 (数据节点)
┌─────────────────────────┐        ┌─────────────────────────┐
│  用户训练程序进程        │        │  Worker进程             │
│  ┌───────────────────┐  │        │  ┌──────────────────┐   │
│  │ Python训练脚本    │  │        │  │ Worker Service   │   │
│  │                   │  │        │  │                  │   │
│  │ ┌───────────────┐ │  │        │  │ • 内存管理       │   │
│  │ │ Client库      │ │  │  RPC   │  │ • 请求处理       │   │
│  │ │ (嵌入式)      │◄─┼──┼────────┼─►│ • LRU驱逐        │   │
│  │ │               │ │  │  共享  │  │ • 数据持久化     │   │
│  │ │ ClientWorker  │ │  │  内存  │  │                  │   │
│  │ │ Api类         │ │  │        │  │ Master(嵌入)     │   │
│  │ └───────────────┘ │  │        │  └──────────────────┘   │
│  └───────────────────┘  │        │                         │
│                         │        │  /dev/shm (共享内存)    │
└─────────────────────────┘        └─────────────────────────┘

机器3 (推理节点)                    机器4 (数据节点)
┌─────────────────────────┐        ┌─────────────────────────┐
│  推理服务进程            │        │  Worker进程             │
│  ┌───────────────────┐  │        │  ┌──────────────────┐   │
│  │ FastAPI服务       │  │        │  │ Worker Service   │   │
│  │                   │  │        │  │                  │   │
│  │ ┌───────────────┐ │  │        │  │ • 内存管理       │   │
│  │ │ Client库      │ │  │  RPC   │  │ • 请求处理       │   │
│  │ │ (嵌入式)      │◄─┼──┼────────┼─►│ • LRU驱逐        │   │
│  │ │               │ │  │        │  │ • 数据持久化     │   │
│  │ │ ClientWorker  │ │  │        │  │                  │   │
│  │ │ Api类         │ │  │        │  │ Master(嵌入)     │   │
│  │ └───────────────┘ │  │        │  └──────────────────┘   │
│  └───────────────────┘  │        │                         │
└─────────────────────────┘        └─────────────────────────┘
                                            ↕
                                    Worker间通信(TCP/RDMA)
```

## 关键点说明

### 1. Client不是独立进程
- Client是一个**库**（libdatasystem.so），链接到用户程序
- 用户程序调用Client库的API
- Client库内部使用`ClientWorkerApi`类与Worker通信

### 2. Worker是独立进程
- Worker是一个**守护进程**，独立运行
- 通常每台机器运行一个Worker进程
- Worker管理该机器的内存资源

### 3. 同机部署场景
一台机器可以同时运行：
- 用户程序（嵌入Client库）
- Worker进程

```
同一台机器
┌─────────────────────────────────────────┐
│  用户程序进程                            │
│  ┌─────────────┐                        │
│  │ Client库    │                        │
│  │ (嵌入式)    │                        │
│  └──────┬──────┘                        │
│         │ Unix Socket + 共享内存        │
│         ↓                                │
│  ┌─────────────┐                        │
│  │ Worker进程  │                        │
│  │ (独立)      │                        │
│  └─────────────┘                        │
└─────────────────────────────────────────┘
```

**优势**：
- 零拷贝：通过共享内存（mmap）直接访问数据
- 低延迟：Unix domain socket通信
- 高效：避免网络传输开销

### 4. 跨机部署场景
Client和Worker在不同机器：

```
机器A                          机器B
┌─────────────┐               ┌─────────────┐
│ 用户程序    │               │ Worker进程  │
│ ┌─────────┐ │   TCP/RPC     │             │
│ │Client库 │◄┼───────────────┼►│           │
│ └─────────┘ │               │             │
└─────────────┘               └─────────────┘
```

**特点**：
- 通过TCP/ZMQ进行RPC通信
- 数据通过网络传输（或RDMA）
- 延迟较高，但支持分布式部署

## ClientWorkerApi类的作用

### 命名解析
```
ClientWorkerApi
    ↓
Client端的、用于与Worker通信的、Api类

不是：Client和Worker的混合体
而是：Client库中负责与Worker通信的模块
```

### 类的职责
```cpp
// 在Client库内部
class ClientWorkerApi {
    // 负责与Worker进程通信
    Status MSet(const string& key, const Buffer& data) {
        // 1. 序列化请求
        // 2. 通过RPC发送给Worker
        // 3. 等待Worker响应
        // 4. 返回结果给用户
    }

    Status MGet(const string& key, Buffer& data) {
        // 1. 发送Get请求给Worker
        // 2. Worker返回数据位置（共享内存FD）
        // 3. mmap映射共享内存
        // 4. 零拷贝读取数据
    }
};
```

### 为什么叫ClientWorker而不是Client？
因为Client库中有多个模块：
- `ClientWorkerApi`：与Worker通信
- `ClientMasterApi`：与Master通信（如果有独立Master）
- `ClientLocalCache`：本地缓存管理
- `ClientConnectionPool`：连接池管理

`ClientWorkerApi`明确表示这是"Client中负责与Worker交互的部分"。

## 通信方式

### 1. 同机通信（Client和Worker在同一台机器）
```
Client进程                    Worker进程
    │                             │
    │  1. 注册请求 (Unix Socket)  │
    ├─────────────────────────────►
    │                             │
    │  2. 返回共享内存FD          │
    ◄─────────────────────────────┤
    │                             │
    │  3. mmap映射共享内存        │
    │                             │
    │  4. 直接读写共享内存        │
    │     (零拷贝)                │
    │                             │
```

### 2. 跨机通信（Client和Worker在不同机器）
```
Client进程 (机器A)            Worker进程 (机器B)
    │                             │
    │  1. MSet请求 (TCP/RPC)      │
    ├─────────────────────────────►
    │     + 数据payload            │
    │                             │
    │  2. 响应 (TCP/RPC)          │
    ◄─────────────────────────────┤
    │                             │
```

## 代码示例

### Python用户代码
```python
import datasystem

# 创建Client对象（Client库嵌入在这里）
client = datasystem.Client("127.0.0.1:31501")

# 调用MSet，内部会：
# 1. Python绑定调用C++ Client库
# 2. Client库中的ClientWorkerApi类处理请求
# 3. ClientWorkerApi通过RPC与Worker通信
# 4. Worker处理请求并返回结果
client.MSet("key1", b"data")

# 调用MGet
data = client.MGet("key1")
```

### C++内部实现
```cpp
// src/datasystem/client/object_cache/client_worker_api.cpp
Status ClientWorkerRemoteApi::Publish(...) {
    // 这是Client库内部的代码
    // 负责与Worker进程通信

    PublishReqPb req;
    req.set_object_key(objectKey);
    // ... 设置其他字段

    // 通过RPC stub发送请求给Worker
    PublishRspPb rsp;
    Status status = rpcSession_->Publish(req, &rsp);

    return status;
}
```

### Worker端处理
```cpp
// src/datasystem/worker/object_cache/worker_service.cpp
Status WorkerService::Publish(const PublishReqPb& req, PublishRspPb* rsp) {
    // 这是Worker进程的代码
    // 接收Client的请求并处理

    string objectKey = req.object_key();
    // ... 处理数据存储

    rsp->set_status(StatusCode::OK);
    return Status::OK();
}
```

## 总结

| 概念 | 定义 | 形态 | 位置 |
|------|------|------|------|
| **Client** | 客户端库 | 嵌入式库（.so/.a） | 用户程序进程内 |
| **Worker** | 数据管理服务 | 独立守护进程 | 独立进程 |
| **ClientWorkerApi** | Client中与Worker通信的模块 | C++类 | Client库内部 |

**关键理解**：
- ❌ ClientWorker ≠ 一个节点既做Client又做Worker
- ✅ ClientWorkerApi = Client库中负责与Worker通信的API类
- Client是库，Worker是进程，它们是分离的
- 一台机器可以同时运行用户程序（含Client库）和Worker进程
