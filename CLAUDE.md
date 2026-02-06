# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openYuanrong datasystem is a distributed heterogeneous cache system that provides multi-level caching (HBM/DRAM/SSD) for AI workloads, particularly optimized for NPU (Neural Processing Unit) scenarios. It's part of the larger openYuanrong Serverless computing ecosystem.

**Key capabilities:**
- Distributed cache across HBM (NPU memory), DRAM, and SSD
- Zero-copy shared memory access for local data
- Direct NPU-to-NPU data transfer via HCCS/RoCE
- Multi-language SDKs: Python, C++, Go, Java
- ETCD-based cluster management with consistent hashing

## Build Commands

### Standard Build
```bash
# Release build (default)
bash build.sh

```

### Testing
```bash
# Build and run all tests
bash build.sh -t run

```

### Code Quality
```bash
# Format C++ code (uses .clang-format)
find src/ -name "*.cpp" -o -name "*.h" | xargs clang-format -i

```

### Build Options
- `-X on/off`: Enable/disable heterogeneous object support (NPU HBM features)
- `-A on/off`: Enable RDMA/UCX transport
- `-M on/off`: Enable URMA framework
- `-S address/thread/undefined`: Enable sanitizers for debugging
- `-c on/html`: Enable code coverage

## Architecture Overview

### Component Hierarchy

**Client → Worker → Master (embedded)**

1. **Client** (`src/datasystem/client/`): Embedded in user applications, provides three main interfaces:
   - **Heterogeneous Object**: NPU HBM-based storage with device-to-device transfers
   - **KV**: High-performance key-value cache with shared memory
   - **Object**: Reference-counted object cache for distributed futures

2. **Worker** (`src/datasystem/worker/`): Core data management node (one per machine):
   - Manages local DRAM (shared memory) and optional SSD (spill directory)
   - Handles LRU eviction: DRAM → local SSD → remote L2Cache (OBS/SFS)
   - Coordinates inter-worker data transfers via RPC
   - Each worker has local `objectTable_` for cached objects

3. **Master** (`src/datasystem/master/`): Metadata management (embedded in workers):
   - **Distributed mode** (default, `enable_distributed_master=true`): Each worker embeds a Master instance
   - **Centralized mode** (testing only): Single Master node, other nodes are workers only
   - Uses consistent hashing (`HashRing`) to partition metadata across Masters
   - Each Master maintains `metaTable_` (sharded metadata) and manages object locations
   - Handles global reference counts, TTL expiration, and metadata replication (RocksDB)

4. **Cluster Management**: ETCD-based service discovery with `HashRing` for consistent hashing

### Storage Hierarchy

**Three-tier caching system:**

1. **DRAM (Shared Memory)**: Hot data cache, zero-copy access via `mmap`
   - Managed per-worker via `Arena` allocators
   - Client directly accesses worker's shared memory regions

2. **Local SSD (Spill)**: Warm data overflow storage
   - Configured via `--spill_directory` flag
   - Used when DRAM is full, managed by `WorkerOcEvictionManager`
   - Local to each worker, not shared

3. **L2Cache (Remote Storage)**: Cold data persistence
   - Shared across all workers (OBS/SFS remote storage services)
   - Configured via `--l2_cache_type` (obs/sfs/none)
   - Provides durability and cross-worker data recovery

### Memory Eviction Mechanism

**Write Modes** (`WriteMode` enum in `object/object_enum.h`):
- `NONE_L2_CACHE`: No L2 persistence, data lost if evicted from DRAM
- `WRITE_THROUGH_L2_CACHE`: Sync write to L2, safe to evict from DRAM
- `WRITE_BACK_L2_CACHE`: Async write to L2, eviction waits for L2 sync
- `NONE_L2_CACHE_EVICT`: **Evictable volatile data**, deleted entirely when evicted (no L2 backup)

**Eviction Triggers** (`worker/object_cache/worker_oc_eviction_manager.cpp`):
- **High Water Mark** (80%): Eviction starts when memory usage exceeds `max(total × 0.8, total - eviction_reserve_mem_threshold_mb)`
- **Low Water Mark** (60%): Eviction stops when memory usage drops below `total × 0.6`
- **Parameter**: `--eviction_reserve_mem_threshold_mb` (default: 10240 MB, range: 100-102400 MB)
  - Example: `--shared_memory_size_mb 560 --eviction_reserve_mem_threshold_mb 100`
    - High water: max(448, 460) = 460 MB → triggers eviction
    - Low water: 336 MB → stops eviction

**Eviction Algorithm** (Clock/Second-Chance):
- Objects added to `EvictionList` with counter (Q1=1 for unreferenced, Q2=2 for referenced)
- Eviction scans from `oldest_` pointer, decrements counters, evicts when counter=0
- **New objects protected**: Added to list tail with counter, unlikely to be evicted immediately
- **Async execution**: Eviction runs in background thread pool, doesn't block writes

**MSetD2H Idempotency** (`client/object_cache/object_client_impl.cpp:MSet`):
- Always checks key existence via `MultiCreate` RPC (even for small data sizes)
- Skips D2H copy and publish for existing keys → **prevents duplicate writes**
- Recent fix (commit df92dcff): Ensures existence check even when data < shm_threshold

**Eviction Actions by WriteMode**:
- `WRITE_THROUGH_L2_CACHE` / `WRITE_BACK_L2_CACHE`: Delete from DRAM, keep in L2
- `NONE_L2_CACHE_EVICT`: Delete entirely (Action::END_LIFE) via `DeleteNoneL2CacheEvictableObject`
- `NONE_L2_CACHE`: Retain in DRAM (Action::RETAIN), spill to local SSD if configured

### Data Location Discovery (Two-Layer Routing)

**How a worker finds where an object's data is stored:**

1. **Layer 1 - Metadata Routing** (object key → Master):
   - Each worker maintains local `tokenMap_` (consistent hash ring, ~1MB, read-only)
   - Hash the object key to determine which Master manages its metadata
   - `tokenMap_` is synchronized across all workers via ETCD watch

2. **Layer 2 - Data Location** (Master → Worker):
   - Query the responsible Master's `metaTable_` (sharded metadata)
   - Master returns object's storage locations (may have multiple replicas)
   - Worker fetches data directly from the storage worker via RPC

**Key insight**: `tokenMap_` (routing info) is small and globally replicated; `metaTable_` (actual metadata) is large and sharded.

### Communication Layers

**Client ↔ Worker:**
- Shared memory (`mmap`) for zero-copy data access
- Unix domain sockets for control messages
- Client mmaps worker's memory regions via file descriptors

**Worker ↔ Worker:**
- TCP/ZMQ with Protocol Buffers (current)
- RDMA/UCX support (in development, see `src/datasystem/common/rdma/`)

**NPU ↔ NPU (Heterogeneous Objects):**
- Direct HBM-to-HBM transfers via HCCS/RoCE
- P2P-Transfer library coordinates HCCL send/receive ordering
- Supports load balancing across NPU links

### Three Main Interfaces

1. **Heterogeneous Object** (`client/hetero_cache/`, `worker/object_cache/device/`):
   - `DevMSet/DevMGet`: Persistent device objects with metadata
   - `DevPublish/DevSubscribe`: One-time pub-sub (auto-delete after consumption)
   - `MSetD2H/MGetH2D`: Host-device data swapping
   - Use case: KVCache for LLM inference, model parameter distribution

2. **KV Interface** (`client/object_cache/`):
   - Zero-copy reads via shared memory
   - Batch operations: `MSet`, `MGet`, `MSetTx` (transactional)
   - TTL support and LRU eviction
   - Write-through/write-back/none persistence modes
   - Use case: Microservice state, checkpoint storage

3. **Object Interface** (`client/object_cache/`):
   - Global reference counting (`GIncreaseRef/GDecreaseRef`)
   - Nested object dependency tracking via `OCNestedManager`
   - Automatic replication for hot data (cross-node reads create local copies)
   - PRAM and Causal consistency models
   - Use case: Distributed futures programming model

### Key Abstractions

- **Arena/Allocator** (`common/shared_memory/`): Shared memory management with jemalloc backend
- **HashRing** (`worker/hash_ring/`): Consistent hashing for metadata partitioning (tokenMap_)
- **RpcChannel/RpcStub** (`common/rpc/`): RPC abstraction over ZMQ/RDMA
- **OCMetadataManager** (`master/`): Manages metaTable_ (sharded object metadata)
- **WorkerOcEvictionManager** (`worker/object_cache/`): LRU eviction across DRAM/SSD/L2Cache tiers
- **WorkerOcSpill** (`worker/object_cache/`): Local SSD spill management

## Code Style

### C++
- **Standard**: C++17
- **Formatting**: Google style via `.clang-format` (120 char line limit, 4-space indent)
- **Namespace**: All code in `datasystem` namespace
- **Logging**: Use `LOG(INFO)`, `LOG(ERROR)`, `LOG_IF(severity, condition)` macros
- **Error handling**: Status-based returns using `datasystem::Status` class
- **Headers**: Include guards with `#ifndef DATASYSTEM_PATH_FILE_H` pattern
- **Pointers**: Right-aligned (`int* ptr` not `int *ptr`)
- **Prohibited**: Never use `<regex>` (use re2 library instead - enforced by CMake check)

### Python
- **Standard**: Python 3.9+
- **Testing**: Use `unittest.TestCase`, descriptive test method names
- **Imports**: Absolute imports with `from __future__ import absolute_import`

### General
- **Copyright headers**: All source files must include Apache 2.0 license header
- **Naming**: snake_case for variables/functions, PascalCase for classes/types
- **Dependencies**: Minimize external dependencies, prefer standard library

## Development Workflows

### Deployment for Testing

The system requires ETCD and worker processes to be running:

```bash
# Start ETCD (in background)
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://localhost:2379 &

# Start worker using dscli (after pip install or build)
dscli start -w --worker_address "127.0.0.1:31501" --etcd_address "127.0.0.1:2379"

# Stop worker
dscli stop --worker_address "127.0.0.1:31501"
```

### Running Tests

**Python tests** require worker to be running:
- Test configuration reads from `output/datasystem/service/worker_config.json`
- Tests automatically start/stop workers via `start_all`/`stop_all` helpers

**C++ tests** automatically start workers via test fixtures:
- Test class hierarchy: `YourTest → DevTestHelper/OCClientCommon → ExternalClusterTest → ClusterTest`
- Startup sequence in `SetUp()`: Configure options → Start ETCD → Start Workers → Wait ready
- Each test gets isolated: data dir (`./ds/<TestCase>/`), socket dir (`/tmp/<random>/`), ETCD prefix, ports
- Heterogeneous object tests use `MSetD2H` (Device→Host write) and `MGetH2D` (Host→Device read)

### Adding New Features

1. Implement C++ core in `src/datasystem/client/` or `src/datasystem/worker/`
2. Add Protocol Buffer definitions in `src/datasystem/protos/` if needed
3. Add language bindings:
   - Python: `src/datasystem/pybind_api/`
4. Add tests in `tests/ut/` (C++), `tests/python/`, or `go/*_test.go`
5. Run full test suite: `bash build.sh -t run`

## Important Notes

- **Deployment modes**: System defaults to distributed Master mode (`enable_distributed_master=true`). Centralized mode is for testing only.
- **Storage tiers**: DRAM (shared memory) → Local SSD (spill) → L2Cache (OBS/SFS remote storage)
- **Heterogeneous object features** require NPU hardware and ACL library (Ascend Computing Language)
- **RDMA support** requires RDMA-capable NIC and rdma-core libraries
- **Build artifacts**: Output goes to `./output/`, build files to `./build/`
- **Packaging**: Build creates `yr-datasystem-v{VERSION}.tar.gz` in output directory
- **Multi-language**: Python SDK is built by default, Go/Java require `-G on`/`-J on` flags
- **Test labels**: Tests are labeled as `level0`, `level1`, `object`, `ut`, `st` for selective execution
- **Coverage**: Use `-c html` to generate HTML coverage report in `./coverage_report/`

## Common Issues

- **Shared memory errors**: Check `/dev/shm` has sufficient space
- **Worker connection failures**: Verify ETCD is running and worker is registered
- **NPU-related build failures**: Disable heterogeneous support with `-X off` if no NPU hardware
- **Test timeouts**: Increase timeout with `-m <seconds>` flag (default: 80s)
