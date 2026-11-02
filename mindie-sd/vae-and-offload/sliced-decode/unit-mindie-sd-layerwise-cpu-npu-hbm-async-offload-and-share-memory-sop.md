---
schema: mindie-graded-unit/1
id: unit-mindie-sd-layerwise-cpu-npu-hbm-async-offload-and-share-memory-sop
title: "MindIE-SD DiT 双流异步参数 Offload（storage.resize_）与单机多实例 IPC 共享内存规程"
tier: L2
category: sop
evidence_strength: software_contract
domain_path: mindie-sd/vae-and-offload/sliced-decode
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-mindie-sd-vae-tae-sliced-state-decode-and-offload-sop
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "Master timeout waiting for child processes ready"
    - "timeout waiting for share handle"
    - "Offload parameter validation failed"
  keywords:
    - enable_offload
    - min_reserved_blocks_count
    - ShareMemoryManager
    - init_share_memory
    - untyped_storage
    - safe_loads
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/offload.py
  source_digest: 50a668defb5d93c3d06659cec679c27c283c83f7e80349edff953b5585758ba1
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/offload.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/share_memory.py
  supersedes: []
revision: 1
---

## 适用场景与触发条件

当超大参数量扩散模型（如 Flux.1、Wan2.1-14B、Qwen-Image、HunyuanImage-3.0）单卡 HBM 无法同时驻留全部 Transformer Block 权重而需要启用 `mindiesd.offload.enable_offload`，或在单机多实例部署场景下使用 `mindiesd.share_memory`（`init_share_memory` / `ShareMemoryManager`）跨进程共享只读权重句柄以节省内存时，必须遵循本规程。

## 核心操作规程

1. **双流异步层间 Offload 机制（`mindiesd/offload.py:enable_offload`）**：
   - **入参校验**：`model` 必须为 `torch.nn.Module`，`blocks` 必须为非空 `torch.nn.ModuleList`，常驻块数 `min_reserved_blocks_count` 必须满足 `0 < min_reserved_blocks_count < len(blocks)`（默认值为 `2`）。
   - **Pin Memory 镜像与 `untyped_storage().resize_(0)` 原地释放**：
     - 初始化时，为前 `min_reserved_blocks_count` 个常驻块保留 NPU 权重；对其余非常驻块，在 CPU 上创建 `pin_memory()` 锁页内存副本保存权重数据，并在模型移至 NPU 后将这些非常驻块 NPU 参数的底层存储执行 `param.data.untyped_storage().resize_(0)`，立即将 NPU 显存占用降为 `0` 字节，同时保持 `Parameter` 对象引用不变。
   - **独立 `h2d_stream` / `d2h_stream` 与计算流流水线**：
     - **前向预钩子（Pre-Hook）**：在第 $i$ 个 Block 开始计算前，在专用 `h2d_stream` 上提前将后续待执行 Block 的权重执行 `untyped_storage().resize_(cpu_storage.size())` 并异步 `copy_(cpu_pin_param, non_blocking=True)`，与当前 Block 的 Cube/Vector 计算重叠。
     - **前向后钩子（Post-Hook）**：在非常驻 Block 完成前向计算后，等待计算流事件完成，立即将其 NPU 存储再次 `resize_(0)` 归还 HBM。
2. **多实例 ZMQ IPC 共享内存管理（`mindiesd/share_memory.py`）**：
   - 调用 `init_share_memory(instance_world_size, instance_id, master_addr="127.0.0.1", base_port=5555)` 初始化 `ShareMemoryManager`。
   - **端口映射与超时防挂死**：每个 `device_id = torch.npu.current_device()` 自动绑定独立的 `pub_port = base_port + device_id + 100` 与 `rep_port = pub_port + 1`；Master 实例（`instance_id == 0`）设置 `RCVTIMEO = 10000 ms` 等待所有从实例发送 `b"READY"` 并回复 `b"ACK"`，从实例设置 `RCVTIMEO = 5000 ms`，超时显式抛出 `TimeoutError`。
   - **安全反序列化铁律**：广播共享内存句柄时，必须使用 `safe_dumps` / `safe_loads`（基于白名单的 `SafeUnpickler`）序列化句柄，严禁使用不安全的 `socket.recv_pyobj()` 以防本地端口投毒导致远程代码执行（RCE）。
   - 进程退出时由 `atexit.register(ShareMemoryManager.cleanup)` 自动关闭所有 ZMQ 套接字（`linger=0`）并销毁 Context。

## 验证与反模式警报

- **验证判据**：开启 `enable_offload(model, model.blocks, min_reserved_blocks_count=2)` 后，NPU 静态权重显存降低约 $\frac{N_{\text{blocks}} - 2}{N_{\text{blocks}}}$，且 Profiler 中 `MemcpyAsync (H2D)` 落在独立 Stream 上与前一层计算重叠。
- **反模式警报**：
  - 严禁在 `enable_offload` 激活状态下于模型前向传播中途直接访问尚未换入层的 `param.data`（此时其 `untyped_storage().size() == 0`）。
  - 严禁在同一台机器同一 `device_id` 上并发启动两组共享相同 `base_port` 的 `ShareMemoryManager` 实例，否则会触发 ZMQ 端口冲突。
