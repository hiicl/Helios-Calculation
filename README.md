# Helios Calculation
一个内存中心式的、硬件感知的分布式 AI 计算平台

## 1. 架构定义 (Architecture Definition)
![Helios架构图](helios.drawio.svg)

Helios Calculation 采用一种创新的内存中心式 (Memory-Centric) 架构，其核心是一个拥有海量内存和超高带宽的控制器 (Controller)，以及一组部署在高性能计算节点（如 IBM AC922）上的轻量级代理 (Agent)。

该架构将整个计算集群（包括 Controller 的内存）抽象为一个统一的、分层的内存资源池，并通过 Controller 主导的、基于数据特性的智能调度，为上层的大规模并行训练框架（如 Megatron-LM）提供极致的性能和无与伦比的稳定性。

其核心是分片元数据驱动的两层分布式模型：

- **宏观分布式** (Controller 分片元数据调度): Controller 作为唯一的“主宰”，负责将模型和数据在逻辑上切分，并分发给物理上分散的 NUMA 容器。
- **微观分布式** (NUMA容器内分片执行): 在单个 NUMA 容器内部，Megatron-LM 框架负责对其管辖的多块 GPU 进行二次的、硬件层面的微观分布式。

### NUMA 容器生命周期 (NUMA Container Lifecycle)
Helios Calculation Agent 的核心职责是管理其所在节点上所有 NUMA 容器的完整生命周期。这个过程由 Controller 远程触发和协调，确保了资源分配的精确性和隔离性。

- **指令接收**: Agent 从 Controller 接收到一个启动容器的 RPC 请求，其中包含所有必要的参数。
- **镜像拉取**: Agent 调用本地的容器引擎（如 Docker/Podman），从指定的仓库拉取 Megatron-LM 运行环境镜像。
- **硬件绑定与启动**: Agent 根据指令中的参数，构造并执行一条精确的 docker run 命令，将容器实例严格地绑定在指定的硬件资源上。
- **状态监控**: 容器启动后，Agent 持续监控其健康状况、资源使用情况，并实时上报给 Controller。
- **生命周期管理**: Agent 负责容器的停止、重启和销毁，并清理相关资源。

## 2. 功能特性 (Features)
- **分片元数据调度体系** (核心特性)
  - 统一调度单元：分片元数据
  - 全生命周期管理：创建→预热→活跃→冷却→归档
- **元数据驱动决策** 
  - 基于热度/流动性/稳定性动态路由
  - 分片状态感知迁移
- **四层存储架构**:
  - **L1 缓存**: 节点内 GPU 显存，作为最高性能的稳定区
  - **L2 缓存**: 节点内 CPU 主机内存，用于存放高流动性数据
  - **L3 缓存**: Controller 内存池，全局共享的温/冷数据缓存
  - **L4 磁盘归档**: 长期未访问分片的持久化存储，释放内存资源
- **NUMA 为中心的资源管理**: 所有调度和资源绑定都以 serverId:numaId 为基本单位，确保极致的硬件亲和性。
- **无侵入集成**: Megatron-LM 只需调整加载路径与数据初始化流程，核心逻辑保持不变
- **数据特性感知调度**: 数据块被赋予“热度、温度、流动性、稳定性”等属性，策略引擎基于这些属性动态决定数据在三层缓存中的最佳驻留位置。
- **读写路径分离与动态路由**: 根据传输类型和数据特性，动态选择 RDMA (低延迟) 或 ZMQ (高带宽) 等最优传输路径。
- **三层高性能网络架构**:
  - **数据主干网 (Data Backbone)**: 基于非对称的 RoCE 网络，Controller 端配置海量带宽（如 10 x 双口 100G），而各计算节点配置标准带宽（1 x 双口 100G）。专用于 L3 缓存读取和 GDR 协调，通过上下行端口分离实现流量隔离。
  - **数据迁移网 (Migration Fabric)**: 基于每个 NUMA 节点专属的 200G QSFP 网卡，构建跨服务器的 L2 缓存（主机内存）数据迁移通道，完全旁路 Controller，极大降低其网络负载。
  - **管理网 (Management Plane)**: 基于低带宽的 2.5G RJ45 网口，负责心跳、遥测、评分等轻量级控制信令，确保核心数据网络不受干扰。
- **服务化与解耦**: Agent/Controller/CLI 职责分离，Controller 内部亦可进一步拆分为调度、状态管理、缓存等微服务。
- **统一管理界面**: 提供 Web UI 和独立的本地管理 CLI，兼顾全局监控与本地精细化调试。

## 3. 关键功能原理 (Key Mechanisms Explained)
### 3.1 Controller 主导的分布式初始化
#### 3.1.1 模型切片实现
- **集成Megatron-LM分区逻辑**：Controller加载完整模型后调用Megatron-LM的tensor/pipeline并行算法进行切片
- **分片元数据生成**：为每个分片创建包含偏移量、大小和校验和的元数据
- **分层存储**：切片结果存储在L3缓存服务中，按NUMA节点优化分布

#### 3.1.2 身份分配机制
- **全局rank生成**：创建唯一的三元组标识(PP_RANK, TP_RANK, DP_RANK)
- **容器启动指令**：通过Cap'n Proto RPC发送包含rank信息和L3地址的启动命令
- **环境变量注入**：Agent在启动容器时注入分布式角色参数

#### 3.1.3 数据拉取协议
- **请求格式**：包含全局rank和分片ID的ShardRequest
- **响应内容**：返回分片数据+校验和的ShardResponse
- **传输优化**：使用RDMA直接内存访问减少CPU开销

#### 3.1.4 L3缓存服务
```mermaid
graph TD
    A[RDMA连接] --> B[请求路由]
    B --> C{请求类型}
    C -->|GET| D[缓存查询]
    C -->|PUT| E[写入队列]
    D --> F[内存检索]
    F --> G{是否命中}
    G -->|是| H[RDMA传输]
    G -->|否| I[磁盘加载]
    I --> J[内存预热]
    J --> H
```

#### 3.1.5 容错机制
- **分片校验**：CRC32校验和验证数据完整性
- **自动重试**：最多3次重传失败分片
- **故障恢复**：容器异常时重新调度并拉取相同分片

二次分布式: Megatron-LM 在容器内部，将拉取到的模型分片，利用其原生能力，进一步分发到其管辖的多块 GPU 显存中。

### 3.2 分片元数据驱动的智能调度
这是 Helios Calculation 智能调度的核心。Controller 为每个分片元数据维护一组动态属性，策略引擎基于这些属性进行调度决策和生命周期管理：

```mermaid
graph TB
  A[分片访问] --> B[属性更新]
  B --> C[热度分析]
  B --> D[流动性追踪]
  B --> E[稳定性评估]
  C --> F[缓存决策]
  D --> G[迁移决策]
  E --> H[位置优化]
```

- **最后访问时间 (last_access)**: 记录分片最后一次被访问的Unix时间戳
- **热力评分 (HeatScore)**: 统一评分机制，计算公式：  
  \( \text{HeatScore} = \alpha \cdot \frac{N_i}{T} + \beta \cdot e^{-\frac{t_{\text{now}} - t_{\text{last}}}{\tau}} \)  
  其中：
  - \( N_i \): 时间窗口T内的访问次数
  - \( T \): 统计窗口(默认300秒)
  - \( \tau \): 时间衰减常数(默认120秒)
  - \( \alpha, \beta \): 权重系数(默认0.7, 0.3)
- **流动性 (Fluidity)**: 记录分片迁移次数
- **稳定性 (Stability)**: 多维评分(频率×模式×时间)

**分片稳定区与双重阈值策略**:
- **高水位 (>85%)** 
  当L1显存利用率过高时，触发迁移机制：
  - 按稳定性评分排序分片
  - 将低稳定性分片"驱逐"到L2/L3
  - 更新分片元数据状态为"已降级"

- **低水位 (<70%)**
  当L1显存利用率较低时：
  - 扫描L2/L3中高热度分片
  - 预取满足条件的分片到L1
  - 更新分片元数据状态为"已预热"

### 3.3 跳板机制与分片优化 (Plank Mechanism & Shard Optimization)
Helios-Calculation 的网络架构针对分片元数据访问实施智能优化策略：

#### 读写分离策略
- **直接读取（Read-Only）**  
  当容器需要从远端L3缓存获取数据时：
  - 优先使用IB/ROCE网络通道（GPUDirect RDMA）
  - 保持低延迟的同时最大化高速互连带宽
  
- **写入/更新（Write-Back）**  
  数据回写场景：
  - 使用IB/ROCE上行通道保证快速提交与一致性
  - 支持批量聚合写入减少网络请求

#### 迁移优化（Migration via Plank）
为缓解IB/ROCE通道压力：
```mermaid
graph TB
    A[分片访问统计] --> B{访问次数 > 阈值}
    B -->|是| C[触发迁移决策]
    C --> D[以太网迁移]
    D --> E[目标节点主机内存]
    E --> F[本地GPU直接访问]
    B -->|否| G[保持RDMA通道]
```
- **迁移触发条件**:
  - 跨服务器分片访问次数在时间窗口内超过阈值
  - 热点数据检测（同一分片多容器频繁访问）
  
- **迁移执行**:
  1. 通过以太网将数据迁移到目标节点主机内存
  2. 标记原分片状态为"已迁移"
  3. 后续访问直接使用本地内存副本

#### 阈值策略与回收
- **动态阈值调整**:  
  根据网络负载实时计算迁移阈值：  
  `阈值 = 基础值 × (1 + 当前RDMA利用率/100)`
  
- **分片回收机制**:
  - 当分片访问者分散在多个跨服务器NUMA容器时
  - 控制器下发回写命令，将分片迁回L3存储
  - 避免频繁迁移造成的网络抖动

### 3.4 读写路径分离与动态路由
为了将网络性能压榨到极致，Helios Calculation 对读写操作采用不同的优化策略。

```mermaid
graph LR
  R[读请求] -->|低延迟需求| RDMA[RDMA 通道]
  W[写请求] -->|高带宽需求| ZMQ[ZMQ 通道]
  C[设备间传输] -->|智能决策| PLANK[跳板机制]
```

- **读路径优化 (RDMA)**: 追求极致低延迟。对于频繁、单向的访问（无论数据大小），优先使用 RoCE 上的 RDMA 通道，将宝贵的低延迟资源留给最关键的操作。
- **写路径优化 (ZMQ)**: 追求最高带宽。由于写入操作本质是复制，对延迟不敏感，因此优先使用普通网卡（如 200G QSFP）上的 ZMQ+CRC 通道，利用其高带宽特性实现最高吞吐量，同时降低成本。
- **动态路由决策**: 完全基于传输类型（读/写）而非数据大小：
  - 读操作：无条件选择 RDMA 路径（最低延迟）
  - 写操作：无条件选择 ZMQ 路径（最高带宽）

### 3.5 分片状态迁移流程
Controller 监控分片状态并执行生命周期管理：

```mermaid
graph LR
  A[分片创建] --> B[预热]
  B --> C[活跃]
  C -->|访问频率↓| D[冷却]
  D -->|达到阈值| E[迁移]
  C -->|显存压力| F[降级→L2]
  E --> G[归档→L3]
  G -->|1小时无访问| H[过期→磁盘归档]
  H -->|再次访问| B
  G -->|再次访问| B
```

- **状态转换规则**:
  - **预热**: 新分片加载到最近访问节点
  - **活跃**: 高频访问分片保持在L1/L2
  - **冷却**: 低频访问分片标记为迁移候选
  - **归档**: 长期未访问分片存储到L3
  - **过期归档**: 分片在L3中1小时无访问后持久化到磁盘

## 4. Megatron-LM 调度与执行工作流
本章节详细描述一个 Megatron-LM 训练任务从提交到执行的完整生命周期，并阐明各组件的具体职责。

### 步骤 1: 任务提交 (Job Submission)
**执行者**: 用户  
**接口**: Web UI 或 RESTful API  
**操作**: 用户提交一个训练/推理任务
1. **可视化任务配置**:
   - 提供交互式界面，直接编辑YAML配置文件
   - 实时验证参数有效性并提供提示
   - 可视化展示模型架构和参数规模（根据配置生成示意图）
2. **配置模板管理**:
   - 支持保存/加载常用配置模板
   - 提供参数优化建议（基于历史任务数据）
   
例如：

```yaml
jobName: gpt3-175b-training
model:
  name: gpt3
  size: 175B
dataset:
  path: /path/to/my/dataset
parallelism:
  pipeline_parallel_size: 8
  tensor_parallel_size: 8
  data_parallel_size: 16
dockerImage: registry.my-company.com/megatron-lm:latest
```

### 步骤 2: Controller 调度决策 (Scheduling Decision)
**执行者**: Helios Calculation Controller (调度器服务)  
**操作**:
- **资源计算**: Controller 解析任务配置，计算出总共需要的 NUMA 容器数量 (e.g., 8 * 8 = 64 个流水线/张量并行实例)。
- **拓扑感知分配**: 调度器查询集群状态，寻找满足 64 个空闲 NUMA 槽位的物理服务器组合。它会优先将流水线并行 (Pipeline Parallel) 的各个阶段放置在不同的服务器上，以最大化网络隔离；同时将张量并行 (Tensor Parallel) 的实例放置在同一台服务器的不同 NUMA 节点上，以利用 X-Bus。
- **生成执行计划**: 调度器生成一个详细的执行计划，该计划为 64 个容器中的每一个都分配了唯一的全局 Rank、角色（PP Rank, TP Rank 等）、以及它将被部署到的物理位置 (serverId:numaId)。

### 步骤 3: Agent 执行容器化 (Container Execution)
**执行者**: Helios Calculation Agent  
**操作**:
- **接收指令**: 每个被选中的 Agent 从 Controller 接收到一个或多个启动容器的 RPC 指令。
- **构造命令**: Agent 根据指令，为每个容器构造一个精确的 docker run 命令。

**示例命令**:
```bash
docker run -d --rm \
  --gpus '"device=0,1,2,3"' \
  --cpuset-cpus="0-19" \
  --memory="256g" \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  -e GLOBAL_RANK=5 \
  -e PIPELINE_PARALLEL_RANK=0 \
  -e TENSOR_PARALLEL_RANK=5 \
  -e DATA_PARALLEL_RANK=0 \
  -e CONTROLLER_L3_CACHE_ADDRESS="rdma://10.0.0.1:5000" \
  -v /path/to/shared/data:/data \
  registry.my-company.com/megatron-lm:latest
```

**参数解释**:
- `--gpus`: 精确指定分配给此 NUMA 容器的 GPU 设备。
- `--cpuset-cpus`: 将容器进程严格绑定到此 NUMA 节点对应的 CPU 核心上。
- `--memory`: 限制容器可使用的主机内存。
- `--network=host`: 使用主机网络模式，以获得最低的网络延迟，方便 RDMA 和 ZMQ 通信。
- `-e ...`: 注入所有必要的环境变量，告知容器其在分布式训练中的角色。

### 步骤 4: 训练启动与监控 (Training & Monitoring)
**执行者**: Megatron-LM 容器 & Helios Calculation Agent  
**操作**:
- **分片元数据拉取**: 容器内的 Megatron-LM 进程启动后，根据环境变量连接 Controller L3 缓存，拉取属于自己的分片元数据。
- **二次分布式**: Megatron-LM 将模型分片加载到分配给它的多块 GPU 显存中。
- **开始训练**: 所有容器准备就绪后，开始协同进行分布式训练。
- **状态上报**: Agent 持续监控容器的运行状态（运行中、失败、完成）、GPU/CPU利用率，并通过管理网络将这些遥测数据上报给 Controller。

## 5. 建议目录树 (Proposed Directory Structure)
```plaintext
/Helios Calculation-platform
|
├── api/                      # Cap'n Proto 协议定义文件
|   └── v1/                   # API 版本控制
|       └── Helios Calculation.capnp
|
├── cmd/                      # 所有可执行程序的入口
|   ├── Helios Calculation-controller/    # Controller 服务主程序
|   ├── Helios Calculation-agent/         # Agent 服务主程序
|   └── Helios Calculation-cli/           # 本地管理 CLI 工具主程序
|
├── pkg/                      # 可被外部应用引用的公共库
|   ├── log/                  # 日志库
|   ├── config/               # 配置加载库
|   └── rpc/                  # RPC 客户端/服务端封装
|
├── internal/                 # 项目内部私有代码
|   ├── agent/                # Agent 服务的内部实现
|   |   ├── monitor/          # 硬件监控模块 (CPU, GPU, Network)
|   |   ├── executor/         # 容器执行与生命周期管理
|   |   └── server/           # Agent 的 RPC 服务端实现 (for Controller & CLI)
|   |
|   ├── controller/             # Controller 服务的内部实现
|   |   ├── scheduler/          # 调度器核心算法
|   |   ├── policy/             # 策略引擎 (数据迁移, 冷却等)
|   |   ├── state/              # 集群状态管理器
|   |   ├── metadata/           # 数据块元数据存储接口
|   |   ├── transport/          # 传输服务，协调网络路径
|   |   ├── partitioning/       # 模型切片模块
|   |   └── l3_service/         # L3缓存服务实现
|   |
|   ├── services/               # 作为独立微服务运行的组件
|   |   ├── l3_cache_service/   # L3 共享缓存服务 (RDMA K-V Store)
|   |   └── web_api_service/    # Web UI 的后端 API 服务
|   |
|   └── common/                 # 项目内部共享的工具和定义
|       ├── types/              # 核心数据结构定义
|       └── utils/              # 通用工具函数
|
├── deploy/                   # 部署相关的文件
|   ├── docker/               # 用于构建 Megatron-LM 容器的 Dockerfile
|   └── configs/              # 生产环境配置文件模板
|
├── web/                      # Web UI 前端代码 (React/Vue)
|
├── scripts/                  # 编译、测试、部署等辅助脚本
|
└── README.md                 # 本文档
```

## 6. 核心技术验证指标 (Core Technology Validation Metrics)
- **Numa 路径管理**:
  - 跨节点 RDMA 调度: 验证 Controller 能否根据拓扑和数据特性，正确发起 GDR-to-GDR 或基于跳板的 RDMA 传输。
  - 动态路径决策: 验证系统能否根据读/写类型和数据大小，自动选择 RDMA 或 ZMQ 路径。
  
- **智能数据迁移**:
  - 高频数据本地化: 验证被频繁访问的远程数据块，是否能被策略引擎自动迁移到计算任务所在的 NUMA 节点。
  - 流动性感知: 验证高流动性数据是否被优先保留在 L2 主机内存而非 L1 显存。
  - 双重阈值稳定区管理: 验证在显存利用率 >85% 和 <70% 时，能否正确触发数据的驱逐和预取机制。
  
- **性能基准**:
  - 读路径 (RDMA): 验证点对点访问延迟是否能稳定在 <2μs。
  - 写路径 (ZMQ): 验证大文件传输带宽是否能逼近物理网卡上限 (如 >20GB/s for 200G NIC)。
