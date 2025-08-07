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

## 2. 设计优势：调度与执行解耦 (Design Advantage: Decoupling of Scheduling and Execution)

Helios 采用调度层与执行层彻底解耦的设计，使得平台本身无需为容器内安装的任何 GPU 优化库（如 FlashAttention, Apex, xFormers）做出改变。

### 核心分工
- **调度层 (Helios)**: 专注于资源分配和容器调度，不关心容器内部实现
- **执行层 (容器)**: 完全封装运行环境和计算逻辑，独立于调度系统

### 关键优势
1. **零修改兼容**: 容器内安装任何 GPU 优化库都不需要修改 Helios 代码
2. **灵活适配**: 支持不同任务的专用镜像（训练/推理等）
3. **未来证明**: 新优化库只需更新容器镜像，无需平台改动
4. **职责分离**:
   - 平台团队专注调度效率和系统稳定性
   - 算法团队专注容器内计算优化

这种架构解耦是 Helios 扩展性和适应性的核心基础。

## 3. 功能特性 (Features)

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
  - **L4 硬盘归档**: 长期未访问分片的持久化存储，释放内存资源
- **NUMA 为中心的资源管理**: 所有调度和资源绑定都以 serverId:numaId 为基本单位，确保极致的硬件亲和性。
- **无侵入集成**: Megatron-LM 只需调整加载路径与数据初始化流程，核心逻辑保持不变
- **数据特性感知调度**: 数据块被赋予“热度、温度、流动性、稳定性”等属性，策略引擎基于这些属性动态决定数据在三层缓存中的最佳驻留位置。
- **读写路径分离与动态路由**: 根据传输类型和数据特性，动态选择 UCX (低延迟) 或 ZMQ (高带宽) 等最优传输路径。
- **三层高性能网络架构**:

  - **数据主干网 (Data Backbone)**: 基于UCX框架的IB/RoCE主干网，Controller端配置海量带宽（如10x双口100G），各计算节点配置标准带宽（1x双口100G）。专用于L3缓存服务，通过智能路由实现流量优化。
  - **数据迁移网 (Migration Fabric)**: 基于每个 NUMA 节点专属的 200G TCP/UDP 网卡，构建跨服务器的 L2 缓存（主机内存）数据迁移通道，完全旁路 Controller，极大降低其网络负载。
  - **管理网 (Management Plane)**: 基于低带宽的 2.5G RJ45 网口，负责心跳、遥测、评分等轻量级控制信令，确保核心数据网络不受干扰。
- **服务化与解耦**: Agent/Controller/CLI 职责分离，Controller 内部亦可进一步拆分为调度、状态管理、缓存等微服务。
- **统一管理界面**: 提供 Web UI 和独立的本地管理 CLI，兼顾全局监控与本地精细化调试。

## 4. 关键功能原理 (Key Mechanisms Explained)

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
- **传输优化**：使用UCX-RDMA协议减少CPU开销

#### 3.1.4 L3缓存服务

```mermaid
 graph TD 
     A[UCX连接] --> B[请求路由] 
     B --> C{请求类型} 
     C -->|GET| D[缓存查询] 
     C -->|PUT| E[写入队列] 
     D --> F[内存检索] 
     F --> G{是否命中} 
     G -->|是| H[UCX传输] 
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

  $$
  \text{HeatScore}_i = \alpha \cdot \frac{N_i}{T} + \beta \cdot e^{-\frac{(t_{\text{now}} - t_{\text{last}})}{\tau}}
  $$

  **参数说明**

  | 符号 | 含义 | 示例取值 |
  |------|------|----------|
  | \(N_i\) | 分片 \(i\) 在时间窗口 \(T\) 内的访问次数 | 37 次 |
  | \(T\) | 统计窗口（单位：秒） | 300 秒（5分钟） |
  | \(t_{\text{now}}\) | 当前时间（Unix时间戳） | 1721780000 |
  | \(t_{\text{last}}\) | 分片上一次被访问的时间 | 1721779980 |
  | \(\tau\) | 时间衰减常数（"保温时间"） | 120 秒 |
  | \(\alpha\) | 热度权重系数 | 0.7 |
  | \(\beta\) | 温度权重系数 | 0.3 |

  **公式解释**

  - 第一项 \(\frac{N_i}{T}\)：单位时间内的访问频率（"热度"）
  - 第二项 \(e^{-\frac{(t_{\text{now}} - t_{\text{last}})}{\tau}}\)：上次访问的"新鲜度"（"温度"）
  - 两者线性组合形成统一的HeatScore，兼顾访问频率和时效性

  **调优建议**

  | 场景 | 调整方向 |
  |------|----------|
  | 高频访问但不最近 → 权重偏向热度 | \(\alpha=0.9, \beta=0.1\) |
  | 关注最新数据 → 权重偏向温度 | \(\alpha=0.4, \beta=0.6\) |
  | 需要缓解频繁迁移 | 增大 \(\tau\) |

  **计算示例**

  给定参数：

  - \(N_i = 20\)（5分钟内访问20次）
  - \(T = 300\) 秒
  - \(t_{\text{now}} = 1721780000\)
  - \(t_{\text{last}} = 1721779900\)（100秒前）
  - \(\tau = 120\)
  - \(\alpha = 0.7, \beta = 0.3\)

  计算过程：

  - 热度部分：\(\alpha \cdot \frac{N_i}{T} = 0.7 \cdot \frac{20}{300} = 0.7 \cdot 0.0667 \approx 0.0467\)
  - 温度部分：\(\beta \cdot e^{-\frac{(t_{\text{now}} - t_{\text{last}})}{\tau}} = 0.3 \cdot e^{-\frac{100}{120}} = 0.3 \cdot e^{-0.833} \approx 0.3 \cdot 0.434 \approx 0.130\)
  - 热力评分：\(\text{HeatScore} \approx 0.0467 + 0.130 = 0.1767\)

  此值可用于缓存层级分配（L1/L2/L3）和迁移触发判断。
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

  - 使用IB/RoCE主干网（UCX）通道（GPUDirect RDMA）
  - 保持低延迟的同时最大化高速互连带宽
- **写入/更新（Write-Back）**

  数据回写场景：

  - 使用IB/RoCE主干网（UCX）上行通道保证快速提交与一致性
  - 支持批量聚合写入减少网络请求

#### 迁移优化（Migration via Plank）

 为缓解IB/RoCE主干网（UCX）压力：

```mermaid
 graph TB 
     A[分片访问统计] --> B{触发条件} 
     B -->|显存压力>85%| C[按稳定性排序分片] 
     B -->|跨服务器访问次数>阈值| D[触发迁移决策] 
     B -->|"热点数据(多节点访问)"| E[降级至L3] 
     C --> F[降级低稳定性分片至L2] 
     F --> G[标记分片状态为“降级→L2”] 
     D --> H[通过迁移网迁移] 
     H --> I[目标节点主机内存] 
     I --> J[标记分片状态为“已迁移”] 
     E --> K[标记分片状态为“降级→L3”] 
```

- **迁移触发条件**:
  - 跨服务器分片访问次数在时间窗口内超过阈值
  - 热点数据检测（同一分片多容器频繁访问）
- **迁移执行**:
  - 通过迁移网将数据迁移到目标节点主机内存
  - 标记原分片状态为"已迁移"
  - 后续访问直接使用本地内存副本

> **优化迁移说明**
> 
> - **降级→L2**: 当分片被其他服务器频繁访问时，系统将其预迁移到访问节点的主机内存（L2）。这种预迁移避免了频繁的跨网络跳板操作或占用IB/RoCE主干网（UCX）。
> 
> - **降级→L3**: 当分片同时被多个跨服务器节点频繁访问时（通常≥3节点），系统将绕过常规冷却流程，直接将其迁移至L3全局缓存。这种网络优化迁移可减少单节点的网络压力。
> 
> **降级机制特点**:
> 
> 1. 降级操作独立于冷却机制，权限优先级高于冷却
> 2. 降级触发仅基于访问频率（跨服务器访问）
> 3. L2降级优化单节点访问，L3降级优化多节点访问

#### 阈值策略与回收

- **动态阈值调整**:
  根据网络负载实时计算迁移阈值：
  `阈值 = 基础值 × (1 + 当前IB/RoCE主干网（UCX）利用率/100)`
- **分片回收机制**:
  - 当分片访问者分散在多个跨服务器NUMA容器时
  - 控制器下发回写命令，将分片迁回L3存储
  - 避免频繁迁移造成的网络抖动

### 3.4 UCX/ZMX分层网络架构（增强版）

#### 兼容性实现

- **CX-4硬件支持**：
  - 强制启用跳板路径（主机内存中转）
  - 通过PCIe Gen3实现GPU→主机内存→网卡路径
  - 最大带宽64Gbps（8x8Gb链路聚合）
- **CX-5+硬件优化**：
  - 自动检测启用RDMA直连
  - GPU→NIC Peer-to-Peer支持
  - 带宽利用率≥95%
- **统一抽象层**：
  ```cpp
  // 硬件检测路由模块 
  TransportRoute select_route(DeviceInfo dev) { 
    if (dev.nic_model >= "ConnectX-5")  
      return DIRECT_PATH; 
    else  
      return PLANK_PATH; // CX-4使用跳板 
  } 
  ```

#### 分层网络策略

 Helios Calculation 采用UCX/ZMQ双协议栈，严格遵循分层网络隔离策略：

```mermaid
 graph TB 
   subgraph 节点内部 
     L1[GPU显存] <-->|PCIe/NVLink| L2[主机内存] 
   end 
  
   subgraph 跨节点通信 
     L2 <-->|UCX的IB/RoCE主干网| L2_Other[其他节点] 
     L2 <-->|UCX的IB/RoCE主干网| L3[Controller] 
     L2 <-->|ZMQ的TCP/UDP迁移网| L2_Other 
   end 
```

- **节点内部交互 (L1↔L2)**:
  - 无网络需求，通过PCIe/NVLink直连
  - 延迟：<1μs，带宽：>100GB/s
- **L2↔L2 跨节点交互**:
  - 读路径：UCX的IB/RoCE主干网（自动选择最优协议）
  - 写路径：ZMQ协议的TCP/UDP迁移网（高带宽）
  - 典型场景：降级迁移到其他节点L2
- **L2↔L3 交互**:
  - 统一使用IB/RoCE主干网（UCX）
  - 读操作：UCX-RDMA协议
  - 写操作：UCX-Tagged Message协议
  - 典型场景：降级/冷却到L3
- **网络选择原则**:
  - 读操作无条件选择最低延迟路径（UCX自动优化）
  - 写操作无条件选择最高带宽路径（ZMQ用于迁移网，UCX用于主干网）
  - 迁移操作继承写路径网络选择
  - 智能路由：UCX根据硬件能力自动选择直接路径或跳板路径（详见3.7节）

### 3.5 支持弹性扩缩容的状态迁移协议

#### 节点动态管理

- **运行时扩容**：
  1. 新Agent注册到Controller
  2. Controller分配待迁移分片(HeatScore < 阈值)
  3. 源节点→新节点异步迁移
  4. 元数据原子切换(＜10ms中断)
- **节点缩容**：
  1. Controller标记节点为"排出中"
  2. 分片按稳定性评分迁移
  3. 容器优雅终止(超时强制回收)
  4. Agent从注册表移除

#### 状态迁移增强

 Controller 监控分片状态并执行生命周期管理，支持弹性扩缩容：

```mermaid
 graph LR 
   A[分片创建] --> B[预热] 
   B --> C[活跃] 
   C -->|访问频率↓| D[冷却] 
   D -->|达到阈值| M[迁移中] 
   M -->|复制完成| G[归档→L3] 
   C -->|显存压力| F[降级→L2] 
   C -->|多节点频繁访问| H[降级→L3] 
   M -->|复制期间| C 
  
   %% 降级路径样式 
   style F stroke:#ff9900,stroke-width:2px 
   style H stroke:#ff9900,stroke-width:2px 
   %% 冷却路径样式 
   style D stroke:#0066cc,stroke-width:2px 
   style M stroke:#0066cc,stroke-width:2px 
  
   G -->|1小时无访问| I[过期→磁盘归档] 
   I -->|再次访问| B 
   G -->|再次访问| B 
```

- **迁移路径与网络接口**:
- **降级→L2 (橙色)**:
  - 触发：显存压力/多节点访问
  - 网络：TCP/UDP迁移网（ZMQ）
  - 接口：跨节点L2↔L2写通道
  - 目标：目标节点主机内存
  - **降级→L3 (橙色)**:
    - 触发：≥3节点频繁访问
    - 网络：IB/RoCE主干网（UCX）
    - 接口：L2↔L3写通道
    - 目标：Controller内存池
  - **冷却迁移 (蓝色)**:
    - 触发：访问频率低于阈值
    - 网络：IB/RoCE主干网（UCX）
    - 接口：L2↔L3写通道
    - 目标：Controller内存池
- **状态转换规则**:
  - **预热**: 新分片加载到最近访问节点
  - **活跃**: 高频访问分片保持在L1/L2
  - **冷却**: 低频访问分片标记为迁移候选
  - **迁移中**: 分片正在迁移中，旧副本仍可读（详见3.6节）
- **降级→L2**: 当分片被其他服务器频繁访问时，通过TCP/UDP迁移网预迁移到目标节点主机内存
- **降级→L3**: 当分片被≥3节点频繁访问时，通过IB/RoCE主干网（UCX）直接降级至L3
- **冷却迁移**: 通过IB/RoCE主干网（UCX）迁移到L3

> **显存压力说明**
> 
> 显存压力指GPU显存资源紧张的状态。当L1缓存(GPU显存)利用率超过高水位阈值(>85%)时，系统会触发以下机制：
> 1. 按稳定性评分排序分片
> 2. 将低稳定性分片"驱逐"到L2
> 3. 更新分片元数据状态为"已降级"
> 
> **分片降级说明**
> 
> 1. 降级操作独立于冷却机制，权限优先级高于冷却
> 2. 降级触发仅基于访问频率（跨服务器访问）
> 3. L2降级是预迁移操作：先将数据迁移到目标节点主机内存
> - TCP/UDP迁移网络无法直接从GPU显存取数
> - 需要先将数据迁移到CPU主机内存
> - GPU可直接读取主机内存，避免频繁跨网络跳板操作
> 4. L3降级优化多节点频繁访问场景
> 5. 所有降级操作通过四阶段协议实现服务零中断（详见3.6节）

### 3.6 UCX协议下的无缝迁移实现

 基于UCX框架重构迁移协议，实现硬件感知的数据重定位：

#### 核心原则

- **先复制后清理**：复制完成前保持旧副本可读
- **原子元数据切换**：确保位置信息一致
- **状态机管理**：明确迁移生命周期

#### 四阶段流程

```mermaid
 sequenceDiagram 
     participant C as Controller 
     participant A1 as Agent@Server-1 
     participant L3 as L3缓存 
     participant R as 请求者 
    
     C->>A1: CopyDataToL3(block_A) 
     Note over C: 状态: MIGRATING 
     A1->>L3: 复制数据... 
    
     loop 复制期间 
         R->>C: 请求block_A位置 
         C->>R: Server-1(旧地址) 
         R->>A1: 读取block_A 
         A1-->>R: 返回数据 
     end 
    
     A1-->>L3: 复制完成 
     L3->>C: 确认复制 
     Note over C: 原子更新: 位置=L3, 状态=STABLE 
     C->>A1: CleanupData(block_A) 
     A1->>A1: 删除本地副本 
```

#### 阶段详解

1. **决策与准备**
   - 触发：迁移网带宽持续超阈值
   - 操作：
     - 状态更新为 `MIGRATING_TO_L3`
     - 发送 `CopyDataToL3`指令
2. **数据复制与网络选择**
   - **节点内部迁移 (L1→L2)**:
     - 无网络传输，内存直接拷贝
     - 延迟：<100ns
   - **跨节点L2迁移**:
     - 网络：TCP/UDP迁移网（200G ZMQ）
     - 协议：批量压缩传输
   - **L2→L3迁移**:
     - 网络：IB/RoCE主干网（100G UCX）
     - 协议：分片流式传输
   - **读请求处理**:
     - 查询返回旧位置
     - 源节点通过UCX-RDMA协议持续服务读请求
     - 迁移完成后自动切换新位置
3. **原子切换与确认**
   - 目标存储完成CRC校验后发送确认
   - Controller原子更新元数据：
     - 降级迁移：更新位置为L2/L3
     - 冷却迁移：更新位置为L3
     - 状态更新为 `STABLE_WARM_L2`或 `STABLE_WARM_L3`
4. **清理旧副本**
   - 发送 `CleanupData`指令
   - 源节点删除副本释放资源

## 5. Megatron-LM 调度与执行工作流 (Megatron-LM Scheduling and Execution Workflow)

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

## 6. 监控系统设计 (Monitoring System Design)

### 历史数据存储

- **存储架构**:
  ```mermaid
  graph LR 
    A[Agent指标] --> B[Controller聚合] 
    B --> C[RRD时序数据库] 
    C --> D[30天滚动存储] 
    D --> E[Web API] 
    E --> F[折线图可视化] 
  ```
- **关键指标**:
  - 网络吞吐：UCX/ZMQ 分路径统计
  - 分片热度：HeatScore 趋势分析
  - 节点负载：GPU显存/CPU利用率
- **压缩策略**:
  - 原始数据：1秒粒度存1小时
  - 5分钟聚合：存7天
  - 1小时聚合：存30天

## 7. 系统测试与验证 (System Testing and Verification)

这个“模拟测试”功能，可以在不消耗任何 GPU 计算资源、不启动真实 Megatron-LM 容器的情况下，验证从任务调度到数据链路的完整流程。它通常被称为“模拟测试 (Simulation Testing)”或“端到端模拟 (End-to-End Mocking)”。

### Helios 模拟测试 (Simulation Mode) 工作流

这个模式的核心是：Controller 启动一个“模拟”模式，它在内存中创建虚拟的 Agent 和容器，并模拟它们之间的 RPC 交互，然后将每一步的结果实时反馈给 Web UI。

#### 阶段一：启动与准备 (Initialization & Preparation)

- **触发测试**:
    - 执行者: 管理员
    - 接口: Web UI
    - 操作: 在 Web UI 上点击一个“启动系统模拟测试”的按钮。这个操作会向 Controller 的 API 发送一个 `start_simulation_test` 的请求。
- **Controller 进入模拟模式**:
    - 执行者: Helios Controller
    - 操作:
        - Controller 启动时，加载一个特殊的“模拟集群拓扑”配置，其中定义了若干虚拟的 AC922 服务器和 NUMA 节点。
        - Controller 在其状态管理器 (State Manager) 中创建这些虚拟节点的“模拟对象 (Mock Objects)”，并将它们的状态标记为“在线”。
        - L3 缓存服务被激活，并根据测试配置，在内存中预置 (pre-seed) 一些虚拟的测试分片，例如 `test_shard_001`, `test_shard_002`.

#### 阶段二：模拟调度与执行 (Simulated Scheduling & Execution)

- **模拟任务提交**:
    - 执行者: Controller (自动触发)
    - 操作: Controller 自动生成一个虚拟的 Megatron-LM 训练任务，并将其提交给自己的调度器 (Scheduler).
- **模拟调度**:
    - 执行者: 调度器
    - 操作: 调度器像处理真实任务一样，查询状态管理器中的（虚拟）节点资源，并生成一个执行计划，决定将虚拟容器分配给哪些虚拟的 `serverId:numaId`.
- **模拟 RPC 交互 (核心步骤)**:
    - 执行者: Controller (内部服务)
    - 操作: Controller 不会真的通过网络发送 RPC，而是进行内部的函数调用来模拟这个流程：
        a. 模拟 Agent 调用: Controller 的调度器调用一个模拟的 `Agent.StartContainer` 函数。
        b. 模拟容器启动: 这个模拟函数会立即触发下一步。
        c. 模拟数据拉取: 它会接着调用模拟的 `Megatron-LM.pull_data` 函数。这个函数会向真实的 L3 缓存服务发起一个 RPC 请求，尝试拉取预置的 `test_shard_001`.

#### 阶段三：结果验证与展示 (Verification & Visualization)

- **链路验证**:
    - 执行者: L3 缓存服务 & Controller
    - 操作:
        - 如果 L3 缓存服务成功收到了来自“模拟 Megatron-LM”的请求，并正确返回了 `test_shard_001` 的数据（或元数据），则证明从调度决策 -> 身份分配 -> 数据拉取的整个核心逻辑链路是通畅的。
        - Controller 记录下这次交互的成功状态。
- **Web UI 实时展示**:
    - 执行者: Web UI
    - 操作:
        - Web UI 通过 WebSocket 与 Controller 建立长连接。
        - Controller 在模拟过程中的每一步（调度成功、模拟 RPC 调用、L3 缓存命中等），都会通过 WebSocket 向 UI 推送一条结构化的日志消息。
        - UI 将这些日志实时地、格式化地展示在一个“测试日志”面板中，让管理员可以清晰地看到整个模拟流程的执行情况。

#### 流程图解

```mermaid
sequenceDiagram
    participant UI as Web UI
    participant C as Controller (Simulation Mode)
    participant L3 as L3 Cache Service

    UI->>C: API: start_simulation_test()
    Note over C: 创建虚拟节点, 预置测试分片到L3

    C->>C: 1. 自动生成虚拟任务
    C->>C: 2. 调度器分配虚拟容器
    C->>L3: 3. 模拟容器RPC: GetData(test_shard_001)

    L3-->>C: 4. 返回测试分片数据 (成功)
    Note over C: 核心数据链路验证通过!

    C-->>UI: WebSocket: "调度成功..."
    C-->>UI: WebSocket: "模拟RPC调用..."
    C-->>UI: WebSocket: "L3缓存命中..."
    C-->>UI: WebSocket: "测试完成: 成功"
```

#### 此测试策略的优势

- **极速反馈**: 整个测试在内存中完成，通常只需要几秒钟，远快于启动真实集群。
- **零资源消耗**: 无需占用任何宝贵的 GPU 或计算节点资源。
- **CI/CD 友好**: 这个测试可以作为自动化集成测试的一部分，在每次代码提交后自动运行，确保核心逻辑没有被破坏。
- **覆盖核心逻辑**: 它精准地验证了系统中最复杂、最关键的部分——调度与数据分发流程。


## 8. 参数调优指南 (Parameter Tuning Guide)

### 热力评分参数
热力评分公式中的参数 α（热度权重系数）、β（温度权重系数）和 τ（时间衰减常数）作为平台高级配置项，位于Controller的配置文件（controller.yaml）中。平台管理员可根据集群特性和负载模式调整这些参数。

#### 配置示例
```yaml
# controller.yaml
heat_score:
  alpha: 0.7   # 范围: 0.0~1.0, 默认0.7
  beta: 0.3    # 范围: 0.0~1.0, 默认0.3
  tau: 120     # 单位: 秒, 默认120
```

#### 调优建议
| 场景 | 调整方向 | 预期效果 |
|------|----------|----------|
| 高频访问场景 | 提高 α (0.8~0.9) | 更关注访问频率 |
| 实时性要求高 | 提高 β (0.5~0.7) | 更关注数据新鲜度 |
| 减少频繁迁移 | 增大 τ (>300) | 延长热度衰减时间 |

## 9. 网络选择逻辑 (Network Selection Logic)
Helios根据传输操作类型和硬件能力自动选择最优网络路径，核心原则：
- **读操作**：无条件选择最低延迟路径（UCX自动优化）
- **写操作**：无条件选择最高带宽路径（ZMQ用于迁移网，UCX用于主干网）

#### 决策流程图
```mermaid
graph TB
  A[传输请求] --> B{操作类型}
  B -->|读操作| C[UCX路径]
  B -->|写操作| D[ZMQ路径]
  C --> E{网卡型号}
  E -->|ConnectX-5+| F[直接路径]
  E -->|ConnectX-4| G[跳板路径]
  D --> H[迁移网通道]
```

#### 路径性能指标
| 路径类型 | 延迟 | 带宽 | 适用场景 |
|----------|------|------|----------|
| UCX直接路径 | ≤1.2μs | 最高95% | CX-5+读操作 |
| UCX跳板路径 | ≤2.2μs | 最高85% | CX-4读操作 |
| ZMQ迁移路径 | ≤5ms | ≥170Gbps | 数据迁移 |

## 10. 硬件适配规范与跳板机制 (Hardware Adaptation Specifications and Plank Mechanism)

 **CX-4网卡能力矩阵**：

 | 特性 | 支持状态 | 性能影响 | 解决方案 |
 |------|----------|----------|----------|
 | GPU→NIC RDMA Write | ✅ 稳定支持 | 延迟≤1.5μs | IB/RoCE主干网（UCX） |
 | NIC→GPU RDMA Read | ⚠️ 平台依赖 | +0.5-1μs延迟 | TCP/UDP迁移网（ZMQ） |
 | GPU Peer-to-Peer | ❌ 不支持 | N/A | NUMA优化 |
 | PCIe Gen4 | ❌ 不支持 | 最大64Gbps | 链路聚合 |
 | 动态重绑定 | ❌ 不支持 | 手动配置 | 静态绑定 |
 | Unified Memory | ⚠️ CUDA≥11.4 | 带宽降30% | 选择性启用 |

 **跳板机制核心逻辑**：

```mermaid
 graph TB 
   A[网络操作] --> B{网卡能力检测} 
   B -->|高级功能支持| C[直接路径] 
   B -->|功能受限| D[跳板路径] 
   C --> E[GPU直连NIC] 
   D --> F[主机内存中转] 
   E & F --> G[完成传输] 

   subgraph 自动适应 
     H[硬件扫描] --> I{ConnectX≥5?} 
     I -->|是| J[启用RDMA Read] 
     I -->|否| K[强制跳板模式] 
   end 
```

 **路径选择原则**：

1. **写操作(GPU→NIC)**:
   - 无条件使用直接路径
   - 最大化利用GDR优势
2. **读操作(NIC→GPU)**:
   - 检测网卡型号和驱动
   - ConnectX-5及以上：尝试直接路径
   - ConnectX-4及以下：强制跳板路径
   - 失败时自动降级

 **UCX统一抽象层**:
- 提供一致API接口
- 内部自动路由选择
- 实时性能监控
- 无缝支持硬件升级

## 11. 建议目录树 (Proposed Directory Structure)

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
