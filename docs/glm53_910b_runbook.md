# GLM-5.3-Flash 在 Atlas 800 A2（910B）上的部署手册

> 适用环境：2 × Atlas 800 A2（每台 8 × 910B1，64 GB HBM）、CANN 9.2.0、GPUStack 2.2.3、
> 上游网关 New API（`maas.ovaijisuan.com`）。
> 本文所有数字均为实测，非估算。

## 0. 结论速览

当前生产配置（GPUStack 模型 `glm-5.3-flash-w4a8`）：

| 项 | 值 |
| --- | --- |
| 副本 | 2（一台一个，各占 8 卡） |
| 权重 | `/root/Models/japanufo-GLM53-w4a8-Ascend-Mixed`（302 GB） |
| 镜像 | `swr.cn-central-221.ovaijisuan.com/whcs-fae/vllm-ascend:glm53-kda` |
| 后端版本 | `910b-glm53-kda-custom`（vLLM 自定义版本，custom_framework=cann） |
| 上下文 | **900 000**（KV 955 024 tokens，并发 1.06×） |
| 图片 / 视频 | `image×16`、`video×1` |
| 批处理 | `max-num-batched-tokens=32768`、`max-num-seqs=16` |
| 显存 | `gpu-memory-utilization=0.94` |
| 投机解码 | MTP，3 tokens |
| KV 池 | Mooncake（AscendStoreConnector，`kv_role=kv_both`） |
| 并行 | TP8 + EP（单机内） |

关键指标：单流 TPOT **15.4 ms**（decode ≈ 65 t/s）；冷 TTFT 1K **434 ms** / 8K **1.27 s** /
32K **4.06 s** / 131K **17.7 s**。

## 1. 环境与硬件

| 项 | 值 |
| --- | --- |
| 节点 | `910b01` 10.0.0.142、`910b02` 10.0.0.240 |
| NPU | 8 × 910B1 / 台，单卡 60.96 GiB HBM |
| 驱动 / CANN | 25.5.1 / 镜像内 9.2.0 |
| 跨机互联 | NPU 侧 200GE RoCE ↔ ToR，HCCL allreduce ≈ 21.8 GB/s |
| 主机网卡 | `enp67s0f0`（`enp189s0f*` 无载波，忽略） |
| 模型盘 | NFS `100.125.40.2:/MaaS_Models` → `/root/Models`（两机共享） |

HCCL 约定：`HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` 均为
`enp67s0f0`。**单机部署无需 `HCCL_IF_IP`**；跨机时必须设为本机**主机 IP**（不是 NPU IP）。

## 2. 镜像：自建 app 层（不改 csrc）

上游 nightly 缺 GLM-5.3 的部分性能补丁，因此采用「官方 nightly 底座 + 薄 Python 层」的出包方式，
与 `vllm-backport` 的 base/app 分层思路一致，但因为本仓不改 csrc，**无需自建 base**。

| 项 | 值 |
| --- | --- |
| 底座 | `quay.io/ascend/vllm-ascend:nightly-main` = commit `4e6eac4e4`（2026-10-23） |
| 底座内版本 | vllm `0.31.0`，vllm-ascend `0.19.1rc2.dev3033+g4e6eac4e4` |
| 产出 | `swr.cn-central-221.ovaijisuan.com/whcs-fae/vllm-ascend:glm53-kda` |
| 分支 | `feat/glm53-kda-projection`（`ae891e7ef` 补丁、`f539195b8` CI） |
| 镜像内自检 | `vllm_ascend/models/glm5next/kda_projection.py` 存在、`fg_b_proj=True`、旧 `self.g_b_proj` 已移除 |

### 2.1 补丁来源

上游 PR 状态（本次核对）：

| PR | 内容 | 状态 | 底座是否已含 |
| --- | --- | --- | --- |
| #16936 | GLM-5.3 MRV2 MTP capture lifecycle | 已合并（2026-10-18） | **是** |
| #17542 | Triton KPool + gated norm | 已合并（2026-09-30） | **是** |
| #17012 | KDA f/g 投影合并为一次 `torch.bmm` | **CLOSED（冲突未解决）** | **否** |

即：底座已含前两个，只需补 #17012 的三个文件。

### 2.2 移植 #17012 时的两个坑

1. **PR 引用了底座不存在的模块**：补丁头部 `from vllm_ascend.models.glm5next.config import Glm5NextConfig`，
   而 `4e6eac4e4` 没有 `config.py`（该模块是更晚的主线才拆出来的）。
   处理：保留仓库原有的版本感知导入（`vllm_version_is("0.31.0")` 分支），只新增 `KDAFGProjection`。
2. **不要跟着 PR 删除新版代码**：PR 基线较旧，冲突处（构造函数中 `g_b_proj` 段）在新版里还包含
   `register_ascend_fla_customop()` 注册逻辑，必须保留。判断原则是「删除被合并掉的投影，保留新版新增的注册」。

### 2.3 出包流水线

`.github/workflows/build-ascend-app.yml` + `docker/Dockerfile.ascend_app`，两道防漂移 guard：

- **guard 1**：`csrc/ CMakeLists.txt cmake setup.py requirements.txt pyproject.toml docker/Dockerfile`
  自 `base_commit` 起若有改动 → 拒绝出包（提示必须先重建底座）。
- **guard 2**：拉底座镜像读取 `pip show vllm-ascend` 版本串，**必须包含 `+g<base_commit 短哈希>`**。
  这是分层出包最隐蔽的风险：Python 是新的、`.so` 是旧的，症状是运行期数值错误而非构建失败。

## 3. 权重：三种来源的取舍

| | `macdoor/GLM-5.3-Flash-w4a8` | `japanufo/…-Ascend` | `japanufo/…-Ascend-Mixed` |
| --- | --- | --- | --- |
| 体积 / 分片 | 180 GB / 48 | 191 GB / 50 | **302 GB / 74** |
| 配方 | 全量 W4A8 | 全量 W4A8（v5） | **13 层 FP16 + 4 层 W8A8 + 29 层 W4A8** |
| FP16 层 | — | — | `0,1,2,7,11,15,19,23,27,31,35,39,43` |
| 整层 W8A8 | — | — | `3,12,44,45` |
| 每卡权重（TP8） | ≈21 GiB | ≈22 GiB | **36.15 GiB（实测）** |
| 必备文件 | 齐 | 齐 | **缺 3 个（需重建）** |

每层内部：`mlp.shared_experts.{gate,up,down}_proj` 三件套始终是 **W8A8**，其余 864 个 routed expert 是 **W4A8**
（符合官方量化配方「dense/shared 走 W8A8、routed 走 W4A8」）。

### 3.1 重建 Mixed 缺失的三个文件

ModelScope 上 Mixed 仓库**确实没有** `quant_model_description.json`、
`quant_model_weights.safetensors.index.json`、`image_processing_glm5_next.py`。重建方法：

```bash
# 1) image_processing_glm5_next.py：三个仓库内容完全一致，直接复制
cp <任一参考仓库>/image_processing_glm5_next.py <mixed>/

# 2) index.json：扫描本仓分片头（8 字节长度 + JSON），生成 name -> shard 映射
#    {"metadata": {"total_size": <字节>}, "weight_map": {...}}

# 3) description.json：按张量逐个判定
#    - BF16 的 *.weight            -> FLOAT
#    - I8 + 形状等于 8bit 参考      -> W8A8_DYNAMIC
#    - I8 + 形状等于 4bit 参考      -> W4A8_DYNAMIC
#    - 同一模块的 weight/weight_scale/weight_offset/scale_bias 共用该类型
```

**判型关键**：modelslim 的 4bit 是**沿 dim0 折半**（`[2048,4096]` → `[1024,4096]`），不是折 dim1。

**必须补齐的元数据（最容易漏，漏了不报错但结果错）**：

```json
"model_quant_type": "W8A8_DYNAMIC",
"is_rot_used": true,
"optional": {"quarot": {"rotation_map": {"global_rotation": "optional/quarot.safetensors"}}}
```

Mixed 目录自带 `optional/quarot.safetensors`（67 108 960 B），说明权重经过 **QuaRot 旋转**；
缺这三项会导致加载成功但推理数值错误。

生成后可用参考仓库自校验：以 macdoor 的 description 回放生成逻辑，差异应只剩上述 3 个元数据键。

## 4. Mooncake KV 池（跨副本共享前缀缓存）

目的：两个副本共享同一份 KV 池。常态由网关粘性命中本地缓存，一旦请求漂到另一副本，
从池中取 KV 而不是重算（实测 **17.3 s → 2.6 s，6.7×**）。

### 4.1 配置文件（两台内容必须一致）

**放 NFS，不要放 `/etc/mooncake`**——GPUStack 的模型容器只挂载 `/root/Models`，
`/etc/mooncake` 在容器内不可见。

`/root/Models/.mooncake/mooncake.json`：

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "10.0.0.142:50088",
  "global_segment_size": "2GB",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true,
  "enable_ssd_offload": false,
  "tenant_id": "default"
}
```

`global_segment_size` 是**每卡**注册给池的显存（2 GB × 8 = 16 GB/节点），会挤占 KV，需同步下调 `gpu-memory-utilization`。

### 4.2 启动 master（只在 node1）

```bash
docker run -d --name mooncake-master --network host \
  --entrypoint mooncake_master <镜像> \
  --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 \
  --default_kv_lease_ttl 11000 --enable_offload=false --client_ttl=120
```

镜像内已含 `mooncake-transfer-engine`（0.3.13）与 `ascend_transport.so`，`mooncake_master` 在
`/usr/local/python3.12.13/bin/`。

### 4.3 hugepages

```bash
echo 50000 > /proc/sys/vm/nr_hugepages    # 建议 50000
```

不要设 200000：在部分节点上分配超时会长时间卡住。

### 4.4 验证

```bash
# 客户端注册数应 = 8 卡 × 副本数
docker logs mooncake-master | tail -1 | grep -oE "Clients: [0-9]+|Keys: [0-9]+"
# 期望：Clients: 16  Keys: 数千
```

功能验证（直连两个副本，同一段 130K prompt）：A 冷 17.4 s → B 复用 2.6 s。
副本日志中 `External prefix cache hit rate` 为池命中率（接收方上涨，提供方为 0）。

## 5. GPUStack 部署

### 5.1 后端版本

在 vLLM 后端下新增版本（`PUT /v2/inference-backends/1`，`version_configs` 追加一项）：

```yaml
910b-glm53-kda-custom:
  image_name: swr.cn-central-221.ovaijisuan.com/whcs-fae/vllm-ascend:glm53-kda
  run_command: ''
  entrypoint: ''
  custom_framework: cann
  env: {}
```

### 5.2 环境变量

```
HCCL_OP_EXPANSION_MODE=AIV
HCCL_BUFFSIZE=1024
TASK_QUEUE_ENABLE=1
OMP_NUM_THREADS=1
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
HCCL_INTRA_ROCE_ENABLE=1
HCCL_SOCKET_IFNAME=enp67s0f0
GLOO_SOCKET_IFNAME=enp67s0f0
TP_SOCKET_IFNAME=enp67s0f0
PYTHONHASHSEED=0
MOONCAKE_CONFIG_PATH=/root/Models/.mooncake/mooncake.json
LD_LIBRARY_PATH=<mooncake 包目录>:<镜像原有完整 CANN 路径...>
```

**`LD_LIBRARY_PATH` 必须写全量**：直接写一个前缀会丢掉 CANN 路径，报
`libhccl.so: cannot open shared object file` → torch_npu 加载失败。
镜像原值可通过 `docker inspect <镜像> --format '{{range .Config.Env}}{{println .}}{{end}}' | grep LD_LIBRARY_PATH` 取得。

**不要设** `LD_PRELOAD=libjemalloc.so.2`：该镜像没有这个库，会刷 `ld.so` 报错。

### 5.3 启动参数

```
--quantization=ascend --trust-remote-code
--max-model-len=900000
--tensor-parallel-size=8 --enable-expert-parallel
--data-parallel-size=1 --pipeline-parallel-size=1
--max-num-seqs=16 --max-num-batched-tokens=32768
--gpu-memory-utilization=0.94
--enable-prefix-caching --enable-chunked-prefill
--safetensors-load-strategy=prefetch
--speculative-config={"num_speculative_tokens":3,"method":"mtp","enforce_eager":true}
--compilation-config={"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,32]}
--limit-mm-per-prompt={"image":16,"video":1}
--reasoning-parser=glm45 --tool-call-parser=glm47 --enable-auto-tool-choice
--seed=1024
--additional-config={}
--kv-transfer-config={"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"mooncake","lookup_rpc_port":"0"}}
```

### 5.4 卷与凭据

- 卷：`/root/Models`（NFS）、`/etc/mooncake` 无需挂载（改用 NFS 路径）
- **私有镜像拉取凭据**：GPUStack 2.2.3 无 registry 凭据接口，需把 auth 写进 worker 容器：

```bash
# 两台 worker 各执行；worker 重启后需重新注入
docker cp dockercfg.json gpustack-worker-2.2.3:/root/.docker/config.json
```

## 6. 实测性能

同口径（均经 New API 网关、开启对话粘连）：

| 指标 | A100 `w4a16`<br>2×[4×A100 TP4+PP2] | 910B `w4a8`（macdoor）<br>2×[8×910B TP8+EP] | 910B `w4a8`（Mixed）<br>2×[8×910B TP8+EP] |
| --- | ---: | ---: | ---: |
| 上下文上限 | 131 072 | 1 048 576 | 900 000（受显存限制） |
| KV / 副本 | 未暴露 | 2 434 216 tok | 955 024 tok |
| 单流 decode | **91.7 t/s** | ~62.5 t/s | 65.1 t/s |
| 单流 TPOT | **10.9 ms** | 16.0 ms | 15.4 ms |
| 冷 TTFT 1K | 602 ms | **427 ms** | 434 ms |
| 冷 TTFT 8K | 1 749 ms | **1 213 ms** | 1 269 ms |
| 冷 TTFT 32K | 5 207 ms | 4 399 ms | **4 061 ms** |
| 冷 TTFT 131K | 18 804 ms | **16 518 ms** | 17 651 ms |
| 冷 TTFT 1M | ✗ | **247 s** | 256 s |
| prefill 32K | 6 042 t/s | 7 174 t/s | **7 748 t/s** |
| prefill 131K | 6 697 t/s | **7 627 t/s** | 7 131 t/s |
| 热 TTFT 8K | 501 ms | 452 ms | **434 ms** |
| 热 TTFT 32K | **780 ms** | 2 045 ms | 2 164 ms |
| 热 TTFT 131K | **2 004 ms** | 2 449 ms | 2 693 ms |
| 热 TTFT 1M | ✗ | **8.7 s** | 12.9 s |

要点：

- **910B 相对 A100**：prefill 快 1.1–1.5×、冷首字低 10–30%、上下文 8×；但单流 decode 慢约 1.5×。
- **Mixed 相对 macdoor**：速度基本持平（±8%），代价是 **KV 容量少 54%**（2.32× → 1.06×）。
  Mixed 买的是**精度**，不是性能。

## 7. 参数取舍：为什么是 900K 而不是 1M

vLLM 启动时会自检「能否至少装下 1 条 `max-model-len` 的请求」，装不下直接拒绝启动。

每卡预算（60.96 GiB）：

| 项 | 大小 |
| --- | --- |
| `gpu-memory-utilization=0.94` → 可用 | 57.3 GiB |
| Mixed 权重（实测 `Model loading took 36.15 GiB`） | 36.15 GiB |
| batched=32768 的激活/工作区 + 视觉编码器（32K token 预算，1 路视频） | ≈9.4 GiB |
| 剩给 KV | **11.74 GiB** |
| 1M 需要（每 token **12.9 KB**） | **12.88 GiB** ❌ |

实测演进：

| 配置 | 可用 KV | vLLM 估算最大长度 | 结果 |
| --- | ---: | ---: | --- |
| gpu-mem 0.90 / seqs 32 / batched 32768 | 9.31 GiB | 754 560 | ❌ |
| gpu-mem 0.94 / seqs 16 / batched 32768 | 11.74 GiB | 955 520 | ❌（差 1.14 GiB） |
| 上一行 + `max-model-len=900000` | 11.74 GiB | — | ✅ |

若要真正回到 1M，三选一：

| 方案 | 代价 |
| --- | --- |
| 换 macdoor 权重（21 GiB/卡） | 精度回标准 W4A8；1M + 32K batched + 16 图均可宽松满足 |
| `gpu-memory-utilization=0.96` | 压线通过，运行时 OOM 风险高，不建议 |
| `batched=16384` | 保住 1M，但多图/长 prompt 首字变慢 |

## 8. 踩坑清单

| 现象 | 根因 | 解法 |
| --- | --- | --- |
| `kex_exchange_identification: Connection closed` | 网关/sshd 侧限流或来源 IP 变更 | 等待解封或核对白名单 |
| worker 拉镜像 `You may not login yet` | GPUStack 无 registry 凭据接口，worker 内无 auth | 注入 `/root/.docker/config.json`（重启后需重做） |
| `libhccl.so: cannot open shared object file` | 覆盖了 `LD_LIBRARY_PATH`，丢了 CANN 路径 | 写全量路径 |
| `Hybrid models do not support recompute mode` | GLM-5.3 是混合注意力，不支持该策略 | 去掉 `kv_load_failure_policy: recompute` |
| `--api-server-count` + `--headless` 冲突 | 从节点不存在 API server | 从节点不要给 `--api-server-count` |
| `NnopbaseRunForWorkspace failed` 启动失败 | 加了 `enable_flashcomm1` / `multistream_overlap_shared_expert` / `weight_nz_mode` 等 flag | 保持 `--additional-config={}`（官方 W4A8 命令本就不带） |
| `NnopbaseExecutorTilingAndUpdateBinInfo failed` | TP4 时 KDA/线性注意力算子的 head 分片不被支持 | 使用 TP8 |
| `At most 1 image(s) may be provided` | `--limit-mm-per-prompt` 限制 | 按需放大 |
| `ValueError: To serve at least one request ... 12.88 GiB needed` | KV 不足以容纳 1 条满长请求 | 降 `max-model-len` 或降 `batched` 或换小权重 |
| 改完 GPUStack 参数不生效（容器仍 Up 数天） | PUT 只改记录，不滚动重启 | 副本数 0 → 2 强制重建 |
| 首个请求 TTFT 60 s+ | 自定义算子/Triton 编译（新容器首启） | 上线前 warmup |
| SWR 推送 `Invalid image, fail to parse 'manifest.json'` | BuildKit 产出的 manifest 被判不合法 | 用 `DOCKER_BUILDKIT=0 docker build` 重新构建后推送 |

## 9. 运维手册

**改配置**：`PUT /v2/models/{id}` → 然后 replicas `0 → 2`（否则不重建）。

**上线必须 warmup**：新容器首个请求会触发编译，实测 60 s 以上。

**关键监控**：
- `NPU KV cache size` / `Maximum concurrency`
- `Prefix cache hit rate` / `External prefix cache hit rate`（后者为 Mooncake 池命中）
- 投机解码接受率（正常 36–47%）
- `docker logs mooncake-master`：`Clients` 应为 8 × 副本数

**扩缩容**：单机 8 卡为一份副本，两机即两份；跨机不要组 TP16（decode 反而慢 11–27%）。

## 10. 未验证项与风险

1. **`video:1` 未实测**。模型侧完整支持视频（`video_token_id`、`video_processor`、帧采样代码齐备），
   但 A2 + 本镜像上未跑过真实视频；首次使用需观察 TTFT 与显存，异常则回退 `video=0`。
2. **Mixed 精度未评测**。更高的精度档位是否带来实际收益，需要一套评测集对比。
3. **`reasoning_effort` 语义受限（重要）**。模型 chat template 只识别 `low` / `high`：

   ```jinja
   {%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined
                                      and reasoning_effort in ['low', 'high'] else 'max' -%}
   ```

   即 `none` / `minimal` / `medium` / `xhigh` / `max` / 不传 **全部落到 `max`**。
   另外经实测，vLLM 的 OpenAI 接口**忽略顶层 `reasoning_effort` 字段**，只有
   `chat_template_kwargs.reasoning_effort` 生效。因此在网关上如果直接把 `reasoning_effort`
   放在请求顶层，该参数完全不起作用；表现出的"档位差异"多为采样噪声。
