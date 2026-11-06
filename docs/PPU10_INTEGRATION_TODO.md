# PPU1.0 GDN 性能接线与端到端验收 TODO

updated-at: 2026-11-06 03:18:25 UTC

范围：先闭合已保留的 PPU1.0 GDN 实现与实际调用入口，不重新开展 kernel 调优。
源码核对基线：main `66ec165`；serving 工作分支 `2e107e2`。
工作分支已实现公共 PPU1.0 默认接线和 route/image 审计，用户已返回36组默认入口
数值/逐位一致性 PASS；性能未测、完整 bundle 身份未核。新的 serving prefill
prepared 接线已完成本地编译和回归，待独立 box 准入。retained device kernel
本体不变，仅增加 host launch 接缝；main 与 wheel 尚未更新。

## 已确认与尚未确认

- 旧 main 已确认：`gdn_forward(backend="ppu10", algorithm="auto")` 选择 `original`。
- 旧 main 已确认：显式 `algorithm="residual"` 但省略 `delivery`，仍选择 `scalar`。
- 已保留：`algorithm="residual", delivery="full-chunk"` 是历史胜出实现；
  full-C64 和 guarded-tail 由 native 入口选择，代码没有丢失。
- 旧 serving 的 PPU1.0 prefill 是 sequential semantic-validation 路径；本次已接
  retained residual，但没有把 PPU1.5 的准入或公共默认 PASS 算成新 adapter 的 PASS。
- 未确认：用户当前源码安装测试实际调用了哪条路径；不能把当前慢因归到 wheel。
- 既有 decode/verify/commit/auxiliary 不是待从零实现；已有用户反馈的正确性 PASS，
  但不等于新 residual 接线、完整模型集成或真实性能均已验收。

代码依据：[公共路由](../actlize_la/gdn_interface.py)、
[保留路径](PPU_RETAINED_PATH.md)、[serving 合同](FLARE_SERVING.md)。

## 按顺序关闭

### P10-01 — 钉死当前调用路径与真实 shape

- [ ] **IMPLEMENTED / 本地审计通过；用户原测试的实际路径仍未采集**

已增加 `get_gdn_route_info`，与执行共用 resolver，准备阶段记录真实加载的
binding/device DSO 的路径及哈希。缺符号、替换文件、重复同名 DSO 负控均拒绝。
`load_binary=False` 明确为 policy-only，不算设备证明。

记录调用命令、源码 SHA、实际加载的二进制及 SHA256、编译目标、API 参数、
选中算法和 native 符号。路由信息在准备阶段可查询，不在 graph replay 热路径
添加打印、设备查询或同步。增加测试，避免“安装了新代码、加载了旧库”。

对当前 Qwen3.5-35B-A3B case，先确认 8192/5120 是 prompt/generation token 数。
若 B=1 且一次 prefill8192：Q/K=[1,8192,16,128]，V/O=[1,8192,32,128]；
普通 decode 每次 T=1。实际 prefill 分块、batch、dtype、MTP width 必须从调用中
记录，不能从 GGUF 文件名猜测。Hk16/Hv32 与 FlashNext 的 Hk16/Hv48 分开。

**关闭条件：** 可以直接证明该 case 跑的是 original、scalar residual、
full-chunk residual 或 serving recurrence 中的哪一条，并把 prefill/decode
分开计时。若已跑 full-chunk，就排除“未接线”作为该次慢因。

### P10-02 — 固定数值、状态和支持范围的准入合同

- [ ] **LOCAL_VALIDATED + USER_REPORTED_DEFAULT36_PASS / 待完整设备 bundle 核验**

既有 61-case residual 独立数学检查及4个算法负控通过；新增非对称 state
转置、重复 query scale、错 head 映射和缺失测试分母负控。box 脚本保留原36-case
scalar 对照，再跑36个目标几何 default/explicit 对照；旧数值门保持不变。

复用独立逐 token reference 与现有误差门，覆盖 Hk16/Hv32 和 Hk16/Hv48、
T=1/63/64/65/127/128/129/2048/8192、零/非零且非对称的初始状态。
明确 legacy [B,Hv,K,V] 与 serving [slot,Hv,V,K] 的转换；128x128 shape 相同
不能按字节重解释。核对 Q/K normalization、epsilon、query scale 恰好一次、
natural-log gate 与 exp(g)、beta 的 BF16 舍入、head 映射。

**关闭条件：** 输出及最终状态通过既有独立数值门；纯接线保持同一 residual
算术时要求逐位对齐；不同舍入/重结合算法不用 RAW-BIT 冒充合理的跨算法门。
植入状态漏转置、重复 query scale 和错误 head 映射，必须被测试拒绝。

### P10-03 — 公共入口接到已保留的 PPU1.0 实现

- [ ] **USER_REPORTED_NUMERIC_PASS / 待原始身份核验与 main 集成；依赖 P10-01/02**

用户返回：`[PPU default GDN] PASS cases=36 repeat=8 numeric+default-bits; performance=NOT_RUN`。
对应已交付的 d4b4713 命令；尚未独立读取完整 bundle/SHA。仅记公共默认数值与逐位
一致性通过，不把它写成性能、serving 或完整模型的 PASS。

工作分支的 PPU1.0 `auto`、显式 residual 省略 delivery、直接
`gdn_chunk_residual` 三个入口均已选择 retained full-chunk。旧 scalar benchmark
与数值 oracle 的控制臂已改成显式选择，不能随默认一起移动而产生假绿。
默认最终必须体现已准入的最快实现，用户不需要手选内部 variant。
本地检查：280项通过（native ABI 项使用本机 SM120 构建验证公共结构/拒绝边界，
不是 PPU 执行证明）；61个 residual 数学 case、4个算法负控通过；T8192/Hv32和48
的两种 gate 下非对称状态转置负控通过。无设备 runner 返回 SKIP/NOT_ADMITTED，
exit3，不报 PASS。PPU native kernel/source graph 无改动。

修正适用 prefill 范围内的 `auto` 和 residual 的默认 delivery，使普通使用者
不必知道调优 variant 名字。保留 explicit original/scalar 作为对照；T=1 decode
和 MTP 短序列单独定路由，不把长 prefill 胜出结果外推过去。
受支持范围、tail 分支、旧扩展缺符号和不支持目标必须有明确策略，不能静默
回到较慢实现再报告成“优化路径”。选择只依赖已知输入/设备元数据，不读取 gate
数值，不运行在线 profile/JIT。

**关闭条件：** 默认调用的实际符号、数值及同路径性能与显式 full-chunk 对照
一致；错误/缺失二进制负控为 FAIL。涉及算法变化的默认切换须有新 box 准入，
不能只改一行 Python 就标记完成。

### P10-04 — PPU1.0 serving prefill 接入 residual prepared/out

- [ ] **LOCAL_VALIDATED / prepared adapter 已实现，待 box 数值与图准入**

真实 SDK2.2.1 的 PPU1.0/PPU1.5 shared library 均编译链接通过；SM120 独立
shared library 也编译链接通过。306项 host 回归 +143个 subtest 通过，其中 native
ABI 边界使用实际 SM120 DSO，不冒充 PPU 设备执行。两处 retained TU 只新增 host
configure/launch 函数，device bodies 与 d4b4713 逐字相同；其余 retained TU/headers
无差异。

box gate 新增 Hv32/48、capacity64/65/129/8192 的 prepared/retained-C-entry 输出
和 FP32 final state RAW-BIT 配对、独立 recurrence、workspace canary 负控，以及
capacity64/65 的同图8次更换 length/live/initial-slot/final-slot。已有连续 prefill、
prefill→decode、verify/commit 和辅助算子门继续运行。使用
`TARGET=ppu10 bash tools/run_flare_serving_box.sh`，上传 `results.tar.gz`。
本轮未测新 adapter 的性能；prepare/pack/state transpose 必须计入后续完整算子计时。

**box 反馈后的辅助门修正：** `checked_gdn_tests` 遗留的两处条件只将 `ppu15`
prefill 识别为 chunk，把已接 residual 的 PPU1.0 套到 FP32 recurrence 的逐元素门。
用户报告 state 的118,699/786,432个元素失败；同seed/几何的独立 CPU residual
参考复现118,697个逐元素失败、整体 relative L2=0.2941%，但不是设备逐位复现。
改为两代 PPU prefill 使用既有2% residual门；`close()` 原阈值、decode/SM120
严格门、retained-entry RAW-BIT gate均不改。补测实际辅助测试控制流的三目标
分类与误写/数值错误负控，并在box结果记录实际 relative L2及算法合同。
错误清零后，checked/unchecked 的完整输出与状态池还必须 RAW-BIT 相同；
仅改 checked state 一个bit的负控也必须红。最终本地323项 +143个subtest通过，
包含17项本次专项回归。CPU控制流 stand-in 不充当设备执行证据。
这次不修改kernel或生产路由；修正后的完整box套件仍待用户重跑。

优先复用 `gdn_wy_forward_residual_full_chunk`：它已经接收外部 output、state、
inverse/snapshots/vnew/gates 和 stream，不重写 kernel。将 configure 等准备动作
移到准备阶段，暴露 graph-safe launch；补 slot gather/scatter、状态转置、
packed request/tail 处理及有上界的 workspace。

**必须核对 scratch ABI：** 当前 PPU1.0 binding 的 residual inverse 是 padded
[groups,128,128]；PPU1.5 serving 规格是 [groups,64,64]。不得仅放宽 target
条件就共用 PPU1.5 的容量与布局，须逐缓冲核对 native 下标并加越界负控。

**关闭条件：** 对照显式 residual 的输出/state、连续分块、非零 state、inactive
slot、tail 均通过；capture/replay 不分配、不 JIT、不同步、不 CPU readback；
同一图更换设备端长度/slot 后仍正确。包装/转换成本计入完整 prefill。

### P10-05 — decode / verify / commit 与新 prefill 连续运行

- [ ] **OPEN / 已有实现，待新路径联合验收；依赖 P10-04**

复用现有增量算子与 compact records，验证 prefill→decode、verify widths1/2/4/8、
consumed prefix 从0到全部、拒绝后继续生成、ShortConv 历史与 recurrent state
同前缀提交。保留“接受 draft 数 / target 消费数 / 用户输出数”三个不同计数。

**关闭条件：** 只提交有效前缀，零消费及 inactive slot 原始字节不变，长序列
状态误差在原有门内；decode 与 verify+commit 独立报时间，不用 prefill 吞吐替代。

### P10-06 — 实际长输入与 FLA 同口径对照

- [ ] **OPEN / 当前目标未建立同路径性能证据；依赖 P10-01/02/03**

先复核 T2048 历史几何，再测 T8192 及调用方真实分块长度；serving 臂在 P10-04/05
关闭后加入。PPU1.0 同卡同输入、同 dtype/gate/state/output 合同，对照显式保留路径、
默认路径与 FLA。真实 gate 输入和固定强/弱衰减诊断 fixture 分开记录。

**关闭条件：** 提交同轮源码/二进制身份及可复跑命令、小型结果 tar；主表采用
ACU 完整算子 kernel 时间之和，不漏 fill/pack/transpose/commit。使用
`/sim/eec/shared/junfu.qx/asight/bin/acu`。非插桩 API 时间单独列，不和 ACU 混算。
报告重复测量范围及全部输赢；不承诺 T8192 自动继承 T2048 的胜幅，不调整旧门
来让结果通过。模型 prefill/decode 的差距与单算子差距分开归因。

### P10-07 — Flare / QSA 与完整图集成证据

- [ ] **OPEN / 后排；已有接口不等于模型已接线，依赖 GDN 主线**

核实 Flare 实际调用的是这些 prepared bindings；不重写框架或 sampling。
复用已有 QSA 分页/增量接口，核对 BF16 KV 配置、page64/128、完整压缩块与尾部、
GQA、拒绝后因果可见性。QSA 属于 FlashNext 集成工作，不能把 Qwen3.5 的普通
full-attention 层当 QSA；compression_ratio0 的 MTP attention 仍交 FlashAttention。

**关闭条件：** 完整模型图重放更换 request/slot/page/长度/接受数仍正确；多层
scratch 共享不覆盖尚未消费的 correction records；执行显存稳定。明确区分
算子测试、模型图执行、package export/Rust replay，不能互相替代。

### P10-08 — main、源码安装和 wheel 路由统一交付

- [ ] **OPEN / 最后；依赖已承诺范围的数值及设备准入**

把已验收的最小接线闭包合入 main，更新统一入口及支持范围；不把开发分支整体
当成准入。源码安装与 wheel 同一输入必须选择同一实现，installed-only 检查
不依赖源码目录或旧扩展。PPU1.0/1.5/1.7、SM120/SM90 的目标与证据分开列。

**关闭条件：** 默认入口可直接复现已验收路径；清晰报告 source SHA、binary
hash、architecture、SDK 和数值/性能状态；PASS、SKIP/NOT_RUN、FAIL 分开计数。
交付构建、安装、box 测试命令与小型证据包。用户不需要手选内部调优 variant。

## 执行规则

按编号逐项做、逐项交付；P10-04/05 不挡住 legacy 路径先完成长输入诊断。
每项关闭时补 source/binary 身份、测试分母和结果，编译或 host PASS 不充当设备
PASS。若接线正确后仍慢，另列有测量依据的优化任务，不把性能未知写成性能胜出。
