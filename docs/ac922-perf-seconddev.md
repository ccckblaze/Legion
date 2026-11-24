# AC922 × Qwen3.8-Flash-Next 深窗聚合：根因定位与二开路线（2026-11-24）

> 本文是战役账（AC922-bringup-runbook.md §356-§357 有全部原始数据）的代码层续篇：
> 面向以 Legion 为主力的二开，钉死"123K+ 多路并发聚合塌方"的机制位置与验收线。
> 机器：AC922 4×V100-SXM2-16GB（layer-split 4 stage，物理卡 0,1,3,4，跨 NUMA 2+2）。
> 负载：Qwen3.8-Flash-Next NVFP4-GPTQ pack，`--spec mtp --spec-k 2`，q8_0 KV（19,200 B/token）。

## 1. 事实链（黑盒已穷尽，勿再扫参）

| 观察 | 数值 | 出处 |
|---|---|---|
| 深窗单流健康 | fw1 262K tg 31.2（q8） | §356.9 |
| 4×262K 同刻 | sum 32-36，形状恒为 0.6/0.8/1.9/~30（最后 prefill 完的独走满速） | W0/W2/W4/X0-X5 十一臂 |
| 4×262K 错峰 15s | sum 43.8，同形态复现（非碰撞瞬态） | Y1 steady |
| 毒药丸：8×8K + 1×262K | 8K agg 124.9→42.6（−66%），八流**均匀**掉到 5.5-6.5 | Y1 poison |
| 断崖位置 | 8K C4 44.4 ✓ / 32K C8 17.4 △ / 123K C4 **3.7** ✗ / 262K C4 ✗ | Z0/X4 |
| 配置面判负 | on-demand 准入、priority、max-batch-tokens 4096、staging off、swaps 48、heat-cap、groups 4/1、spec-k 4、m884、mtp-split 全部不破形态 | Wave05-08 |

关键分裂观察：**短窗流之间确实同窗共批**（毒药丸均匀减速），**深窗流之间不同窗**（11 臂恒为单流独走）⇒ 不是"批了但慢"，是"深窗没批"。

## 2. 主假说（P0 审计后钉到代码）

**H-R：多流 KV 总量超过 VRAM 驻留池 ⇒ 超出的流跌入 host-tier 流式 decode，每窗跨链路 tile 读 K/V，带宽墙同时解释全部六条观察。**

代码证据（`src/engine/lm.rs`）：
- `:1069-1074` `kv_host_tokens`：超出 VRAM 的 KV 进 pinned host tier（每卡自己的 NUMA 节点，"read over the link"）。
- `:1074-1077` `kv_stage_tokens`（默认 16384，`:1545`）："each step reads a host block over the link once instead of once per query tile; **Steps reaching more host tokens run in pieces**" ⇒ host 部分按 tile 切分，深窗每窗 host 读量 ∝ (ctx − resident)。
- `:2061` 池 sizing："the asked context stays, its cache takes the room the compute buffers" 之后——我们的 boot `--kv-cache-tokens 262144` = **恰好 1×满窗**（q8 池 4.69 GiB/GPU×4）。

数值自洽性（这是假说的判决性优势）：
- 阈值 = **池 token 总量**，不是单流长度：8K×4=32K ≪ 262144 ✓ 正常；32K×8=262144 **恰等于池** △ 开始退化；123K×4=492K ✗ 塌方；262K×4=1048K ✗✗ 全灭。
- "最后 prefill 完的独走满速"：KV 插入序决定谁占着 262144 的池 ⇒ 不是 LIFO 调度偏置，是**驻留偏置**。
- 毒药丸均匀减速：1×262K 把其他三条 8K 流的驻留挤出池 ⇒ 全体转 host 流式、同窗共慢。
- 单流 123K/262K 不掉速：123K 和 262K 单独都在池内。
- Strata 对照：同机同包，`--kv-resident` 分池 + host-tier 预算纪律 + resident complement 管理，123K C4=15.3、fw4=106.9 —— 差距全部落在"深窗流能否留在 VRAM"这一维。

候选降级：
- 旧候选 1（step compute buffer 行数预算）：prefill 事故的 need_mib=2407 是 tokens=3072 chunk 的事；decode 窗 gather tile 只 16384 stage tokens——降级为次要（与 H-R 同根：都是 link/VRAM，但药方不同）。
- 旧候选 2（Volta QSA 多行打分）：毒丸的短窗共批证明打分不是墙。
- 旧候选 4（调度偏置）：并入 H-R 的驻留解释。
- 候选 3（异源路由干扰）：仍有独立权重（W0 日志 hit_rate 0.91→0.81、21 swaps/窗），用 K3 臂分离。

## 3. 判别臂（P1 修正版，全配置级，各 ~15-20 分钟）

| 臂 | 内容 | 判决 |
|---|---|---|
| **K2** | `reburst 61K C4`（4×61500=246K < 池，全驻留） | agg ≥60 ⇒ H-R 坐实（池阈值=聚合墙主因）；~10 ⇒ H-R 死，回候选 1/3 |
| **K1** | `--kv-cache-tokens 492K/786K`（压 expert cache 换驻留）重测 123K C4 | 恢复幅度 = H-R 剂量响应；同时读 boot 的 "KV host tier holds..." 自述行 |
| K3 | 同源 4×123K C4 vs 异源 | Δ>3× ⇒ 路由干扰为二级主因（per-seq 专家分区二开立项） |
| K4 | `--spec-k 1` 123K C4 | 验 verify 行×3 对 host tile 读量的放大系数 |
| K5 | `RUST_LOG` 开 step 级 debug（引擎自带 `seqs=batch.len()` 行，`lm.rs:9108`），统计深窗档每窗序列数 | "没批"的直接窗口级证据；无需打补丁 |
| K6 | nvidia-smi dmon/NVML 链路计数采样（与 K1/K2 同跑） | host-read 带宽占用 vs 塌方时刻对齐 |

## 4. 二开修复形态（P2/P3 修订，按 H-R 命中后排序)

1. **驻留调度补丁**（最对症，Rust 侧）：把 decode 池按活跃流数动态收缩每流驻留窗（resident = pool/active_seqs，超出部分走 host-tier 但保留"热尾"在 VRAM）——Strata `--kv-resident` 分池 + slot_fresh 语义的 Legion 化。
2. **stage tile 流水重叠**（内核+调度）：`kv_stage_tokens` 跨窗预取下一流块（毒药档 link 77 GB/s < 单流上限，有重叠空间）；姊妹教训（预测式 prefetch −2~15%）要求带阈值纪律。
3. **KV 更窄缓存**：q4_0 是 QSA 内核支持内的下一档（19200→~11k B/token，池容量×1.7）——质量门走 §356.9 三门禁（acc+sha+needle；det 在 Legion 线不可用）。
4. 热专家 NUMA 冗余（用户 P3 项）：属于专家 link 路（另一条 61% 账），与 KV link 路分账管理，勿混。
5. 排队/插队调度（用户 P3 项）：治 TTFT 公平不治带宽墙，排后。

## 5. 验收线

- 机制清晰 = K2/K1/K5 三臂给出方向一致的判决。
- 聚合甜点 = 在 123K C4 上 ≥ Strata 的 15.3（追平线），目标 25+（甜点位）；8K/chat 王座档不回退（44.4/294.6 基线）。
- 质量门：acc 6/6 + 逐题 sha（对齐 F16 参考，容忍 fib 类抖动题）+ needle 120K PASS；数值路径有变的臂（q4_0、驻留窗收缩）另加分歧二分探针。

## 6. 观测点速查（二开时复用）

- boot 自述：`KV pool sizing ... reserve_gib`、`memory plan part="context" ... kv_host_tier_tokens`、`model loaded ... kv_bytes_per_token/kv_cache_gib`、`KV admission ...`。
- 周期行：`expert cache, last 512 logged steps hit_rate=... swaps=...`。
- 请求行：`request done ... decode_tok_s=... decode_mem_gb_s=... decode_link_gb_s=... decode_mem_roofline_pct=...`（毒药档 45 GB/s 有效读 vs 单流 186 —— 这仨字段就是带宽墙的表计）。
- 窗口级：`RUST_LOG=<...>=debug` 后 `seqs=` 行（K5）。

## 7. 判别结果（2026-11-24 22:5x CST，wave10/11 实测追加）

**H-R（池驻留假说）证死**：K2b = 4×62,453（合计 250K token ≪ 池 262,144，**零 host-tier**）仍复现完整饥饿谱 `tg=[3.2, 4.6, 8.1, 31.7]`。驻留不是主矛盾。

**新账（跨臂一致性）**：tg_med × ctx ≈ **593 (8K) → 457 (62K) → 396 (123K) → 160 (262K，被排队截断)**——每窗成本近似 ∝ 窗口长度（与是否驻留无关）；且 1/4 流比值（31.7/3.2≈10）是非线性的 ⇒ 单窗 gather 线性论不闭合。

**H-B′（wavelet 结构假说）**：深窗下引擎可能不共窗，而是按 prefill 完成序**分波串行**（每波 1-2 流满速 + 其余排队），K2b 的 3.2/4.6/8.1/31.7 谱正是波次时隙的指纹。判决交给 K5（`lm.rs:9108` 的 `seqs=batch.len()` debug 行，RUST_LOG 开窗）——若深窗档 `seqs` 恒为 1-2 ⇒ "没批"实锤，二开第一刀切在**decode 窗成形/波次调度**（`queue.rs` pick + `lm.rs` admit/step 交界），而非 gather 内核。

**候选状态板**（截至 K2c）：
- H-R 池驻留 ✗（K2b 证死）
- gather 线性带宽墙 ✗（K2b 零 host-tier 仍塌）
- 异源路由干扰 ？（K3 未跑；X0/X5 的 heat-cap/swaps 无效已给它减分）
- **H-B′ 波次/窗成形偏置 首要嫌疑**
- spec verify 行放大 ？（K4 未跑）

## 8. 根因定案（2026-11-24 23:2x CST，K4x/K5/巨步窗口日志实锤）

**一句话**：`--spec-k` 无关（K4x=1 与 K5=2 谱形全同）；**每当一条已 prefill 完成的流首次进 decode 组，引擎重放该流整个 prompt 以物化 GDN 状态**——每窗 ~O(ctx) 秒级巨步 × N 流错峰入组 = 大窗聚合塌方。三处铁证（`logs/K5-225625.log`）：

1. `host time per step ... exec_us=24901680.9`（设备段 24.9s，plan 0.19s / device-launch 0.44s / finish 14.6ms——不是调度、不是拷贝、不是 commit）。
2. `mtp take-in pieces positions=63461`——join 步把 MTP 头的 **prompt 全部位置**分片喂入。
3. 巨步后紧跟 `INFO replay commits: one all-slot graph`——全槽位重放图提交。

**数值自洽**（巨步 ≈ 该流 pp 墙钟）：62K→24.6s×3 ✓；262K→~100s×3 = ttft 489/594/601 阶梯 ✓；毒药丸 wall 65.6→192.2（Δ127s≈一条 262K 重放+排队）✓；32K C8 agg 17.4（8×~12s 重放）✓；chat C16 免疫（重放 ~0.2s）✓。

**候选终板**：池驻留 ✗（K2b）、路由干扰 ✗（X0/X5+K4x 同谱）、verify 行放大 ✗（K4x）、调度偏置 ✗（seqs 谱为入组序）、**join 重放 ✓✓✓（唯一幸存，全档案自洽）**。

**与 Strata 对照的机制翻译**：Strata 满窗并发 106.9 的本钱 = prefill 每 chunk 的 GDN 态**就地物化** + 入组 = 槽位指针交换（O(1)）。Legion 把该物化推迟到首 verify 整段重放。这不是 Volta 算力问题，是**状态生命周期设计**问题。

**二开第一刀（P2 定案）**：`src/engine/lm.rs` 步循环 join 路径——prefill 收尾时按 chunk 提交 GDN 态（或让 prompt 末态即入组态，免全槽重放图）；验收线 = 62K×4 的 tg_list 从 [3,5,8,33] 收敛到 [≥25×4]，sum_tg 47.6→**≥100**；123K C4 3.7→**≥15**（Strata 追平线）；chat/8K 王座档不回退。补丁点候选：`replay commits` 发布方（model/core/spec 层）与 `mtp take-in pieces` 调用侧（engine/lm 层）的时序交界。

## 9. 终修与解药（2026-11-24 23:5x CST，Wave14/K7 判后定稿）

**§8 修正**：巨步不是"join 后的额外重放"——K7（NOSPEC）四条巨步（`steps=1 exec_us≈23.4/23.7/23.5/23.2s`，间距=pp 墙钟、`seqs_per_step` 1→4 递增）与 request-done 的 `prefill_seconds=23.6/23.7/47.2/70.3` 完全对账。巨步**就是 prefill 本身**：Legion 的长 prompt 在 lend 态按 `LongPrompts` 大步设计"整段吞"（见下），`mtp take-in`/`replay commits` 只是巨步收尾时的顺带标记。§8 的二开方向据此作废，替换为本节。

**机制（代码位钉死）**：
1. `src/engine/lm.rs:1343-1390` `LongPrompts` 默认：`min_tokens=2048`（触发 lend）、`batch_tokens=65536`、`alone_tokens=262144`——注释原文自证："a 123K prompt in two 64K steps costs 37 rounds instead of 34: +9% prefill"（有意设计，买 pp 速度）。
2. `src/engine/lm.rs:8440-8449` lend 态 `big=true` 时 chunk 选择："A borrowed cache stages every expert per step: the longer the better" → chunk=`long_batch`（无 decode 在飞=整段 262144；有则 65536）→ 62K 一步吞完。
3. 同窗 decode 行只在步首"ride ahead … as one micro-chunk"（`riders` 逻辑，lm.rs:8398-8402 区）——即**每条巨步期间，在飞流每步只得 ~1 token**。四条流的 62K 谱 `[3.3,4.8,8.4,35.1]` = 前流被后流巨步冻结、末流独走尾窗的算术指纹（尾窗 7.3s×35=256 ✓）。
4. chat/8K 免疫的原因：prompt 剩余 < `min_tokens=2048` 不触发 lend，走常规 512-chunk。

**内置解药（从未启用）**：
- `--itl-target/--ttft-target`（`src/engine/lm/slo.rs`，完整步时模型）按 ITL 目标收缩每步 prompt 预算——但 `slo.rs:72` 硬性要求 `--pipeline-groups 1`；X4 形状 groups=4 ⇒ **解药被形状封锁**（判据："decoding groups in flight … no step time measures a decoding stream's inter-token latency"）。
- `--long-batch-tokens/--long-alone-tokens` 直接封顶巨步（serve.rs:1363-1369），保留 lend 的专家 staging；`--prefill-staging false`（serve.rs:379，默认 true）彻底回到 512-chunk 交错档。

**Wave15 判别臂（全配置级）**：S0=CTL；S1=`--long-batch-tokens 8192 --long-alone-tokens 8192`；S2=`--prefill-staging false`；S3=`--pipeline-groups 1 --itl-target 500 --ttft-target 120000`。每臂测 62K×4（fw4k 2600）、123K×4（fw4k 5120）、8K×4（fw4k 340）+ 日志摘要（巨步计数、prefill_seconds 阶梯）。验收线不变：62K sum≥100、123K C4≥15、8K/chat 不回退。

**二开刀口（按 Wave15 结果取一或组合）**：
- 若 S1/S2 净正但 pp 代价超标：**自适应 `long_batch = min(batch_tokens, f(在飞 decode 数))`**（Rust 侧小补丁，`plan_step` chunk 选择点 lm.rs:8440 区）——decode 在飞时按 ITL 预算切步，空窗时恢复大步。
- 若 S3 最优：**放宽 `slo.rs:72` 的 groups>1 禁令**（用"组内最近 token 间隔"替代整步时测量），把 SLO 与 pipelined groups 同时拿到。
- 结构性选项：**riders 每 micro-chunk 一次**（而非每步一次）——步不变、冻结消除；改动在 `Executor::set_pipeline_riders` 消费侧。
- 用户 P3 项归档：热专家已有（`--expert-heat-pass-cap`，X4 在用）；"总上下文+最大并发调度"≈ `--kv-admission`+`max_seqs` 语义，Wave15 后评估。

**预测（待 Wave15 证伪）**：S2 的 123K C4 sum_tg 0.6-2 → **8-20**（rider 步时 ≈0.2s 档）；S1 → 5-12（3.3s 档）；两者 pp 阶梯（prefill_seconds 尾差）分别 +10-30% / +2-3×。若 S2 兑现且 pp 代价 ≤30%，4 卡深窗并发从"塌方"改判为"可用"，定档形状随之更新。

## 10. Wave15 判负表与 Wave16/17 计划（2026-11-25 00:5x CST）

**Wave15（X4 基形，spec 默认 on；fw4k = 4×C 同上下文并发，256 tok/流）**

| 臂 | 配置增量 | 8K 谱 | 62K 谱 | 62K wall | 123K 谱 | 123K wall | 判 |
|---|---|---|---|--:|---|--:|---|
| S0 | — | [13.5,17.5,25.3,39.0] | [3.3,4.7,8.4,35.3] | 102.0 | [1.2,1.5,2.8,29.9] | 261.6 | 基线（阶梯毒） |
| S1 | `long-batch/alone 8192` | [13.7,17.5,24.9,39.5] | [2.8,4.0,7.3,35.7] | 124.1 | [1.0,1.3,2.3,30.2] | 315.3 | **FAIL**：谱不变，pp −22% |
| S2 | `prefill-staging false` | [35.8,39.5,39.6,39.6] | [37.2,37.3,37.3,37.3] | 613.8 | [27.5,31.9,31.9,32.0] | 1251.6 | 公平但 pp 4.8-6× FAIL |
| S3 | `groups1 + itl 500 + ttft 120s` | [1.0,1.3,2.1,42.3] | [1.1,1.1,1.1,42.5] | 754.2 | 进行中 | — | **FAIL**：双输（pp 330 t/s + rider 1.1） |

**机制定案（四臂合读）**：rider 推进 = 每引擎步 1 次采样（+spec 行）；步时长 ≈ chunk/pp_rate；而 pp_rate ∝ chunk（staging 摊销：S2 实测 412 t/s@512 vs S0 2657@65536，S1 2020@8192）。三约束成环——**公平要小步、快 pp 要大步、SLO 小步毁 pp**。配置层无解，判负终结黑盒扫参。
S2 的"uniform 37.3"实为 all-pp-then-burst（四流 `prefill_seconds=606.837` 逐位相同、解码集中在尾 6.9s）——不是窗内交错，reburst agg 口径下反而崩（1024/1251=0.8）。

**Prefix cache 真相（§355 考古）**：X4 的 `--prefix-cache-states 0` 是当年**为冷算口径对齐刻意关的**（runbook:10506 "measurement only"），非机制不可用。`--prefix-cache-interval N`（serve.rs:1478-1488）在 Flash-Next 单机上"prefill 顺带存中间态、不加步"，让重复前缀请求从分支点续读——**评测用关、生产用开，两套口径分开记账**。基准对照（15.3 vs 3.7）本身同条件冷算，结论不受影响。

**Wave16（挂账中）**：S4 = staging on + `long-batch 2048/micro 1024`（探"大步→小步时 pp_rate 斜率是否被 staging_reuse 保住"）；S5 = S4 + `prefill-keep-cache 4096 --prefill-keep-spread`（lend 期保留每层 cache，"smaller micro chunks stop waiting on them"——上游为此形状留的正解）。
**Wave17（挂账中）**：S6 = X4 + `prefix-cache-states 8 --prefix-cache-interval 4096`；S7 = states 4 + interval 8192 + `long-batch 16384`。判据改双口径：reburst C4@123K（agg，对齐 3.7/15.3 历史线）+ fw4k 谱形/墙钟。VRAM 哨兵：`saved_states_mib` boot 行 + 卡3 空闲（484 MiB 旧账，8×50MB 若挤先降 states=4）。质量门：S6/S7 若采纳，acc+逐题 sha+needle 120K（冷 vs 缓同答性）。
