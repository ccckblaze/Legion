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

## 11. Wave16/17 终局：生产形状改判（2026-11-25 01:5x CST，全部实测）

**Wave16 判负（rider 环彻底坐实）**：S4/S5（`long-batch 2048` + micro 1024 ± `prefill-keep-cache 4096/spread`）谱仍阶梯 `[1.8,3.4,5.9,35.4]`/`[1.6,3.0,5.4,35.2]`，wall 164/188（劣于 S0 的 102）。缩步不换来公平——rider 每引擎步只推进一次采样，步小了 pp 反噬。配置层对"冷算异源同刻并发"宣告终结。

**Wave17 翻案（prefix cache = 生产正解）**：S6/S7（X4 + `--prefix-cache-states 4/8 --prefix-cache-interval 8192/4096`，S7 另加 `--long-batch-tokens 16384`；注意必须去掉 `--prefix-cache-states 0`——那是当年评测对齐用的，见 §10 考古）：

| 臂 | 123K C4 reburst(agg/wall) | 满窗262K C4(agg/wall/tg_med) | 62K fw4k | poison55 混合流 |
|---|---|---|---|---|
| Z0/X4（缓关） | **3.7** / 555.1 | 不可达（500） | [3.3,4.7,8.4,35.3] | 42.6 阶梯 |
| S6 | **129.9** / 15.8 | — | [1.7,2.2,4.5,34.4] | — |
| S7 | **129.4** / 15.8 | **86.2** / 19.0 / 26.5 | [2.0,2.6,4.9,33.9] wall 162 | **87.3 / [12.0→30.5] 均匀** |
| Strata 参照 | 15.3（当年对齐冷算） | 3.4（墙钟口径） | fw4same 1.2/1.7/3.5/36.5 | — |

**质量门（S7 boot，全绿）**：needle 123K ×3 PASS（含缓存恢复路径）、acc 6/6、det 双跑 `det_same=true` 且逐题 sha 全等、VRAM 每卡剩 1121-1139 MiB。

**诚实口径**（引用前必读）：reburst C4 同 prompt = 共享前缀最好情况；poison55（共享底+分叉尾）才是生产代表形——S7 在它上面拿到均匀满速，这是改判的真正依据。**冷算异源同刻**（四条互不相干的 123K 同秒进入）仍是阶梯（≈X4），属二开 rider 环未解项，与生产主场景解耦。

**定档建议（S7 为 Legion 生产形状）**：X4 全基形 − `--prefix-cache-states 0` + `--prefix-cache-states 4 --prefix-cache-interval 8192 --long-batch-tokens 16384`。

**二开路线（冷算异源档，挂账不变）**：rider 环三约束（§10）→ 刀口按性价比序：①`--itl-target` 解除 groups>1 禁令（slo.rs:72 的测量语义换成"组内 token 间隔"）；②lend 态 chunk 自适应（decode 在飞时 `min(long_batch, rate·itl_budget)`）；③riders 每 micro-chunk 推进（采样下推到 device rounds，重）。

**运维注意**：boot 命令行以 `pgrep -af legion` 为准固化进 `legion-lane.sh`；8080 转发与鉴权（Legion 线无 `--api-key`，公网面必须走反代/防火墙，勿直绑）。

## 12. Wave18：冷算异源 storm 配置层终扫 + 管理 boot 复验（2026-11-25 02:00-06:00 CST）

**夹具口径修正（先记账，防后人再踩）**：

- `fw4diff.py` 把 body 截到 `24R` 字符（≈4.77R token），`fw4k.py` 不截（≈24R token）。同名 reps 在两个夹具里差 5×：wave18 用 fw4diff 13000/25600 才是真 62K/123K。首轮小尺寸（fw4diff 2600/5120 = 12.4K/24.4K）实测全均匀（[7.0×4]/[4.1×4]）——池 cliff 以下无阶梯。
- `fullwindow.py` 校准只用 FILL 单位收敛，err 落在 ±(5..23) 时 `int(err/24)=0` 死循环 14 次 → `CALIB-FAILED`（wave17 当时是探针恰好落进 ±4 才过的，非协议保证）。已修：|err|≤20 改用单 token filler（" 1"）微调收敛。修后 262K C1 校准精确落位（ptok=261900, gen 重调 244，不变式 prompt+gen==262144 成立）。
- **reburst 快线需要 C=1 顺序 priming**：`reburst.py W 4` 直接开火时，四条同秒 lend 步会把 state 池打脏、首轮不存态 → 二三轮仍全冷（实测 run2/3/4 全 ~400s）。正确协议 = 先 `W 1`（2 请求顺序，天然 priming）再 `W 4`。wave17 的 129.4/15.8 就是这么来的；本次管理 boot 复现 **113.1/18.1**（tg_med 32.2，boot 间热态噪声内）。

**Wave18b 三臂（真尺寸，异源冷算 storm）**

| 臂 | 配置增量 | 62K [tg谱]/wall | 123K [tg谱]/wall | 判 |
|---|---|---|---|---|
| W0 | S7 原样 | [1.2×4]/221（复测 222、221） | [0.7×4]/352 | 基线：**均匀但慢**（pp-bound 墙钟） |
| W1 | `long-batch 512 micro 512` | [0.5,0.5,0.5,0.7]/474 | [0.2,0.2,0.3,0.5]/1213 | **FAIL**：ride 频率不是瓶颈，pp 崩 3.4× |
| W2 | S7 + `--spec mtp --spec-k 4`（shared-Q8_0 草稿） | **[1.9×4]/134（−39%）** | [0.6×4]/444（+26%） | 62K 赚深窗亏，见 gate4 canonical 复验 |

**机制再钉（代码级，供二开）**：`step_piped` 的 `can = prefilling.is_empty() && !self.lent`（pipe.rs:144）⇒ 冷 storm 全程无批式 decode 窗；lend 步里 decode 行每步一次采样（exec.rs:3712 riders 前置单 micro-chunk），ride 速率 = 1/步时 × (1+N·acc)。配置臂只能在「pp 率 ∝ chunk」与「ride 频率 ∝ 1/chunk」之间换轨（W1 证死）；W2 是唯一抬乘子不缩步的形态，但 verify 行在 123K 深窗上有 +26% pp 代价（62K 无）。破环仍归 §11 二开刀口序（①slo groups 解禁 ②lend chunk 自适应 ③riders device-round）。

**管理 boot 生产复验（lane 脚本 = 定档形状，服务器 `/data/work/legion/legion-lane.sh`，仓库 `scripts/legion-lane.sh`）**：canonical 123K reburst 113.1/18.1 ✓；262K fullwindow C1→C4 与 poison55、det/needle 门 = gate3（数字见下表，boot=lane 04:53）。lane 脚本本轮修三处：`--max-seqs` 对齐定档 16、node meminfo 字段位（$4）、`--spec-k` 去重 + `SPEC_MODE` 透传。

**管理 boot 终局 gate 表（boot=lane 04:53，UTC+8 减 8 即日志时间）**

| 件 | 结果 | 对照 |
|---|---|---|
| fullwindow C1（2×262K） | PASS 2/2 held edge=262144，span 223.17 | wave17 235.1，同形 ✓ |
| fullwindow n=8（历史口径） | 截断于 warm 4（引擎无恙） | **口径废弃**：8×262K 超 池262144+host 786432=恰好 4×262K 的预算，n=8 本质超配；正装验收形改为 **n=4**（见下行） |
| fullwindow n=4（验收线） | 见 §12 补录（w18-fw4n4.log） | — |
| poison55 base | agg 83.3，tg 谱 [12.1→30.1]，8 流同刻收尾 | wave17 87.3 同谱 ✓（boot 噪声内） |
| poison55 poison 臂 | agg 26.1，八流均匀压至 [3.4-4.1]，intruder 自身 tg 22.3/pp 1184.9 | 首采。lend 税 −69%，rider 每步一采样的直接显形（见 ac922-decode-window-anatomy.md §2） |
| throne 8K C4 | 44.9 / tg_med 32.0 | wave17 44.4 ✓ |
| throne 8K C16 | 213.3 / tg_med 19.2 | ≠ wave17 chat-C16 294.6（不同夹具，不可直接比） |
| throne 32K C4 | 18.3（**含冷 priming 段**） | burst 段实测 4 流并行 ~32-35，真实 burst agg ≈120；harness 口径注记 |
| det ×1 | det_same=true，acc 6/6，sha 六枚与 wave17 全等 | 定档数值路径未变 ✓ |
| needle 123K | PASS units 5001 span 105.5 | wave17 106.3-107.7 ✓ |

**poison 臂机制注**：base 谱的阶梯（12→30）来自「准入排队 + 末个 prefill 后组轮转均分」；poison 臂把全谱压平到 3.4-4.1 是 lend rider 的均分形态——两组签名分别钉住 `prefilling` 门与 `lent` 门（pipe.rs:145），解剖见 `docs/ac922-decode-window-anatomy.md`。

**gate4 终判（W2 canonical，boot 07:06 spec-mtp-k4，同电池）**

| 件 | S7（管理 boot） | W2（gate4 boot） | Δ |
|---|---|---|---|
| throne 8K C4 | 44.9 / tg_med 32.0 | **108.7 / tg_med 41.9** | **+142%** |
| reburst 123K C1 wall | 101.0 | 75.3 | −25% |
| reburst 123K C4 | 113.1 / tg_med 32.2 | 91.3 / tg_med 28.4 | **−19%** |
| 62K 异源冷 storm wall | 221 | **177.6** | **−20%** |
| 123K 异源冷 storm wall | 352 | 460.3 | **+31%** |
| det | sha 六枚基线全等 | **与 S7 逐位全等**（acc 6/6, det_same=true） | 0（T=0 spec-verify 无损实证） |

**判词**：W2 = **情景限定正**。浅窗（8K）与浅冷风暴大赚，深窗暖并发与深冷风暴亏——MTP 宽草稿把 verify 行带进每步，浅窗步时便宜赚乘子、深窗 QSA 行贵付学费。**生产默认维持 S7**；W2 以 env preset 收录（`LEGION_USE_DRAFTER=1 LEGION_SPEC_MODE=mtp LEGION_SPEC_K=4`），按语料窗深切档是二开的自然下一步（serve 层路由，零数值风险——det 全等已证）。W2 boot 后已恢复 S7（boot 07:24:32，smoke 8K C4=59.1 ✓）。
