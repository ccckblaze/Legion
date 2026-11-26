# AC922 深窗 rider 交错：把"每步一采样"改成"每微块一采样"的设计件

状态：**设计 + 判别臂已排，未编码**（等 §387 的 17 臂结果落袋再决定动不动手）。
配套：`docs/ac922-decode-window-anatomy.md`（机制）、`docs/ac922-aggregate-tg-configmap.md`（配置层）。
行锚取自本树 HEAD（`2a12f99`）。

## 1. 病灶的精确形状（三行代码）

```
src/engine/lm.rs:9044   let riders = big && self.exec.n_stages() > 1 && self.exec.kv_ring().is_none();
src/engine/lm.rs:9059   let mut ride = micro.saturating_sub(decoding.len() * n_tok);
src/engine/lm.rs:9097   chunk.min(budget).min(ride)
```

一条长 prompt 进入 ⇒ `big` ⇒ 在编的 decode 行被塞进**第一个 micro-chunk 的图里**"搭车"，
其余 micro-chunk 只有 lead 在跑。于是：

- **decode 行每步只采样一次**，而步时 ≈ lead 的整个 prompt chunk 时间；
- 满窗冷入时步时以十秒计 ⇒ 同刻 4 条会话里"一条快道 ~30、其余 0.6-2"的阶梯（§386 实测 max/min ≈ 4.0）；
- 这不是"批了但批得慢"，是**decode 根本没跟着窗走**。

## 2. 为什么"缩 chunk"不是解药（已判死的反例）

`src/model/core/exec.rs:2988-2993` 的自述就是刹车：

> Every micro-chunk reads all the experts it routes to, so this should be what one step would take without stages.

Wave18b W1 实测：`--long-batch-tokens 512 --long-pipeline-micro 512` 把 ride 频率提上去了，
但 62K 档 pp 崩 **3.4×**（墙钟 221→474 s）。M2/M4/EV 三臂只是在探这条曲线的**上侧**
（`micro_for` 被 `MIN_MICRO=2048` 掐死，见 configmap §1），不是另一条路。

⇒ 任何"让 decode 更常采样"的方案，只要代价是**每 token 摊到的专家 stage 变多**，就自动输在 PP 上。

## 3. 推荐形态：micro-chunk 之间插一次 decode-only 轮（"rider lane"）

不改 chunk 尺寸、不改专家读次数、不动 lead 的行序——只在 lead 的两个 micro-chunk 之间，
给在编 decode 行跑一次 **decode-only 图**（每行 1 token，或带 MTP 时 1+N token）。

- **插入点**：`src/engine/lm.rs` 的 micro-chunk 序列生成处（`chunks` 向量在 9097-9140 成形，
  随后交给 `Executor::step`；`riders && chunks.len() > 1` 的 `rotate_left(1)` 说明 lead 已被放在最后）。
  最小实现 = 在 `for chunk in chunks` 的驱动处，每当 `decoding` 非空且本 chunk 不含 decode 行时，
  先插一次 `step(decode_only_rows)`。
- **收益上界**：decode 行采样次数从 1/步 → k/步（k = micro-chunk 数）。深窗 tg 的天花板是
  decode-only 步时（本机实测地板 ~17-18 ms/窗 ⇒ 单流深窗可到 35 以上），
  所以 4× 阶梯的收敛空间就是 §386 算的 **~2.7× 聚合 decode**。
- **代价**：每次插入多一个图 launch（decode-only 形状在正常 decode 里已存在，图缓存应能复用；
  若 `Executor` 的图按"行×chunk"形状分键，需要确认 decode-only 轮不会新造形状 ⇒ 这是编码前要在
  `src/model/core/exec.rs` 里核对的第一件事）。lead 的墙钟只多这几次 launch 的时间（预计 <3%）。
- **数值口径（诚实记账）**：decode 行从"与 lead 同图"变成"独立图"，**每行的数学不变但归约的分块
  可能不同** ⇒ 不能宣称 bitwise。按既有纪律走三件套：`det×3` 一致 + acc 漂移哨兵 + 分歧二分探针。
  若探针显示分歧只出现在 tie-break 位置，则与 `--split-last-extra` 同档（可采纳，需注明）。

## 4. 编码前必须先量的一件事（零代码，dev 2 卡即可）

`restore_cache` 的后台回填窗：lend 结束后专家槽位仍在回填，**回填完成前 decode 的专家全走 host 读**。
这条已有 1 s 粒度日志（`expert cache ... hit_rate=`）可直接量化：
复活后 hit_rate 从 0.61 爬到 0.93 用时多久 = 深窗复活后前 N 秒的 tg 为什么低。
若回填窗 ≥ 复活后 decode 时长的 30%，那"rider lane"的收益会被 host 读吃掉一半——
先测再写，避免把二开花在错误的瓶颈上。

**已答（零 GPU，取 11-26 的 4 卡日志归档 `legion-4-2026112{0,1}*.log`）**：hit_rate 的下降事件
之后回到 ≥0.90 只需 **1-7 秒**（`+179s 0.440 → +180s 0.902`；`+29s 0.386 → +30s 0.889`；
最差一例 `+0s 0.734 → +7s 0.937`）。而深窗冷算一步是 100-700 秒量级 ⇒ **回填窗不是阶梯的主因**，
`restore` 回填加速从刀口序里降级（保留为"复活后前几秒的 tg 为什么低"的解释，不解释整段阶梯）。
取证件：`/tmp/ramp1.sh`（只打印秒偏移与 0-1 的比率，不触碰任何模型文本）。

## 5. 判别臂（dev 2 卡 / 或链后 4 卡空档，各 ~25 分钟）

| 臂 | 内容 | 判据 |
|---|---|---|
| R0 | 4×123K 异源并发 + t+55 s 一条 262K 冷入（`eval/mixedpoison55.py base\|poison`） | 每流 `tg_min/tg_max` 与 `agg`；冷入前后 tg 谷值与恢复时间 |
| R1 | 同 R0，`--long-pipeline-micro 4096`（M4 若赢） | 证明"配置层已把阶梯压平"⇒ rider lane 不必编码 |
| R2 | 同 R0，`--spec mtp --spec-k 2` | MTP 把 ride 乘子抬高，看是否掩盖病灶 |
| R3 | 补丁臂（rider lane，env 默认关） | tg_max/tg_min 收敛到 ≤2.0 且 `agg` ≥ R0 且 262K 冷算墙钟不劣化 >3% |

**采纳线**：R3 必须同时满足"深窗阶梯收敛"与"PP 墙钟不劣化"，否则判负——
单看 tg 会重蹈 W1 的覆辙（tg 好看、墙钟翻倍）。

## 6. 与 17 臂结果的耦合关系

- 若 **S8/S8K2** 赢：末级负载不均是主要矛盾 ⇒ 先采纳切分，再重跑 R0 决定 rider lane 是否还有肉。
- 若 **NV1/NV3** 把 262K C1 冷算从 719 s 打回 ~150 s：说明深窗阶梯的一半来自"冷算段太长"，
  rider lane 的收益分子随之缩小，优先级下调。
- 若 **G2/K2G2** 赢（min-group 2 的形状红利）：组轮转已经给 decode 行更多窗，rider lane 可能冗余。
- 三条都不赢而阶梯仍在 4×：**rider lane 是唯一的把手**，按 §3 开工。
