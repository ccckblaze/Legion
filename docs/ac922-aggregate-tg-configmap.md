# AC922 聚合 tg 的 config 层地图：哪些旗标真的在动，哪些只是躺在命令行里

配套阅读：`docs/ac922-decode-window-anatomy.md`（代码层的窗成形与 `lend`→rider 病根）。
本文只谈**配置层**：深窗并发上不去，有相当一部分不是引擎不行，而是我们递给它参数的方式让关键路径根本没启用。
下面每条都带源码/帮助文本取证位置，均为 4×V100（`0,1,3,4`，跨 NUMA）实机复核。

## 1. 三条"数字写了但不生效"的截顶链

| 现象 | 代码事实 | 我们的踩法 | 判决件 |
|---|---|---|---|
| `--long-batch-tokens` 是死数字 | `take = min(long_batch, max_batch − 解码行)`（src/engine/lm.rs:9093-9104） | 车道钉 `16384`，而 `max-batch-tokens` UD=8192 / NVFP4=4096 ⇒ 恒被 `max_batch` 截顶 | NV0-NV3（`max_batch` 4096/8192 × `micro` 1024/2048/4096）、LB（`long-batch 65536` 配 `max_batch 8192` 仍无效，须同改） |
| `micro_for` 整条自适应被掐死 | `LongPrompts::default()` = `micro 4096 / micro_per_stage 2.0 / MIN_MICRO 2048`；`clamp(MIN_MICRO.min(micro), micro)` 在 `micro=1024` 下恒为 1024 | 两条车道都钉 `--long-pipeline-micro 1024` | M2（2048）/ M4（4096，复活自适应）/ EV（4096 + `--long-micro-even`） |
| `--pipeline-min-group` 被 launcher 改写 | 上游默认 **2**（src/engine/exec/pipe.rs:107），帮助自述 `1` 才换出"四组各一" | 车道显式钉 `1` | G2、K2G2 |

关键因果：**NVFP4 冷算段比 UD 贵 6.8×（719.4 s 对 106.3 s，同 harness 同协议）很可能不是权重带宽，而是第一行**——每步 prompt 步长被 `max-batch-tokens 4096` 砍半。
若 NV1-NV3 把 719 s 打回，则该票记"配置自伤"，不是硬件差异。

## 2. 前置条件矩阵（旗标 A 需要旗标 B 在场）

- `--long-dataflow-async`、`--long-dataflow-peer`、`--long-micro-even/taper/per-stage` ⇒ 需要 **`--long-dataflow`**。
  两条车道都带 `--long-dataflow --long-dataflow-peer`，所以 P4 的 async A/B 成立。
- `--pipeline-mtp-split` ⇒ 需要 `--pipeline-groups` **且** `--spec mtp` **且** 无 `--layer-split`。
  历史事故：`wave05 W1` 只挂了 `--pipeline-mtp-split`，那代 base 命令行里没有 `--pipeline-groups` ⇒ 空转一整轮。
- `--pipeline-min-group` ⇒ 需要 `--pipeline-groups`。
- `--split-last-extra <N>` ⇒ 需要多卡且无 `--layer-split`；与 MTP **解耦**（只是把自动切分从按字节换成按层计数：48 层四卡，`2`→13/12/12/11，`8`→14/14/14/6）。
  但上游给出的红利表（C=4 129→137、C=8 171→188）是**带 MTP** 测的，且它那句"会亏 1-3%"的前提是"without `--pipeline-groups`"——我们带 `groups 4`，前提不成立。
  ⇒ 同一票必须在 **无 MTP（S2/S4/S6/S8）** 与 **有 MTP（S8K2）** 两态各取一次，缺一半都会误判。
- `--spec-k` 与窗长强耦合：同 harness 实测 8K C4 **+21.5%**（89.5→108.7）、123K C4 **−19.3%**（113.1→91.3）、62K 冷算 wall **−39%**、123K 冷算 **+26%**。
  ⇒ 它不是"开/关"票，是**按窗带切值**的票；任何把 MTP 当全局开关的结论都是错的。

## 3. 先确认旗标打在哪块二进制上

`legion-lane.sh:14` 的默认 `BIN` **不是生产用件**：`lane.env` 把 UD 车道指到 `bin-L2`，`lane-nvfp4-4.env` 把 NVFP4 车道指到 `bin-nvscale`。
默认件里没有 `--conv-disk*`，照它下结论会误判"冷层二开不存在"。

| 件 | commit | 用途 | 本文相关旗标 |
|---|---|---|---|
| `/data/work/legion-src/target/release/legion` | `75029d3` | 默认（勿用于判定） | 无 `--conv-disk` |
| `/data/work/legion/bin-L2` | `6101045` | UD 生产/bench 车道 | 全部命中 |
| `/data/work/legion/bin-nvscale` | `2130b13` | NVFP4 车道 | 全部命中 |

纪律：**任何"这个开关不存在/不生效"的判断，`serve --help` 必须打在该车道的 `LEGION_BIN` 上，并把 `--version` 的 commit 记进臂档案。**

## 4. 冷层使墙钟口径统一（勿跨类互推）

冷层**跨 boot 复用是真的**：`--conv-disk-no-restore` 的存在（"index the cold tier at load but never hydrate，dump-only 供 A/B 二分"）与 `--conv-disk-state-stride`（默认 8：一份满窗会话 32 个 state 留 10，复活最多重 prefill 一个 stride）都说明 boot 即索引。
⇒ 同一 tier 目录上的多臂对比是**公平的**（每臂同等受益），但绝对墙钟是"暖冷层"口径，不能与"空层全新 boot"互相引用。
容量纪律另见 runbook §383：`trim()` 只计 blocks 漏计 states，名义 cap 与 `du` 实测差 1.7×，补丁验收件 = 同 boot 自述从 150.5 GiB 变 ≈269 GiB 且 `du` 收敛。

## 5. 裁决纪律（防"量尺不足判负"）

- 效应 **< 4 ms/窗** 的票禁止用跨 boot 对比判负；boot 间抖动 5-10%，只读差值不读末位。
- `quant_quality` 的 1,000-token 段不得为分块/流水类票背书"数值中性"。
- 非 bitwise 臂（`--split-last-extra` / `--long-micro-even` / 自学 profile）必过三件套：`det ×3` 一致 + acc 漂移哨兵 + 分歧二分探针。
- 聚合判据首选 **`tg_max/tg_min` 是否收敛**（解剖文档的 4× 阶梯），其次才是 `agg rows/s`；`rows/s` 含 prefill 段，与每流 tg 不可互推。
