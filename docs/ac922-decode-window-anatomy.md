# AC922 decode 窗解剖：并发为什么不是白给的

> 白盒定位件（Wave18 收官产出）。对象：4×V100 SXM2、layer-split、NVFP4 线同款 pack 的
> Qwen3.8-Flash-Next 部署。目的：把「开并发聚合不涨、深窗单流互踩」从行为观察钉到代码行，
> 给二开留地图。数字口径见 `ac922-perf-seconddev.md` §11-§12。

## 1. 窗的一生（代码地图）

每轮 tick 走 `Worker::step_piped`（src/engine/lm/pipe.rs:139）：

```
can = prefilling.is_empty() && !self.lent && self.exec.can_pipe()   // pipe.rs:145
```

- `can` 为真且解码序列够分 ≥2 组 → `Pipe::form`（pipe.rs:79）把 slot `0..` 的解码序列切成
  `groups_of`（pipe.rs:112）个连续组，之后每轮 `pipe_round` 轮转推进一组：采样、提交、起草都在
  组步的出口做（pipe.rs:175）。
- `can` 转假（有 prompt 要读 / 缓存在 lend / 组数退化）→ `pipe_drain`（pipe.rs:249）：等在飞
  的组步收尾，序列交回整步（whole step），slot 重排 `0..`。
- 组形状要求「均衡」（pipe.rs:70 `balanced`）：组数 = `groups_of` 且大小差 ≤1，否则同样排空。

`--pipeline-min-group` 默认 2（src/engine/lm/pipe.rs:107）：一组一序列的轮转，每轮为一个 token
付整轮固定成本，双 V100 实测 2 组各 1 反而慢于整步（70 vs 75 tok/s）。四 V100（#752）：C=4 开
4 组各 1，有 MTP 时 110（与 2×2 持平）、无 MTP 94 < 整步 101。**S7 定档 `--pipeline-groups 4
--pipeline-min-group 1` 是有意的：C=4 深窗下买轮转的公平性，不吃亏；带 MTP 时更优。**

executor 侧另有一道独立否决：`can_pipe()`（src/model/core/exec/pipe.rs:165）要求
`!experts().is_lent()`（exec/pipe.rs:169）——**expert 缓存被借出时，轮转态在 model 层也起不来**。

## 2. lend：深窗风暴的真正开关

`Worker::lend_cache`（src/engine/lm.rs:7899）在**长 prompt 进入**时把 expert 缓存的 VRAM 借给
KV/暂存：`self.lent = true`（lm.rs:7903），同时 `set_pipeline_micro` 切到长 prompt 流水。
触发判据在 `plan_step`：prefilling 里有序列 `left >= long.min_tokens` 即 `long` 借（lm.rs:8285）；
`short_min_tokens` 走「只借暂存不借步形」的轻借（输出保持与未借一致）。长 prompt 结束
`restore_cache`（lm.rs:7984）收回 VRAM，槽位在后台回填——**回填完成前，decode 的专家全走 host 读**。

lend 步内的 decode 走 rider 通道（src/engine/lm.rs:8429-8447）：首个 prompt 领跑、其余序列作为
一个 micro-chunk「搭车」，`ride = micro − decoding.len()×n_tok`。**每步只给每条在编序列一次采样**
⇒ lend 期间 decode tg ≈ 1/步时。深窗步时以十秒计，这就是同刻 4 条长会话里「一条快道 ~30、
其余 0.6-2」的机制：快道是最后一个 prefill 完、赶在下次 lend 前排进轮转组的流；其余在被借空的
缓存上按 rider 速率爬。

配套教训（Wave18b W1，判死）：`--long-batch-tokens 512 --long-pipeline-micro 512` 把 ride
频率提上去了，但 lend 步的语义是「每步把所有路由专家 stage 一遍」（lm.rs:8467 注释），chunk 缩小
= 每 token 摊到的专家读暴涨 ⇒ 62K 档 pp 崩 3.4×（墙钟 221→474s）。**配置层没有既不缩步又提
ride 的形态；W2（`--spec mtp --spec-k 4`）能抬乘子（62K −39%）但 verify 行在 123K 深窗有 +26%
pp 代价。**

## 3. 实测签名 ↔ 机制对照

| 签名 | 观察 | 钉到的机制 |
|---|---|---|
| 冷 storm 无批窗 | 异源 4×62K/123K 冷入，tg 谱均匀 [1.2×4]/[0.7×4]，墙钟 pp-bound | `prefilling.is_empty()` 恒假（pipe.rs:145）+ lend（lm.rs:8285） |
| 同刻长会话阶梯 | 123K C4：一条 ~30-35，其余 0.6-2 | rider 每步一采样 + restore 后缓存回填窗（lm.rs:8429, 7984） |
| 8K C8 同步收尾 | 8 流 tg 12.1→30.1 阶梯、全部同一秒完成 | 准入排队（queue_ms 5.6→35.8s）→ 末个 prefill 后组形成 → 轮转均分 |
| reburst 快线 | C1 priming 后 C4 agg 113（123K） | 命中缓存 → 无 lend → 组轮转满速 |
| SLO 与轮转互斥 | `--itl-target` 在 groups>1 时报错拒启 | slo.rs:54-82：组步时延不代表 ITL，故意禁 |

## 4. 刀口（二开候选，性价比序）

1. **slo.rs 解禁**（src/engine/lm/slo.rs:72）：groups>1 时把 ITL 语义改成「组内 token 间隔」，
   让 SLO 预算与轮转共存。改动小，先决条件是步时模型对组步重标定。
2. **lend 态 chunk 自适应**（src/engine/lm.rs:8437 `micro_for`）：decode 在飞时
   `min(long_batch, rate×itl_budget)`，让 rider 采样率有下界而不崩 pp。需要步时模型在线回归。
3. **riders device-round**（src/engine/lm.rs:8429）：采样下推到 device rounds，每 micro-chunk
   都推进在编序列——重，改 executor 图形状缓存（exec/pipe.rs:177 起的 graphs per shape）。
4. **restore 回填加速**：`restore_cache` 的后台回填完成前全走 host 读；回填带宽/顺序可测
   （expert cache hit_rate 日志 1s 粒度可直接量化回填窗长）。

## 5. 复现

- 形状：`scripts/legion-lane.sh 4 restart`（S7 定档；6 卡同理 `6`）。
- 快线：`eval/reburst.py 122880 1 salt` 后 `eval/reburst.py 122880 4 salt`（**必须 C=1 先 priming**）。
- 满窗：`eval/fullwindow.py run --n 4 --gen 256 --total 262144 --warm 1`（判据 held_edge）。
- 毒入：`eval/mixedpoison55.py base|poison`（8×8K 稳态 + t+55s 一条 262K 冷入，同 boot 对比）。
- 观测：引擎日志 `request done` 行（queue_ms/prefill_seconds/decode_tok_s/*_roofline_pct）+
  `expert cache ... hit_rate=` 行；全程只打印数字与 hash。
