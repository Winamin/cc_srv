---
license: apache-2.0
language:
- zh
- en
tags:
- claude-code
- llama.cpp
- prompt-cache
- kv-cache
- local-inference
- agents
---

# cc_srv

一个本地的 Anthropic Messages 服务，后端是 llama.cpp。把 Claude Code 指过来，本地模型就能当 agent 用了。

整个设计只赌一件事：agent 会话里，每个请求基本就是上一个请求再加几个 token。四层缓存按从便宜到贵的顺序，把这部分重复吃掉。

单张 RTX 5060 Ti，一个 20 请求的真实会话：15 个请求命中了某一层，591,946 个 prompt token 里 447,657 个（75.6%）根本没进模型。配置是 `CC_ARCHIVE=1 CC_BATCH=1 CC_BATCH_N=3`，4 个序列、`n_ctx` 524288。草稿缓存默认关着，下面的数都是没开投机解码测的。

---

## 快速开始

**先改 `lib.py`。** 顶部两个路径：`DLL` 是你的 llama.cpp 动态库目录，`GGUF` 是对话模型，现在都指向开发这台机器的路径。

```powershell
$env:CC_ARCHIVE='1'   
$env:CC_BATCH='1'; $env:CC_BATCH_N='3'
python run.py --port 8788 --n-ctx 131072
curl.exe http://127.0.0.1:8788/health     # {"ok": true}
```

```powershell
claude --settings D:/path/to/cc_srv/_cc_local.json
```

`--settings` 省不掉。Claude Code 2.1.270 里 settings 文件的 `env` 优先级高于进程环境变量，直接 `$env:ANTHROPIC_BASE_URL=...; claude` 的话，它还是去找 `~/.claude/settings.json` 里原来那个地址。

**`_cc_local.json` 里的 `UserPromptSubmit` 也要改。** 那里写死了这台机器的 Python 和 `cc_hook.py` 绝对路径，路径一过期，每次输入都会触发一个失败的 hook。不想记提示词就把整个 `hooks` 块删了。

请求日志在 `logs/server.log`：

```
2026-09-13 20:22:53 [INFO ] [req] req 52039+41 batch_apc pre=77 reuse=51962 p_raw=- cur=52079 1.0s (prefill 0.0s + decode 0.0s) turn=text:e24e0fc8 seq=0
```

52,039 token 的提示词，51,962 走缓存，只转了 77 个，1.0 秒。批处理路径下 prefill/decode 的拆分是 0，只有总数可信。

---

## 缓存对一次会话做了什么

两张图都是上面那 20 个请求，颜色代表命中哪一层。

![每个请求转发的 prefill token 数](fig_prefill.png)

冷启动要转完整提示词，中位数 **22,641 token**。archive 有状态的只转 **1,232**。KV 里本来就有这个序列的只转 **71**。

![墙钟时间对回复长度](fig_latency.png)

回复长度是缓存够不着的部分。拿它当横轴，冷启动（带圈）在每个长度上都比缓存点高，高出来的就是多付的 prefill。往上爬的那两个缓存点（11 秒、24 秒）是 600 到 950 token 的长回复，这种只能慢慢生成，缓存帮不上。

回复长度在 30 到 106 token 之间时：

| 由谁服务 | 回复 token | 转发的 prefill | 墙钟时间 |
|---|---:|---:|---:|
| 冷启动 | 100 | 22,646 | 11.3 s |
| 冷启动 | 106 | 22,641 | 11.3 s |
| 前缀 archive | 81 | 1,055 | 2.0 s |
| 前缀 archive | 32 | 2,324 | 1.7 s |
| 前缀 archive | 37 | 1,048 | 1.4 s |
| 前缀 archive | 30 | 170 | 0.6 s |
| 序列内 | 41 | 77 | 1.0 s |
| 序列内 | 41 | 66 | 0.9 s |

**81 个回复 token 2.0 秒，100 个回复 token 11.3 秒。** 最短的几条缓存回复一秒内就回来了。

---

## 吞吐

算法定死了：墙钟时间里服务了多少上下文 token，两边同一个公式。冷启动的墙钟基本全是 prefill，缓存命中的墙钟基本全是生成，所以缓存那一列其实是吃亏的算法。

![上下文吞吐：冷启动 prefill 对缓存复用](fig_speed.png)

| | 中位上下文 tok/s |
|---|---:|
| 冷启动 prefill，`cc_srv` | 2,004 |
| 冷启动 prefill，LM Studio | 1,832 |
| 两个序列竞争，LM Studio | 956 |
| **缓存复用，`cc_srv`** | **24,368** |

中位数差约 12 倍。复用那一列跨度 3,972 到 52,039：最低的是个 51,244 token 的提示词，archive 手里没有它的分叉点，照样转了 22,677 token，即便这样还是比图上每个冷启动都快；最高的一项复用了 99.9%。

进表的是那次会话里生成 ≤106 token 的 15 个请求，墙钟都由 prefill 主导。生成 617 到 948 token 的 5 个没进，它们的墙钟是生成说了算。引擎日志里还有六次冷启动 prefill，提示词 11,589 到 71,070 token，落在 1,846 到 2,318 tok/s 之间。

两个 caveat，先说清楚这些数不是什么：

- **没开 MTP。** 模型自带多 token 预测头，这里从来不用，`lib.py` 的 `mtp` 就是 llama.cpp 默认的关。每个数字都是老老实实的单 token 前向。
- **基线没调过。** LM Studio 那几行是同一张卡同一个 GGUF，不带缓存栈。光在这几行内部，两个序列一并发 prefill 就从 1,832 掉到 956 tok/s。这也是为什么这里的批处理只敢说赢延迟、不说赢吞吐。

---

## 工作原理

| 层 | 触发条件 | 代价 |
|---|---|---|
| **Logits 缓存** | 提示词逐 token 完全一样 | 不前向，不占 KV |
| **轨迹召回** | 提示词是某条已记录轨迹的前缀 | 不前向，不占 KV |
| **轮次键复用** | 结尾的用户轮次以前被回答过 | 不前向，不占 KV |
| **APC** | 有公共前缀就行 | 只转新增的部分 |
| 冷启动 | 其他所有情况 | 完整 prefill |

前三层吐出来的 token 没有一个是现算的。长会话里的大头是 APC，因为对话的提示词本来就是"上一轮加几个 token"。

**轮次键复用。** Logits 缓存拿整个提示词当键，对话里基本命中不了，每轮都追加了上一轮的问答。改成拿结尾的用户轮次当键，重复的问题就能答了。`CC_QREUSE=1` 只认同一对话里的同一个问题；`CC_QREUSE=2` 配 `CC_QEDIT=1` 还认跨对话的**相似**问题，把旧回复改一改再给。每个生成 token 背后留一份 top-k logits 轨迹，argmax 重放不花前向，改写时的每次替换都拿模型自己在那个位置的缓存 top-k 验过。

**Claude Code 的提示词有三处专杀前缀缓存。** 都修了：

- 提示词里塞着 `<total_tokens>N tokens left</total_tokens>`，`N` 每个请求都变，前缀到它第一次出现就断了。服务端先归一化，`_cc_local.json` 再把 `CLAUDE_CODE_TOTAL_TOKENS_REMINDER` 设成 `infinite`，让它彻底不变。
- 生成时用的助手头，重渲染历史时必须逐字节一致。从模型自己的 chat template 里取，不手写换行。
- 助手消息经常带空文本块。直接拼接会多出两个换行，光这一项就能抹掉上千 token 的复用。丢掉。

### Thinking，以及 `CC_THINK=0` 为什么是锁

模型把推理写进 `<think>...</think>`。默认不发给客户端，但"不发"不等于"关"。推理有三个独立开关，只关一个等于没关：

| 来源 | 怎么打开推理 | `CC_THINK=0` 管不管 |
|---|---|---|
| 运行时开关 | `GET /think?on=1` | 管 —— 请求被**拒绝**，不是被执行 |
| 客户端 | 请求里的 Anthropic `thinking` 字段，Claude Code 会发 | 管 |
| 历史 | transcript 里的 thinking 块，重渲染进提示词变成 `<think>...</think>` | 管 —— 推理正文直接丢掉，token 花销一起丢 |

第三个最容易漏，也最要命：把 transcript 的推理重发一遍，本身就是"继续推理"的指令。只剥输出不剥输入的服务，从第一个 thinking 轮之后每轮都在让模型推理。

所以 `CC_THINK=0` 启动时读一次，当硬锁用。设了之后 `want_think()` 永远返回 `False`，`/think?on=1` 回 `{"think": false, "locked": true}`，历史按推理剥掉之后渲染。也就是裸的闭合空块 `<think>\n\n</think>\n\n`，非 thinking 路径也是这个形状，前缀保得住。想开回去只能重启加 `CC_THINK=1`，运行时调用能解开的锁不叫锁。

有个不对称先说清楚。闭合空块跟模型说的是*"你的推理已经结束了"*，而不是"不许推理"，因为重渲染的历史必须逐字节对上。`CC_THINK=0` 只保证两件事：客户端收不到推理，记过的推理不再进提示词。模型内部到底怎么算的，不打包票。

换这个设置会换提示词头，前缀跟着变，下一请求付一次冷启动。这是换了题目的代价，服务端躲不掉。锁着还要推理的请求记的是 `locked` 不是 `req`，拒绝看得见，不会伪装成命中。

### 前缀之外

**前缀 archive。** 默认开，`CC_ARCHIVE=0` 关。兄弟 subagent 共享很长的前缀，但谁也不是谁的延续，普通前缀缓存管不着。archive 在请求**真正分叉**的地方存整个 KV 状态，用的时候整块恢复。上面那次会话它包了 20 个请求里的 7 个；最典型的是 24,775 token 的提示词只转了 **2,324** 个，剩下 22,451 个都在归档状态里。

归档槽位按 `CC_ARCHIVE_CAP` 计费，默认半个窗口。槽里存的都是别人 prefill 好的状态，值得留的分叉点本来就短。实测十个兄弟 agent 真正共享的前缀是 16,406 token，归档边界落在 16k–42k，对着 131072 的窗口按整窗收就是浪费。省下来的是槽位本身：`CC_ARCHIVE_SLOTS=2` 现在只花过去一个槽位的显存。代价是超过 cap 的长边界不归档（记到 `arc_skip_cap`），那些请求回去重新 prefill。丢的是复用，不是正确性。`CC_ARCHIVE_CAP=0` 就回到整窗计费的老布局。

**批处理。** 9B Q4 的解码步不管吐几个 token 都要把 5.5 GB 权重读一遍，一次读顺手推多个序列就是赚的。`CC_BATCH=1` 让并发请求共享前向，解码吞吐是逐个服务的 3.3 倍。

### 两个批大小为什么分开

`CC_N_BATCH` 和 `CC_N_UBATCH` 默认 2048 和 512，对齐 LM Studio 在同一张卡同个模型上的*评估批处理大小*和 *Physical Batch Size*。这样两边开箱就是可比的。以前硬编码 512 和 128。

| | `CC_N_BATCH` (2048) | `CC_N_UBATCH` (512) |
|---|---|---|
| 限制什么 | 一次 `llama_decode` 的 token 数 | 一次 GPU 前向的 token 数 |
| 提升什么 | Python 侧的调度开销 | **prefill 吞吐** |
| 显存开销 | 几乎为零 | **线性涨** —— compute buffer 是 `n_ubatch × n_embd` |
| 还有谁用它 | `Batch._plan` 拿它算每步 token 预算 | 没人 |

两个不能混用。`n_batch > n_ubatch` 时 llama.cpp 会把逻辑批拆成多次前向，所以**光调大 `n_batch` 除了省点记账什么都买不到**，真正干活的是 `n_ubatch`。它也是吃显存的那个：`n_ctx` 524288 配 Q4_0 KV 是真会爆的。`n_ubatch > n_batch` 启动直接拒绝，报错会把两个设置都点名，不是一个光秃秃的 assert。

还有一处耦合是故意拆开的。调度器原来每个 prefill chunk 取 `eng.bs // 4`，`n_batch` 512 时是 128；`n_batch` 提到 2048 它就变成 512。一个 prefill 从一步里拿走 512 token，正是测出 **p90 199 秒**的那种死法：刚到的提示词吃光整步，已经在解码的序列一个 token 都分不到。现在 chunk 独立封顶 128，`CC_BATCH_CHUNK` 可以直接改。**每步的总预算**还是随 `n_batch` 走的，那本来就是它该管的事。

### 跟 LM Studio 加载面板逐行对

引擎每次启动都打印实际生效的参数，对着日志看就行：

```
load settings: n_batch=2048 n_ubatch=512 n_seq_max=4 kv_unified=True n_rs_seq=0
offload_kqv=True op_offload=True swa_full=True threads=4/4
load_mode=-1 lazy_mode=1 flash_attn=-1 rope=0.0/0.0
```

| LM Studio | 对应字段 | 实际值 |
|---|---|---|
| CPU 线程池大小 4 | `n_threads` | 4/4 —— llama.cpp 默认就是 4 |
| 评估批处理大小 2048 | `n_batch` | 2048，`CC_N_BATCH` 给的 |
| Physical Batch Size 512 | `n_ubatch` | 512，`CC_N_UBATCH` 给的 |
| Max Concurrent Predictions 4 | `n_seq_max` | `CC_BATCH_N=3` 时是 4（3 worker + 1 archive 槽） |
| Unified KV Cache 开 | `kv_unified` | 开 |
| 将 KV 缓存卸载到 GPU 开 | `offload_kqv` | 开 —— llama.cpp 默认 |
| 保持模型在内存中 / 尝试 mmap | `load_mode`、`lazy_mode` | `-1`/`1`，都是 AUTO —— llama.cpp 默认 |
| RoPE 频率基/比例 自动 | `rope_freq_base`、`rope_freq_scale` | `0.0`/`0.0`，0 就是用模型自带的 |
| Context Checkpoints 32 | —— | **没有对应项**，见下 |

除最后一行都已经对上了，而且大多是因为**这些本来就是 llama.cpp 的默认值**。那个面板基本就是在展示库的默认行为。cc_srv 真正钉死的只有四个和默认值不一样的：`n_batch`、`n_ubatch`、`kv_unified`、`n_seq_max`。

**Context Checkpoints 没有对应项，也不可能有。** 它是 `--ctx-checkpoints`，llama-server 的**槽位**选项：`llama_context_params` 里压根没这个字段，实现在 llama-server 的服务端代码里，直接绑 `llama.dll` 的调用方根本走不到。库层面最接近的是 `n_rs_seq`（每序列回滚快照数），`CC_N_RS_SEQ` 能设。但没用：设 32 上下文确实回读 32，部分 `llama_memory_seq_rm` 照样返回 false，序列长度动都不动（`bench/_diag_rs_seq.py`）。那些快照喂的回滚路径只有投机那条支线能走，所以保持 0。cc_srv 的解法是**前缀 archive**，路子不一样：在分叉点存整个状态、整块恢复，而不是把某个序列倒回去。

顺带这个构建里**已经没 `use_mmap` / `use_mlock` 了**。llama.cpp 把它们并进 `load_mode` 和 `lazy_mode` 两个枚举，上表写的就是这两个。

### 模型自带的 chat template

这个模型 `tokenizer.chat_template` 里塞了 7950 个字符的 Jinja。引擎读它，但**不执行它**。之前是把字符串丢给 `llama_chat_apply_template`，那个函数头文件里自己写着：

> NOTE: This function does not use a jinja parser. It only support a pre-defined
> list of template.

这个模板不在 54 个内置里，于是调用**一声不吭回落成裸 ChatML**。拿一个光秃秃的用户轮实测（`bench/_diag_template.py`）：

```
llama_chat_apply_template   <|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n
真 Jinja                    <|im_start|>system\nYou are Qwythos, a model created by
                            Empero AI. ...<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n
                            <|im_start|>assistant\n<think>\n
```

**模型一直不知道自己是 Qwythos。** `src/chat_template.py` 用真正的 Jinja 引擎渲染，跟 llama-server 一个做法。

同一段对话跑两轮，引擎预热过，system 前导 2005 token：

| `CC_TMPL` | `CC_THINK` | 渲染器 | 第 2 轮转发 | 命中复用 |
|---|---|---|---|---|
| `0` | 0 | 只用 library | 20 | 2007 |
| `1` | 0 | 模板的 system 块 | 20 | 2035 |
| `2` | 0 | 模板全权渲染 | 2048 | 0 |
| **`2`（默认）** | **1（默认）** | **模板全权渲染** | **18** | **2083** |

第 1 轮 `CC_TMPL=1` 下 2033 token，`0` 下 2005。多出来的 **28 个 token 就是那条身份消息**，区别仅此而已。

#### 模板和 thinking 是同一个开关

`CC_TMPL=2` 能命中，`CC_TMPL=2` + `CC_THINK=0` 不能。根子和 CoT 被破坏是同一个。

模板**照着内容重建**已完成的 assistant 轮。下一轮拼出的提示词要想对上 KV，推理过程得先被带回来；而推理过程只有客户端拿到过才会被带回来：回复里的 Anthropic `thinking` 块就是客户端下一轮回传的东西。

thinking 锁掉之后，`prepare()` 找模板要的是闭合的空推理块 —— `<|im_start|>assistant\n<think>\n\n</think>\n\n`。这**不是藏起了推理，是在告诉模型「你已经推理完了」**，于是它直接作答。客户端手里没有 thinking 块可回传，模板就把那一轮渲染成 `<|im_start|>assistant\n…`，跟 KV 里存的对不上。

这个分叉是要命的，不光是贵：这里的复用不是 LCP，提示词**延长**了序列已有的内容才复用，混合记忆上的部分截断等于没做。LCP 找到了也被拒。请求日志里 `p_raw=2026` 旁边 `reuse=0` 就是这么来的。任何位置一分叉，后面全得重算。

所以 **`CC_TMPL=2` 就得开 thinking**，配反了 `srv.py` 启动会警告。thinking 非锁不可就用 `CC_TMPL=1`：assistant 轮走引擎自己的渲染器，闭合块能重现，正文照样对得上 KV。system 块还是模板的，身份消息也在。

**投机解码。** `CC_DDC` 拿模型已经吐过的延续当草稿，再批量验。草稿来自上下文本身，成本只剩验证：落地的轮次约快 3 倍，不落地的净亏。跑不跑两道闸门说了算。

第一道按草稿长度分桶，逐桶跟逐 token 解码打。某次实测 2 到 3 token 的草稿只跑到 0.56 倍，丢掉；16+ token 的跑到 3.3 倍，留下。

第二道看内容到底重不重复。建草稿索引要在每个已提交 token 上扫全词表，实测吃掉生成时间的 **7.9%**，只有草稿落地才回本。所以**生成流一不重复 DDC 就让开，一重复就回来**。三个提示词，tok/s：

| | 普通 | 普通 | 重复 |
|---|---:|---:|---:|
| `CC_DDC=0` | 69.2 | 69.8 | 69.6 |
| DDC 开，闸门关（`CC_SPEC_GOV=0`） | 58.1 | 59.0 | 128.8 |
| **DDC 开，闸门生效（默认）** | **66.6** | **67.9** | **141.5** |

不重复内容上 13% 的损耗压到 3%，重复内容上 2.0 倍收益还在。

划不划算看**接受率**，跟上下文长度没关系。整份草稿落地的轮次，比逐 token 解码快 3 倍左右；只落地一半的轮次，两个前向换一个 token。所以 `CC_DDC_MAX_CTX` 默认就是序列自己的窗口，不会提前掐 DDC，只留"草稿不许越过上下文末尾"这一个钳制。

开之前两件事先知道：

- **默认关，跟 archive 一起跑。** 以前草稿缓存是默认层、archive 是选配，现在反过来了：archive 吃的是这个服务实际面对的形状（fan-out），投机只是内容重复时才划算的吞吐开关。以前它俩抢同一个备用序列所以互斥；现在验证 scratch 自己单占一个序列，archive 留自己的，布局不打架，`CC_ARCHIVE=1 CC_DDC=4`（archive 反正默认开，直接 `CC_DDC=4` 也一样）能一起上。代价是 archive 之外再多一个序列的 KV：scratch 按一整窗收，archive 槽各按 `CC_ARCHIVE_CAP` 收。请求先过 archive 做 prefill，草稿缓存再在恢复出来的前缀上跑；填充 archive 的也是这次 prefill。
- **批处理里的投机**走 `CC_DDC_BATCH=1`。不开它，DDC 只在串行路径跑：`CC_BATCH=1` 会把所有请求送进调度器，串行生成路径一次不进，`CC_DDC` 设了也白设。引擎启动时会明说，而不是让计数器悄悄归零。

  开了之后，一次验证就是该步 batch 的一部分。前向本来就是调度器拿来共享的。草稿先拷到该任务名下的 scratch 序列上。`CC_DDC_BATCH_SCRATCH`（默认 1）限制同时投机的请求数，每个 scratch 吃一个序列的 KV，`n_ctx` 131072 下一个 1.2 GB。只接受了一部分的轮次**直接丢掉**，不提交：提交要把接受的前缀在主序列上重解一遍，这个路径不花第二个前向。governor 收到的是"本轮零产出"，所以老是半吊子的草稿长度会被同一道闸门淘汰。

  **这条路径刚写完，还没跑过。** 默认关就是因为这个。串行路径的每个数都是跑出来才写的；这条是对着调度器的不变量写的加复查过，没执行过。先看 `ddc_dropped` 和 `spec`/`full` 这几个计数。

---

## 配置

全是环境变量。

**默认开着的：**

| 变量 | 默认 | 作用 |
|---|---|---|
| `CC_ARCHIVE` | 开 | 前缀 archive。`CC_ARCHIVE=0` 关 |

**想开再开：**

| 变量 | 默认 | 作用 |
|---|---|---|
| `CC_DDC` | 关 | 投机解码；`2` 或 `4` 打开，数字是草稿缓存键宽。在 archive 的槽之外再占一个序列 |
| `CC_DDC_BATCH=1` | 关 | 让 DDC 进批处理调度器里投机，不在 `CC_BATCH` 下让位。刚写完没跑过 —— 见 DDC 一节 |
| `CC_QREUSE=1` | 关 | 复用对重复用户轮次的回复 |
| `CC_BATCH=1` | 关 | 并发请求打进一个批次，规模 `CC_BATCH_N` 定 |

**调优：**

| 变量 | 默认 | 作用 |
|---|---|---|
| `CC_ARCHIVE_SLOTS` | 1 | archive 留几个序列 |
| `CC_ARCHIVE_CAP` | 窗口/2 | 单个 archive 槽最多算多少 token 的 KV。`0` 按整窗收，也就是统一 KV 池之前的老布局。再开槽位就按这个单价加 |
| `CC_ARCHIVE_CAP_DIV` | 2 | 上面默认值的除数，`CC_ARCHIVE_CAP` 没设时生效 |
| `CC_ARCHIVE_MIN` | 512 | 比这个短的前缀不值得归档 |
| `CC_BATCH_N` | 4 | worker 序列数，也就是并发请求数 |
| `CC_BATCH_WAIT` | 512 | 不想等最忙的 worker，最多让出多少前缀 token |
| `CC_N_BATCH` | 2048 | 逻辑批：一次 `llama_decode` 最多带多少 token。也决定调度器每步的 token 预算 |
| `CC_N_UBATCH` | 512 | 物理批：GPU 一次前向实际算多少 token。必须 ≤ `CC_N_BATCH`。显存线性涨 |
| `CC_N_RS_SEQ` | 0 | 每序列回滚快照数（`n_rs_seq`）。上下文收这个值，但部分 `seq_rm` 照样失败（见对照表），所以一直关着 |
| `CC_TMPL` | 2 | 谁来拼提示词。`2` 让模型的 Jinja 模板全权渲染；`1` 只取它的 system 块、正文自己拼（`CC_THINK=0` 时的答案）；`0` 纯 library |
| `CC_THINK` | 1 | thinking。`1` 把推理当 Anthropic thinking 块发下去 —— `CC_TMPL=2` 要命中缓存就得这样，客户端只能回传它拿到过的东西。`0` 是硬锁，配 `CC_TMPL=2` 会警告 |
| `CC_BATCH_CHUNK` | 128 | 单个 prefill 最多从一步里拿多少 token。故意不跟 `CC_N_BATCH` 挂钩 |
| `CC_PROGRESS` | 1 | 报 prefill 进度；`0` 关 |
| `CC_PROGRESS_STEP` | 10 | 进度日志的百分比间隔；`0` 也等于关 |
| `CC_DDC_M` | 32 | 草稿长度上限；一般先卡住的是截断阈值 |
| `CC_DDC_T` | 0.5 | 草稿截断阈值，越低草稿越长。`M=32/T=0.5` 时一轮 15 到 22 token |
| `CC_DDC_MAX_CTX` | 窗口长度 | 草稿缓存让开的长度；默认就是该序列自己的窗口，`0` 连钳制都去掉 |
| `CC_SPEC_GOV` | 1 | 闸门；`0` 是有草稿就投机 |
| `CC_SPEC_MIN` | 12 | 判定某个草稿长度之前要攒多少轮证据 |
| `CC_SPEC_COOLDOWN` | 64 | 被丢的长度多久后重测；每次翻倍 |
| `CC_SPEC_REP_MIN` | 0.15 | 重复率下限，低于它 DDC 让开 |
| `CC_SPEC_REP_GRACE` | 128 | DDC 让开之前，每轮生成的宽限 token 数 |
| `CC_DDC_BATCH_SCRATCH` | 1 | 批调度器能拿去做 DDC 验证的 scratch 序列数。多一个就多一个请求能同时投机，多花一个序列的 KV |
| `CC_QEDIT` | 关 | 留着二级复用改写要用的 top-k 轨迹 |
| `CC_QEDIT_SIM` | 0.9 | 二级轮次的相似度下限 |
| `CC_QREUSE_TOOL` | 0 | 允许复用对工具结果的回复；见下 |
| `CC_STREAM` | 1 | 边生成边流式吐 |
| `CC_LOG` | `info` | `debug` / `info` / `warn` / `error` / `off` |
| `CC_LOGDIR` | `./logs` | `server.log` 和 `requests.jsonl` 放哪 |

archive 和批处理一起开：

```powershell
$env:CC_ARCHIVE='1'
$env:CC_QREUSE='1'
$env:CC_BATCH='1'; $env:CC_BATCH_N='3'
python run.py --port 8788 --n-ctx 131072
```

`n_ctx` 是 **KV 池**，不是单个请求能发多长。池子按"干活的序列各留一整窗、archive 槽按 `CC_ARCHIVE_CAP`（默认半窗）收"来算：3 worker 加 1 archive 是 `131072` × 3 + `65536`，KV 约 4.2 GB 而不是 4.8。再开 archive 槽就按 cap 单价加，所以 `CC_ARCHIVE_SLOTS=2` 正好是过去单个整窗槽位的显存。`CC_ARCHIVE_CAP=0` 精确回到老布局。卡紧就调小 `--n-ctx`。

### 哪几层该关

- **轮次键复用**拿模型以前对同一问题的回复直接作答。对话重放、问题真重复时没问题，但这是**拿语义当正确性用**：没任何东西在当前上下文下重推过那个回复。锋利的是二级。Claude Code 的问题模板话术太多，两个**完全不同**的问题相似度能到 0.65，下限一松就拿甲的答案回答乙。下限默认 0.9，差异里找不出可替换项也直接拒绝。每个回复都必须在当前上下文下现算，就让 `CC_QREUSE` 留在 0。
- **对工具结果的回复**（`CC_QREUSE_TOOL=1`）默认关。这类回复里一般就带着产生该结果的**工具调用**，一重放就重新发起调用，对话直接打转。
- **批处理**改图形态，回复可能跟单序列不一样。约一半提示词在某处分叉，而且是确定性的：引擎文档里写明的批处理对顺序执行的数值差，不是竞态。回复必须跟单序列一致就别开 `CC_BATCH`。
- **投机解码**只在草稿落地时划算：重复代码、重复工具调用、反复读同一文件。闸门在别处会关它，所以关它的理由一般不是速度，是**确定性**。一轮投机走另一条数值路径，回复必须跟纯单 token 服务逐位一致就设 `CC_DDC=0`。
- **前缀 archive 默认开，`CC_DDC` 默认关**，跟以前反过来了。开 DDC 要在 archive 之外再占一个序列。archive 留自己的槽，DDC 留自己的验证 scratch，各花各的（scratch 一整窗，archive 槽各一个 `CC_ARCHIVE_CAP`）。两个开关互不影响。

---

## 文件

```
run.py         入口
srv.py         HTTP 层、Claude Code 适配器、请求日志
eng.py         引擎：KV、APC、logits 缓存、召回、archive
batch.py       调度器：多请求，每步一次前向
qcache.py      轮次键复用与 argmax logit 重放
spec.py        投机解码的两道闸门：草稿长度，以及内容到底重不重复
ddc.py         余量感知的草稿缓存
ddc_decode.py  目标模型验证式草稿解码
stream.py      边生成边流式输出
log.py         统一日志，含 llama.cpp 的 C 层输出
cc_hook.py     记你输入的提示词的 Claude Code hook
lib.py         加载 llama.cpp 动态库
```

要 CUDA 构建的 llama.cpp（动态库从 `lib.py` 里的路径加载）和 GGUF 对话模型。开发和测量都是 RTX 5060 Ti 16 GB，Qwythos-9B-v2-MTP Q4_K_M 配 Q4_0 KV，Claude Code 2.1.270 驱动。
