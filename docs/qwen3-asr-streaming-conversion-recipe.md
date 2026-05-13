# Qwen3-ASR 流式改造移植 recipe

> 把 Qwen3-ASR 从一次性（one-shot）转写改造为流式（streaming）转写的可移植配方。
> **不依赖具体推理后端**（EdgeLLM / vLLM / ONNX Runtime / TensorRT-LLM / llama.cpp 等都适用）。
> 基于 2026-05 在 NVIDIA Jetson Orin NX + TensorRT-Edge-LLM 上的实战经验提炼。

## 0. 一句话结论

**Qwen3-ASR 没法做真正的 sub-300ms 首词低延迟流式 ASR，除非重训 encoder。但用 chunk-and-confirm + prefix prompt 机制，可以做到 ≤ 500ms 的「说完到出文本」延迟，这才是产品 UX 真正关心的指标。**

## 1. 背景与关键架构事实

### 1.1 Qwen3-ASR 不是天生流式的

Qwen3-ASR 是 encoder + thinker（autoregressive LM decoder）架构，原始设计是 one-shot：整段音频一次喂入。官方虽然在 vLLM 后端提供了 `streaming_transcribe` API，但本质也是一次又一次跑 one-shot，**不是真正的流式 encoder 推理**。

官方实现位置：`Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:584-829`（`init_streaming_state` / `streaming_transcribe` / `finish_streaming_transcribe`）。

### 1.2 Encoder 内部结构（必须理解才能正确移植）

```
mel input [num_chunks, 128, 100]
   ↓
3× Conv2d(stride=2, padding=1, kernel=3, out_channels=480)
   ↓ 每个 100-mel 块独立卷积、两边零填充
post-CNN tokens [num_attention_elems, 1024]  (每 100-mel 块→13 token)
   ↓
Linear(7680, 896) + positional embedding (chunk-local)
   ↓
18× transformer layers  
   ↓ block-diagonal attention via cu_seqlens
   ↓ 每个 attention "window" 覆盖 n_window_infer=800 mel = 8 个块 = 104 token
   ↓ 窗口内 bidirectional, 窗口间不相互 attend
ln_post + proj1 + GELU + proj2 → output [N, 1024]
```

**关键事实（决定一切流式策略）**：

1. **Conv 是按 100-mel 块独立做的**，不是 contiguous（这一点反直觉，新手容易猜错）
2. **Transformer 的 `cu_seqlens` 是 bidirectional**：每个 attention window 内所有 token 互相 attend
3. **Attention window 默认 800 mel = 8 块 = 8 秒**（视 hop 而定，但 mel hop 一般是 10ms）
4. **Encoder 没有持久化的内部状态**：每次 forward 都从零开始

参考代码：`Qwen3-ASR/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py:622-624,681-732`

### 1.3 Thinker（decoder）

标准 autoregressive LM。重要参数：
- `max_input_len` — prefill 时单次输入 token 数上限。**Qwen3-ASR 0.6B 默认 128**，太小，对流式有强约束（见 §4.2）
- `max_kv_cache_capacity` — KV 缓存总容量。默认 256

## 2. 为什么"切 encoder"做流式行不通

### 2.1 朴素切分（dead path #1）

直觉做法：客户端推 1s 音频 → encoder 跑一次 → thinker prefill 1s 音频 token → decode 一段 partial → 客户端再推 1s → encoder 又跑一次（只 encode 新的）→ thinker 继续 prefill → decode 更长 partial...

**为什么不行**：Encoder 的 attention `cu_seqlens` 每次 call 都按当前 call 的输入大小重新构造。切 1s 一段，每段 attention window 只覆盖那 1s 内的 token。one-shot 时这些 token 能看到完整 8s 窗口的上下文；切开后看不到 → 信息丢失。

**实证数据**（2026-05-13 在 Orin NX 上的 LCS empirical test）：

| chunk 块数 | 输出长度 | LCS-similarity vs one-shot |
|---|---|---|
| 1 (1s) | 7 token | **0.368** |
| 2 (2s) | 10 token | 0.526 |
| 4 (4s) | 14 token | 0.737 |
| 8 (8s = 1 full window) | 19 token（参考）| 1.0 |

**失败模式**：thinker 在切开的弱上下文 audio embedding 上 emit EOS 过早。输出是参考的**严格前缀**截断，不是随机噪声。

### 2.2 各种"补救"方案为什么也不行

| 方案 | 为什么不行 |
|---|---|
| 加 conv KV cache | 现有 encoder conv 是 per-100-mel-block 独立的，加 cache 解决的是不存在的问题 |
| 加 attention KV cache | 中间 chunk 受益，但**首 chunk 没过去、末 chunk 没未来**，物理上信息缺失，cache 救不了 |
| Attention overlap-and-discard | 同上，首 chunk 无未来 |
| 切 = 1 full window (8 块) 强制对齐 | 数学上 work，但首词延迟 ≥ 8s，违背流式初衷 |
| 改成 chunk-causal attention（推理时 mask） | encoder 训练时是 bidirectional 的，推理时强加 causal mask 质量崩 |

### 2.3 唯一干净的路：retrain chunk-causal encoder（P2 工作）

LoRA fine-tune 在 q/k/v/out_proj 上加 causal mask 训练 1-2 周可能搞定 LCS≥0.95，全量 fine-tune + 数据 4-6 周。**P0 不做**，留作未来真低延迟（<300ms 首词）需求触发。

## 3. P0 推荐方案：Chunk-and-Confirm with Prefix Prompt

### 3.1 机制总览

**直接镜像官方 vLLM 实现**（`qwen3_asr.py:584-829`）。**完全不动 encoder / 不切 encoder / 不加 cache**。每个 hop 跑一次 full one-shot ASR 在不断增长的音频 buffer 上。

```
state = {
    audio_accum: PCM 累积 buffer,
    raw_decoded: "",                    # 上一轮完整 decode 输出
    chunk_id: 0,
    chunk_size_sec: 0.5,                # hop 长度，可调
    unfixed_chunk_num: 2,               # 前 N hop 不用 prefix
    unfixed_token_num: 5,               # 回退末尾 K token
}

每个 chunk 事件:
    audio_accum.append(pcm)
    while accumulated_since_last_hop >= chunk_size_sec * sample_rate:
        run_one_hop(state)

run_one_hop(state):
    # 1. 构造 prefix
    if state.chunk_id < unfixed_chunk_num:
        prefix = ""
    else:
        tokens = tokenizer.encode(state.raw_decoded)
        k = unfixed_token_num
        while True:
            end = max(0, len(tokens) - k)
            prefix = tokenizer.decode(tokens[:end]) if end > 0 else ""
            if '�' not in prefix: break       # UTF-8 守护
            if end == 0: prefix = ""; break
            k += 1
    
    # 2. 跑 ASR（基础推理后端的 one-shot API）
    prompt = base_asr_prompt + prefix
    response = backend.transcribe(prompt=prompt, audio=state.audio_accum)
    
    # 3. 更新状态
    state.raw_decoded = prefix + response.text
    state.chunk_id += 1
    emit partial(parse(state.raw_decoded))

last=true 时:
    run_one_hop(state)
    emit final(parse(state.raw_decoded))
    清理 session
```

### 3.2 为什么这能 work

| 设计点 | 解决的问题 |
|---|---|
| 每个 hop 都跑 full one-shot | encoder 每次都看到完整 bidirectional 上下文，无信息丢失 |
| Prefix prompt = 上一轮文本 - 末尾 K token | thinker 把已确定的 prefix 走 prefill（不浪费 decode）；只需要 decode 末尾几个 token + 新音频对应的新 token |
| `unfixed_chunk_num` 前几个 hop 不用 prefix | 早期 hypothesis 不稳定，给模型自由 |
| `unfixed_token_num` 末尾回退 | 末尾 K token 留给"还可能改"的不确定区，后续 hop 拿更多音频后可以推翻 |
| UTF-8 retry | 中文/多字节 token 边界不能切在字符中间 |

### 3.3 关键性能优化：prefix prompt 替代重 decode

朴素 LocalAgreement 每个 hop 都从零 decode 整段。**Prefix prompt 让 thinker 把已确定文本走 prefill（O(1) per token，远快于 decode 的 O(1) per token but with sampling/sync overhead）**，只 decode 真正不确定的尾巴。

实测在 0.6B Qwen3-ASR + Orin NX 上：
- Hop 5（无 prefix）：50 token decode × 30 ms/step ≈ 1500 ms
- Hop 5（有 prefix，K=5）：5 token decode + prefix prefill ≈ 200-300 ms

**5× 性能提升，可上线 vs 不可上线的差距**。

### 3.4 SLI：end-of-speech 延迟，不是 first-partial

产品 UX 关心的是「用户说完到看到最终文本」，不是「第一个字什么时候出现」。partials 在用户说话过程中已经在流式更新，first-partial 延迟无关紧要。

**target SLI（Orin NX 0.6B 实测目标）**：
- end-of-speech 中位 ≤ 500 ms
- end-of-speech p95 ≤ 1000 ms

延迟拆解（5s utterance、hop=500ms）：
- 未处理音频尾巴（last_hop → last=true）：平均 250ms，worst 500ms
- Encoder（full buffer 5s = 5 block）：~10 ms
- Thinker prefill（prefix text + audio embeddings）：~50-100 ms
- Thinker decode（末尾 K=5 token）：~150 ms
- **合计 ≈ 260-760 ms**

## 4. 通用工程实施 recipe（任何后端都适用）

### 4.1 5 步实施（worker / serving 层）

**Step 1 — 事件分派 scaffold**
- 输入 stdin / WebSocket / HTTP chunked：定义 `begin / chunk / end` 三种事件
- 单 session（P0 简化）：第二个 `begin` 撞 active 直接报错
- 向后兼容：保留原 one-shot 路径（无 `event` 字段则走老路）

**Step 2 — 测量 spike（关键 gate）**
- 实现 no-prefix 版 chunk-and-confirm（每 hop 都从零 decode）
- 用真实音频跑，量每 hop 时延
- **Gate**：hop 处理时间 ≤ hop 间隔。否则要调 `max_decode_tokens_per_hop` 或 `chunk_size_sec`
- 此 spike 是可丢弃的研究品，不是产品代码

**Step 3 — Prefix prompt + 完整 prompt 构造**
- 完整复制后端的 chat template 构造（Qwen3-ASR 用 chat-like prompt + audio 占位）
- Prefix 回退算法 + UTF-8 守护循环
- 用 backend 提供的 tokenizer encode/decode

**Step 4 — `max_input_len` 守护 + 自动分段 + session 清理**
- 见 §4.2 自动分段策略
- Session 错误时同时清理临时文件 + session table 项
- 空闲超时（默认 30s）强制 endSession

**Step 5 — 验收测试**
- 一次性回归测试：现有 one-shot 字节相等
- 流式 happy path：LCS-similarity ≥ 0.95 vs one-shot 基线
- end-of-speech 延迟：5 次中位 ≤ 500ms、p95 ≤ 1000ms
- 自动分段：长音频（>6s）能产生单一 final 事件
- 错误路径：恶意输入正确报错、session 正确释放

### 4.2 自动分段策略（关键工程细节）

**`max_input_len` 是真上限，不是 KV cap**。每 hop 调一次 one-shot，是单次 prefill，限制就是 `max_input_len`。

| 引擎 `max_input_len` | 实际 utterance 上限 |
|---|---|
| 128（默认）| ≈ 5.5-6s（128 - 30 prompt - 2 audio special - 15 prefix ≈ 81 / 13 audio tok/s ≈ 6.2s）|
| 256 | ≈ 15s |
| 512 | ≈ 33s |

**当 utterance 超过单 session 上限**：自动分段（transparent auto-segmentation）：

```
buffered audio 接近上限时:
    1. 跑当前 buffer 的 final hop → segment_text
    2. full_text += segment_text
    3. audio_accum 截到末尾 carryover_sec（默认 0.8s，mel block 对齐）
    4. 重置 chunk_id = 0, raw_decoded = ""
    5. 继续

last=true 时:
    final segment hop → segment_text
    full_text += segment_text
    emit final(full_text)
```

**0.8s carry-over 的作用**：
- 让新 segment 有足够上下文做语言检测（Qwen3-ASR 用早期音频判语言）
- 避免切到单词中间导致下个 segment 第一个字识别错

**P0 简化**：客户端看不到 segment 边界，只收到一个 final。如果产品需要 segment-level 信息，post-P0 加 `{"event":"segment_final"}` 中间事件即可。

### 4.3 可调参数

| 参数 | 默认（Orin NX 0.6B） | 调参方向 |
|---|---|---|
| `chunk_size_sec` | 0.5 | 更小 → 更频繁 partial、更多算力；更大 → 反之 |
| `unfixed_chunk_num` | 2 | 更大 → 早期不固化错误、算力多；更小 → 反之 |
| `unfixed_token_num` | 5 | 更大 → 末尾可改空间大、算力多；更小 → 反之 |
| `max_decode_tokens_per_hop` | 64 | 防止单次 hop 跑到 max_tokens 阻塞下一 hop |
| `auto_segment_cap_sec` | 5.5 | 留 ~10% safety margin 下面 `max_input_len` 上限 |
| `carryover_sec` | 0.8 | segment 间上下文携带量 |

## 5. 后端移植 checklist（哪些是通用、哪些是 backend-specific）

### 5.1 通用部分（任何后端都要做）

- ✅ 事件协议（begin / chunk / end）
- ✅ Session state 管理
- ✅ 累积 audio buffer
- ✅ 每 hop 调 backend 的 one-shot ASR API
- ✅ Prefix prompt 回退算法（包括 UTF-8 守护）
- ✅ Chat template 复制（用 backend 的 tokenizer.apply_chat_template）
- ✅ 自动分段
- ✅ Session 清理 + 超时

### 5.2 Backend-specific 部分

| 后端 | 一次性 ASR 调用 | Tokenizer 访问 | Chat template | 注意 |
|---|---|---|---|---|
| **EdgeLLM (TensorRT-LLM fork)** | `runtime->handleRequest(req)` | `runtime->getTokenizerForTesting()` | `tokenizer->applyChatTemplate()` | C++ 接口；prefix 注入靠 `applyChatTemplate=false` + raw `formattedRequests` |
| **vLLM** | `engine.generate([{prompt, multi_modal_data:{audio:[...]}}])` | `engine.tokenizer` | `tokenizer.apply_chat_template()` | 官方实现就是这个 |
| **ONNX Runtime** | 自己拼 encoder + thinker session.run | 单独跑 HF tokenizer | HF AutoTokenizer | 状态完全自管 |
| **TensorRT-LLM (upstream)** | `executor.enqueue_request(req)` | tokenizer 从 ModelConfig 读 | 同上 | 类似 EdgeLLM |
| **llama.cpp** | 不直接支持 Qwen3-ASR 多模态 | gguf 自带 tokenizer | apply_chat_template (新 API) | 需要先有 multimodal 支持 |

### 5.3 后端选择提示

- **想最小工程量、能接受 GPU 部署**：vLLM（官方就是这个，直接抄）
- **边缘部署、需要 ARM/Jetson**：EdgeLLM 或 ONNX Runtime
- **极致延迟、能写 CUDA**：TensorRT-LLM + 自己改 plugin
- **手机/CPU**：等 llama.cpp 支持 Qwen3-ASR 多模态（截至 2026-05 尚未）

### 5.4 backend 必须暴露的 API

最小集：
- `transcribe(prompt: str, audio: PCM) -> text`
- `tokenizer.encode(s) -> List[int]`
- `tokenizer.decode(ids) -> str`
- `tokenizer.apply_chat_template(messages, ...) -> str`

如果 backend 暴露了 metrics（encoder time / prefill time / decode time），把 step 2 spike 改成读 metrics 而不是只读 wall-clock，可以更精准定位瓶颈。

## 6. 实测数据汇总（NVIDIA Orin NX 16GB + 0.6B 模型）

### 6.1 Encoder 延迟（Spike A 实测）

| 输入大小 | 输出 token | 延迟（中位）|
|---|---|---|
| 1 块（1s） | 13 | 7.7 ms |
| 5 块（5s） | 65 | 10.1 ms |
| 8 块（8s）| 104 | ~12 ms |
| 30 块（30s）| 390 | 32.1 ms |
| 60 块（60s）| 780 | 65.1 ms |

延迟约 ~1ms/block + 7ms 固定开销。Encoder 几乎不是流式瓶颈。

### 6.2 Thinker 延迟（仅有 spike 估计，需 step 2 实测）

| 阶段 | 估计 |
|---|---|
| Prefill 1s audio embeddings（13 token） | ~30-50 ms |
| Prefill prefix text 5 token | ~5-10 ms |
| Decode per token（greedy） | ~30 ms |

5 token tail decode ≈ 150 ms。这是 hop 时延的主要构成。

### 6.3 LCS 实测（why naive chunking fails）

见 §2.1 表格。chunked 1-block: LCS 0.368；chunked 4-block: LCS 0.737；chunked 8-block: LCS 1.0。

## 7. 已知坑 / 踩雷记录

### 7.1 conv KV cache 的弯路

> 自己想出来的"P1 conv KV cache"看起来很对，做了 PyTorch + ONNX 导出 + bit-exact POC，**全是错的**。

原因：production encoder 是 per-100-mel-block 独立 conv（每块两边零填充），不是 contiguous conv。conv cache 解决的是不存在的问题。**先读 production graph 的 I/O 形状**再决定 cache 设计。

避坑：任何 cache 设计前，先 dump 推理图：
```python
import onnx
model = onnx.load("audio_encoder.onnx")
for i in model.graph.input: print(i.name, i.type.tensor_type.shape)
```

### 7.2 attention KV cache 的弯路

天真以为加 attention KV cache 就能流式。**实际上首 chunk 没未来，末 chunk 没未来，cache 救不了**。如果文本质量是 1-LCS-similarity，加 cache 只能从 0.4 提到 ~0.7 左右，仍不可用。

### 7.3 `max_kv_cache_capacity` vs `max_input_len` 混淆

- `max_kv_cache_capacity`：增量 prefill 路径关心（每次 prefill 后 KV 累积）
- `max_input_len`：one-shot 单次 prefill 关心

**chunk-and-confirm 走 one-shot 每 hop，限制是 `max_input_len`，不是 KV cap**。容易把两者搞混。

### 7.4 mel 块边界对齐

Qwen3-ASR encoder 接受 `[num_chunks, 128, 100]` 形状的 mel。每 chunk 是 100 mel timesteps = 1 秒（10ms hop）。

**音频长度必须是 1 秒整数倍**（除最后一块可短，由 audio runner 自动 padding）。客户端送 500ms 切，worker 内部要累积到 1s 块再喂。

### 7.5 UTF-8 字符边界

中文 token 经常是 3-byte UTF-8。`tokenizer.decode(tokens[:-K])` 可能切在字符中间，输出含 `�`（U+FFFD 替换字符）。官方 retry 循环逐个增大 K 直到合法。**移植时不要省掉这个 retry**。

## 8. 路径图（演化路线）

```
P0: chunk-and-confirm + prefix prompt
    ↓ 已知短板：首词延迟 ~800ms（不是 350ms）
    ↓
P1 (post-P0, 性能优化):
    - Thinker 改造支持「真增量 prefill」（M1/M2/M3.5 infrastructure 还在分支上）
    - 每 hop 不重 encode 整段，只 encode 新音频
    - 用 attention KV cache 跨 hop 复用
    - 仍是 chunk-and-confirm 框架，只是更省 compute
    ↓
P2 (long-term, 真低延迟):
    - LoRA 或全量 fine-tune encoder 做 chunk-causal attention
    - 推理时切小 chunk 也不掉点
    - 数周 + 数据
```

## 9. 参考实现

- **官方 vLLM 流式**（最权威，直接抄）：
  - `Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:584-829`（核心 3 个函数）
  - `Qwen3-ASR/examples/example_qwen3_asr_vllm_streaming.py`（使用例）
- **EdgeLLM 实现**（本项目）：
  - `qwen3-edgellm-jetson:native/edgellm_voice_worker/qwen3_asr_worker.cpp`
  - `qwen3-edgellm-jetson:docs/plans/qwen3-asr-streaming-design-2026-05-13.md`（详细设计 + 死路归档）

## 10. 验收 checklist

移植完后检查：

- [ ] 一次性路径未回归（字节相等于改造前的 one-shot 输出）
- [ ] 流式 happy path：LCS-similarity ≥ 0.95 vs one-shot 基线（同一段 audio）
- [ ] End-of-speech 延迟（last=true → final emit）：中位 ≤ 500ms、p95 ≤ 1000ms
- [ ] 自动分段：6s 以上 utterance 单 final 事件出，文本质量 LCS ≥ 0.90 vs 完整 one-shot
- [ ] 错误路径：恶意 JSON、unknown event、过长 chunk → 正确报错 + session 释放
- [ ] 多 session 串行：begin → end → begin → end 第二轮和第一轮等价
- [ ] 空闲超时：30s 无 chunk → 自动 end + emit timeout
- [ ] 中文 / 英文 / 混合：UTF-8 retry 不出错
- [ ] forced_language 参数：与 one-shot 同样形状的 prompt 注入

---

最后修改：2026-05-13
对应的本项目实战详细设计文档：`docs/plans/qwen3-asr-streaming-design-2026-05-13.md`
