# pico-llm part2 报告（SFT + DPO：恐怖故事偏好）

本报告解释两件事：
1) **我改动之前，这个项目是什么状态**（你原本的 repo 能做什么）。  
2) **我新增了哪些 part2 内容**（SFT 数据、DPO 数据、训练与评测脚本），以及每一步的目的、核心代码在做什么。

> 注：本报告只覆盖 **SFT / DPO / Metrics** 流水线；不包含任何 interpretability / 可解释性相关内容。

## 快速入口：把 run 结果整理成一份“可做 poster 的文件夹”

当 `pico-llm/part2/run_all.sh` 跑完后，你会在 `pico-llm/part2/runs/<RUN_TAG>/` 看到 checkpoints、曲线图、metrics 等文件。  
为了方便做 final project poster，我新增了一个“打包脚本”，会把关键图表 + 指标 + 示例生成整理到：

- `pico-llm/part2/part2_results/<RUN_TAG>/`

另外，最新的 `run_all.sh` 默认会在跑完后**自动执行打包**（可用 `BUNDLE_AFTER_RUN=0` 关闭）。

使用方法（从 repo root）：

```bash
PYTHONPATH=pico-llm python3 -m part2.make_part2_bundle --run_tag <RUN_TAG>
```

整理包里推荐先看：
- `pico-llm/part2/part2_results/<RUN_TAG>/INDEX_ZH_EN.md`
- `pico-llm/part2/part2_results/<RUN_TAG>/REPORT_ZH_EN.md`

---

## 1. 改动前：项目原始状态（你拿到时能做什么）

### 1.1 原始 pico-llm（故事 LM：续写）

主脚本是 `pico-llm/pico-llm.py`，它本质上做的是：
- **训练目标**：next-token prediction（给定前缀，预测下一个 token），loss 是交叉熵（cross entropy）。
- **数据来源**：
  - TinyStories（HuggingFace 数据集）+ 你自己的 `--input_files`（每行一个样本）
- **生成方式**：
  - 用 `--prompt` 作为开头（例如 “Once upon a …”），调用 `generate_text()` 自回归生成后续 token
  - 支持 greedy 或 nucleus/top-p 采样

所以你当时理解的“原任务 = 给一个故事开头继续写，用生成质量做测试”是对的：它是 **语言模型续写**。

### 1.2 你 repo 里还有一条“数字序列”研究线

你 repo 里有 `pico-llm/README.md` 描述的 **numeric sequences: baseline vs linear attention** 管线（`train/`, `models/`, `data/`），这是另一条线。  
本次 part2（SFT/DPO）我做的是 **围绕 TinyStories + 文本生成** 这一条，不是数字序列那条。

---

## 2. 新增目标：你提出的 pico-llm part2（SFT + DPO）

你希望最终实现：

1) **SFT（监督微调）**  
   不再是“给故事开头续写”，而是“给一个**故事规格/设定（story specification）** → 写完整故事”。  
   目标：让模型更像“指令跟随 + 写恐怖故事”的助手。

2) **DPO（偏好优化 / post-training）**  
   做一个 preference dataset：对同一个 prompt，有 `chosen`（更恐怖/更符合你要的坏倾向）与 `rejected`（更美好发展/偏离目标）。  
   目标：让模型输出分布更倾向 chosen，且不需要 reward model 或 PPO。

3) **Metrics 测试**  
   在 held-out test set 上做：
   - SFT：masked next-token loss（只算答案部分，不算 prompt 部分）
   - DPO：preference accuracy（模型给 chosen 的 logprob 是否大于 rejected）
   - 额外：抽样生成文本做简单 horror-lexicon 统计（启发式，不是严格指标）

这些设计与 references 中的 post-training 思路（SFT + DPO）一致。

---

## 3. 我新增内容的“计划”（我按什么顺序做）

为保证你把项目放云端 GPU 后能“一键跑”，我采用如下顺序：

1) **数据生成**：先提供一个不依赖外部大模型的模板式生成器（保证离线可跑），再提供可选的 **ChatGPT API** 生成器（提高数据质量）。  
2) **预训练 base checkpoint**：用你原有 `pico-llm.py` 先训练出一个 `transformer_final.pt`（或者允许你直接提供现成 base ckpt）。  
3) **SFT**：用 prompt-spec → story 数据做监督微调（只对 response 计算 loss）。  
4) **DPO**：用 preference pairs 做 DPO（支持 label smoothing / IPO 选项）。  
5) **评测 + 画曲线**：输出 JSON 指标，并画出类似你资料截图的曲线。

---

## 4. 我具体做了什么（按步骤）+ 每一步的目的

新增目录：`pico-llm/part2/`  
它是一个可通过 `PYTHONPATH=pico-llm` 导入运行的小包（`part2.*`）。

下面按“你运行一键脚本时发生什么”来解释。

---

### Step A：生成 SFT/DPO 训练集（JSONL）

文件：
- `pico-llm/part2/make_datasets.py`
- `pico-llm/part2/story_generators.py`
- `pico-llm/part2/chatgpt_api.py`（新增：可选）
- `pico-llm/part2/check_openai.py`（新增：API 检测）

目的：
- 先生成可用的训练数据（你后续可以替换为“用大模型生成”的数据）。
- 数据格式对齐你参考图里常见 DPO 格式：支持 `input/chosen/rejected` 字段（同时兼容 `prompt`）。
- **新增**：支持 `--provider chatgpt` 用 ChatGPT API 生成更高质量的 SFT/DPO 数据；并提供“API 能否跑通”的检测与中途失败的降级策略，保证一键脚本不会卡死。

#### A1. 生成“故事规格 prompt”

核心函数（简化解释）在 `pico-llm/part2/story_generators.py`：

```py
def format_prompt(spec: StorySpec) -> str:
    return (
        "Write a short horror story that follows this story specification.\n\n"
        f"Title: {spec.title}\n"
        f"Setting: {spec.setting}\n"
        ...
        "Story:\n"
    )
```

你可以把它理解为：把 `标题/地点/人物/禁忌/反转` 拼成一个“写作任务书”。

#### A2. SFT 数据：prompt + response

SFT 行结构（每行一个 JSON）：

```json
{
  "id": "sft-000001",
  "prompt": "...",
  "input": "...",
  "response": "..."
}
```

- `prompt` / `input`：同一个内容（为了兼容不同教程/代码习惯）
- `response`：模板生成的恐怖故事（后续你可以换成外部大模型生成）

#### A3. DPO 数据：prompt + chosen + rejected

DPO 行结构：

```json
{
  "id": "dpo-000001",
  "prompt": "...",
  "input": "...",
  "chosen": "...(更恐怖)...",
  "rejected": "...(更美好/偏离目标)..."
}
```

---

### Step A0（新增）：改为可选用大模型 API 生成数据（默认 DeepSeek）

我没有删除模板生成器，而是做成 **三种 provider**：
- `template`：完全离线，随时可跑
- `deepseek`：调用 DeepSeek（OpenAI-compatible）API 生成数据（默认，需要 `DEEPSEEK_API_KEY`）
- `chatgpt`：调用 OpenAI ChatGPT API 生成数据（可选，需要 `OPENAI_API_KEY`）

#### A0.1 API Key 放哪里？

为安全起见，API key **不写死在代码里**，而是通过环境变量传入：

```bash
export DEEPSEEK_API_KEY="..."
bash pico-llm/part2/run_all.sh
```

这样你把项目传到云端时，只需要在环境里设置 key，不会把 key 提交到 git 或写进日志。

#### A0.2 默认模型（最便宜且适合本任务）

默认使用 DeepSeek：
- `DATA_PROVIDER=deepseek`
- `DEEPSEEK_MODEL=deepseek-chat`
- `DEEPSEEK_BASE_URL=https://api.deepseek.com`

OpenAI（可选）默认：
- `DATA_PROVIDER=chatgpt`
- `OPENAI_MODEL=gpt-4o-mini`

#### A0.3 先检测 API 能不能用（health check）

为了避免一键跑到一半才发现 key/网络有问题：
- `run_all.sh` 在 `DATA_PROVIDER=deepseek` 或 `DATA_PROVIDER=chatgpt` 时会先运行 `part2.check_openai` 做一次极小请求验证  
- 若检测失败，**自动 fallback** 到 `template`，不中断后面的训练/评测流程

检测脚本在：
- `pico-llm/part2/check_openai.py`

它调用了 `ChatGPTClient.ping()`（见 `pico-llm/part2/chatgpt_api.py`）：

```py
def ping(self) -> None:
    raw = self._chat(... want_json=True)
    obj = _extract_json(raw)
    if obj.get("ok") is not True:
        raise ChatGPTAPIError(...)
```

#### A0.4 API 中途“不给东西/报错”怎么办？

你要求“如果 api 不给东西了，就当作训练部分完成，继续别的操作”。我做了两层保护：

1) **数据生成阶段的逐样本降级**（在 `make_datasets.py`）  
   - 当 `provider=deepseek` 或 `provider=chatgpt` 时，每条数据单独请求 API（可选开启 batch，一次请求多条）  
   - 若某条失败：
     - 默认 `LLM_FALLBACK=template`：用模板生成补齐这一条（保证数据量完整）
     - 或者 `LLM_FALLBACK=stop`：直接停止生成，保留已生成的部分

对应逻辑（简化）：

```py
try:
    response = client.generate_sft_story(prompt)
except Exception:
    if openai_fallback == "template":
        response = write_horror_story(spec, rng)
    else:  # stop
        break
```

2) **一键脚本层面的不中断**（在 `run_all.sh`）  
   - 即使数据生成命令返回非 0，脚本也会打印 warning 然后继续  
   - 并且在进入 SFT/DPO 前会检查 `*.jsonl` 是否为空；为空就跳过该阶段（避免 crash）

---

### Step A0.5（新增）：预计会用多少 API？我设了哪些阈值？

#### 预计 API 调用次数（默认 `run_all.sh` 的数据规模）

默认数据规模（见 `run_all.sh`）：
- SFT：`256 + 64 + 64 = 384` 条
- DPO：`256 + 64 + 64 = 384` 条（每条包含 chosen+rejected）

当 `DATA_PROVIDER` 为 `deepseek` 或 `chatgpt` 时：
- SFT：每条 1 次 API 调用 → **384 calls**
- DPO：每条 pair 1 次 API 调用 → **384 calls**
- 合计 **768 calls**

如果你开启批量生成（`LLM_BATCH_SIZE>1`），calls 会近似按比例下降（例如 batch=4 时 calls 约减少到 1/4，外加少量失败重试）。

#### 预计 token 用量（粗略上限）

每次调用的输出 token 上限由：
- `LLM_MAX_OUTPUT_TOKENS` 控制（`run_all.sh` 默认 1200）

因此输出 token 的粗略上界是：

```
max_output_tokens_total ≈ calls_total * LLM_MAX_OUTPUT_TOKENS
                        ≈ 768 * 1200
                        ≈ 921,600 output tokens
```

另外还会有输入 token（prompt/spec 文本），其大小取决于 prompt 长度。

#### 我设的阈值/预算（防止失控花费）

你可以只改环境变量，不用改代码：

```bash
# 成本阈值（0 表示不限制）
LLM_MAX_CALLS=200
LLM_MAX_TOTAL_TOKENS=200000

# API 连续失败阈值
LLM_MAX_CONSEC_FAILS=3

# 失败策略（默认 template）：失败就用模板补齐；或 stop：直接停止生成保留部分数据
LLM_FALLBACK=template

# 批量生成：每次 API 调用生成 N 条（减少 calls）
LLM_BATCH_SIZE=4
```

实现位置：
- 预算统计：`pico-llm/part2/chatgpt_api.py` 会从 API `usage` 累计 `calls/total_tokens`
- 预算停止：`pico-llm/part2/make_datasets.py` 在每次请求前检查 `LLM_MAX_CALLS/LLM_MAX_TOTAL_TOKENS`
- API 可用性检测：`pico-llm/part2/check_openai.py`（`run_all.sh` 会先跑它）
- 真实用量会被写入：`runs/.../data/dataset_meta.json`（字段 `openai_usage`）

---

### Step B：把文本变成 token（并构造“只算答案”的 loss mask）

文件：
- `pico-llm/part2/tokenization.py`
- `pico-llm/part2/datasets.py`

目的：
- 你原始 `pico-llm.py` 是标准 LM：对整段 token 都算 next-token loss。  
  但 SFT 里我们通常希望：**prompt 部分不算 loss，只训练模型把 response 写出来**。

#### B1. encode_sft：拼接 prompt + response，并做 mask

核心逻辑在 `pico-llm/part2/tokenization.py`：

```py
def encode_sft(enc, prompt: str, response: str, max_tokens: int):
    prompt_ids = enc.encode(prompt)
    response_ids = enc.encode(response)

    input_ids = prompt_ids + response_ids
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return EncodedExample(input_ids=input_ids, loss_mask=loss_mask)
```

解释：
- `input_ids` 是模型要看到的完整输入（prompt + response）
- `loss_mask` 里：
  - prompt 的 token 标 0（不算）
  - response 的 token 标 1（要学习）

#### B2. dataset 兼容 `input` / `prompt`

在 `pico-llm/part2/datasets.py` 里我做了兼容：

```py
def _get_prompt(row):
    if "prompt" in row: return row["prompt"]
    if "input" in row: return row["input"]
```

这样你未来无论数据用 `input` 还是 `prompt` 字段，都能直接训练。

---

### Step C：SFT 训练（监督微调）

文件：
- `pico-llm/part2/train_sft.py`
- `pico-llm/part2/losses.py`

目的：
- 在 base transformer checkpoint 上继续训练，让模型学会“看规格写恐怖故事”。

#### C1. masked next-token loss：只对 response 计算

在 `pico-llm/part2/losses.py`：

```py
def masked_next_token_loss(logits, tokens, loss_mask):
    pred = logits[:-1]      # 预测 t+1
    gold = tokens[1:]       # 真值 t+1
    mask = loss_mask[1:]    # 对齐到 gold
    per_tok = cross_entropy(pred, gold, reduction="none")
    return (per_tok * mask).sum() / mask.sum()
```

解释（不需要懂矩阵也能理解）：
- 语言模型训练就是“每个位置预测下一个 token”
- 这里我们只把 response 那段 token 的损失累加起来，其余（prompt）丢掉

#### C2. SFT 训练时记录曲线（可选）

你资料里有 “SFT 的 chosen/rejected 概率曲线” 的图，我加了一个 **监控功能**：
- 在 SFT 训练过程中，拿一份 DPO 数据（比如 `dpo_val.jsonl`）
- 计算当前模型对 chosen/rejected 的 logprob 平均值并写入 JSONL

对应参数：
- `--monitor_dpo_jsonl path/to/dpo_val.jsonl`
- `--log_jsonl path/to/logs_sft.jsonl`

---

### Step D：DPO 训练（偏好优化）

文件：
- `pico-llm/part2/train_dpo.py`
- `pico-llm/part2/losses.py`

目的：
- 用成对偏好数据，让模型倾向输出 `chosen`，远离 `rejected`
- 不需要 reward model、也不需要 PPO

#### D1. 先把“整段 response”的 logprob 算出来

在 `pico-llm/part2/losses.py`：

```py
def sequence_logprobs(logits, tokens, loss_mask):
    logp = log_softmax(logits[:-1], dim=-1)
    gold = tokens[1:]
    per_pos = logp.gather(-1, gold[..., None]).squeeze(-1)
    return (per_pos * loss_mask[1:]).sum(dim=0)  # 每个样本一个总 logprob
```

解释：
- 这一步是“模型对这段答案有多相信”（数值越大越相信）
- 我们只对答案 token 求和（mask=1 的部分）

#### D2. DPO loss（支持 label smoothing / IPO）

我按你截图里的公式写了可选项（`pico-llm/part2/losses.py`）：

```py
pi_logratios  = policy_chosen - policy_rejected
ref_logratios = ref_chosen - ref_rejected
logits = pi_logratios - ref_logratios
loss = -logsigmoid(beta * logits)
```

并支持：
- `--label_smoothing 0.05`
- `--ipo`（用 IPO loss）

#### D3. 记录 DPO rewards 曲线

参考资料里 DPO 常画 reward 曲线，我加了日志：
- `chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)`
- `rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)`

输出到 `--log_jsonl logs_dpo.jsonl`

---

### Step E：评测（metrics）+ 抽样生成

文件：
- `pico-llm/part2/evaluate.py`

目的：
- 训练完后，输出可复现的评测 JSON

包含：
- `sft_test_loss`：SFT 测试集的 masked loss
- `dpo_test_pref_acc`：DPO 测试集 preference accuracy（chosen > rejected 的比例）
- `samples`：从测试 prompt 抽样生成一些文本 + 简单 horror lexicon 命中率

---

### Step F：画曲线（生成结构图里那两张图的简化版）

文件：
- `pico-llm/part2/plot_curves.py`

目的：
- 把 `logs_sft.jsonl` 和 `logs_dpo.jsonl` 画成一个 `curves.png`
- 左图：SFT monitor（chosen vs rejected 平均 logprob）
- 右图：DPO rewards（chosen vs rejected）

---

## 5. 一键运行脚本（云端可直接跑）

文件：`pico-llm/part2/run_all.sh`

它做的事（顺序）：
1) `pip install -r pico-llm/part2/requirements.txt`
2) 生成数据：`part2.make_datasets`
3) 预训练 base transformer：调用原脚本 `pico-llm/pico-llm.py`
4) SFT：`part2.train_sft`
5) DPO：`part2.train_dpo`
6) Evaluate：`part2.evaluate`
7) Plot：`part2.plot_curves`

### 网络受限（中国/部分云厂商）时 TinyStories 下载失败怎么办？

你在 VM 上的探测结果显示：
- `huggingface.co` 不可达
- `hf-mirror.com` 可达

因此 pretrain 阶段需要使用 HuggingFace 镜像。`run_all.sh` 已经支持：
- 显式指定镜像：
  ```bash
  HF_ENDPOINT_TINY=https://hf-mirror.com DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
  ```
- 或者让脚本自动判断：如果检测到 `huggingface.co` 不可达，会自动把 `HF_ENDPOINT_TINY` 设成 `HF_ENDPOINT_FALLBACK`（默认 `https://hf-mirror.com`）。

另外，`run_all.sh` 默认会写入一份网络探测 JSON：
- `pico-llm/part2/runs/<RUN_TAG>/network_probe.json`
你可以用 `NETWORK_PROBE=0` 关闭。

### 三个训练阶段分别训练到什么程度会停止？

这套 pipeline 目前没有 “val 不再下降就 early-stop” 之类的自动停止逻辑；它是 **按 epoch/step 或按墙钟时间上限停止**（便于你上云端“一键跑”可控、可复现）。

1) **Pretrain（base checkpoint）**  
   在 `run_all.sh` 里调用 `pico-llm/pico-llm.py`，停止条件由下面参数决定：
   - **墙钟时间上限**：`--max_train_seconds PRETRAIN_MAX_SECONDS`（默认 4 小时）
   - `--max_steps_per_epoch PRETRAIN_MAX_STEPS`（默认 2000）：每个 epoch 最多跑多少 step  
   - `--num_epochs` 在脚本里设为很大（例如 9999），主要由时间上限触发停止
   满足时间上限或 epoch/step 条件就结束该阶段。

2) **SFT（监督微调）**  
   在 `run_all.sh` 里调用 `part2.train_sft`，停止条件：
   - **墙钟时间上限**：`SFT_MAX_SECONDS`（默认 30 分钟）
   - `SFT_EPOCHS`（默认 9999）：主要由时间上限触发停止

3) **Post-train（DPO/IPO）**  
   在 `run_all.sh` 里调用 `part2.train_dpo`，停止条件：
   - **墙钟时间上限**：`DPO_MAX_SECONDS`（默认 30 分钟）
   - `DPO_EPOCHS`（默认 9999）：主要由时间上限触发停止

如果你要“更智能的停止条件”，可以后续加：例如每个 epoch 后评估 `val_loss`，连续 N 次不提升就停止（early stopping）。

### 常用环境变量

```bash
# 指定设备（云端一般有 cuda:0；本地没 GPU 可以用 cpu）
DEVICE=cuda:0

# 三阶段墙钟时间预算（默认总计约 5 小时）
PRETRAIN_MAX_SECONDS=14400
SFT_MAX_SECONDS=1800
DPO_MAX_SECONDS=1800

# 跳过预训练，直接使用你已有的 base ckpt
BASE_CKPT_OVERRIDE=/path/to/transformer_final.pt

# 训练轮数（默认设置很大，主要由 max seconds 触发停止）
SFT_EPOCHS=9999
DPO_EPOCHS=9999

# 选择数据生成方式：deepseek（默认）| chatgpt | template
DATA_PROVIDER=deepseek

# DeepSeek（默认，OpenAI-compatible）
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 生成参数（deepseek/chatgpt 共用）
LLM_TEMPERATURE=0.8
LLM_MAX_OUTPUT_TOKENS=1200
LLM_MAX_RETRIES=5

# API 失败时策略：
# - template：单条失败就用模板补齐（默认，保证数据量）
# - stop：直接停止生成，保留部分数据并继续后续训练/评测
LLM_FALLBACK=template
LLM_MAX_CONSEC_FAILS=3

# 成本/用量阈值（建议设置其一；0 表示不限制）
# - LLM_MAX_CALLS：最多发多少次 API 请求（一次请求生成一条 SFT 或一条 DPO pair；batch>1 时一次生成多条）
# - LLM_MAX_TOTAL_TOKENS：API 返回的 usage.total_tokens 累计上限
LLM_MAX_CALLS=0
LLM_MAX_TOTAL_TOKENS=2500000

# 批量生成：每次 API 调用生成 N 条（减少 calls，默认 4）
LLM_BATCH_SIZE=4

# OpenAI ChatGPT（可选）
OPENAI_MODEL=gpt-4o-mini
```

---

## 6. 你现在应该怎么用（最简单）

1) 本地跑（无 GPU）：
```bash
DEVICE=cpu bash pico-llm/part2/run_all.sh
```

2) 云端跑（有 GPU）：
```bash
DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

> 默认 `DATA_PROVIDER=deepseek`：如果你设置了 `DEEPSEEK_API_KEY` 就会用 DeepSeek 生成训练数据；否则脚本会自动回落到 `template` 数据生成。

3) 如果你已经有一个训练好的 `transformer_final.pt`：
```bash
DEVICE=cuda:0 BASE_CKPT_OVERRIDE=/your/transformer_final.pt bash pico-llm/part2/run_all.sh
```

4) 用 DeepSeek API 生成数据（默认推荐）：
```bash
export DEEPSEEK_API_KEY="..."
DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

5) 如果你希望 API 一旦失败就直接停（保留部分数据继续训练）：
```bash
export DEEPSEEK_API_KEY="..."
LLM_FALLBACK=stop DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

6) 在中国VM上这样跑最稳:
```
export DEEPSEEK_API_KEY="..."
HF_ENDPOINT_TINY=https://hf-mirror.com DEVICE=cuda:0 RUN_TAG=cn-vm-001 bash pico-llm/part2/run_detached.sh
```

---

## 6.1 远程运行：SSH 断线也不停（screen/tmux）+ 去哪里查看结果

当你看到：
```
[part2] started screen session: pico-part2-cn-vm-001
```
就表示已经在后台运行了。

### 实时查看输出（推荐）

进入运行会话：
```bash
screen -r pico-part2-cn-vm-001
```

退出但不停止（detach）：按 `Ctrl-a` 然后按 `d`

### 直接查看日志文件（不用进入 screen）

`run_all.sh` 会把 stdout+stderr 同时写入：
- `pico-llm/part2/runs/<RUN_TAG>/output.log`

例如：
```bash
tail -f pico-llm/part2/runs/cn-vm-001/output.log
```

### 查看结果目录（checkpoints / metrics / plots）

所有产物都在：
- `pico-llm/part2/runs/<RUN_TAG>/`

例如：
```bash
ls -la pico-llm/part2/runs/cn-vm-001/
```

---

## 6.2 现在怎么“玩一玩”你的 LLM（用 checkpoint 生成故事）

你的模型不是“聊天机器人”式的严格对话协议，它本质是一个 **自回归语言模型**：给它一段 prompt，它会继续往后生成。

下面给你两种最常用的玩法：一次性生成（one-shot）和交互式 REPL。

### 6.2.1 找到你要用的 checkpoint

如果你跑的是 `RUN_TAG=cn-vm-001`，通常用 DPO 后的 checkpoint：
- `pico-llm/part2/runs/cn-vm-001/checkpoints/transformer_dpo.pt`

你也可以试试：
- `pico-llm/part2/runs/cn-vm-001/checkpoints/transformer_sft.pt`
- `pico-llm/part2/runs/cn-vm-001/checkpoints/transformer_final.pt`

### 6.2.2 One-shot：给一个 story spec，让模型写故事

用新增脚本 `part2.play_model`：

```bash
PYTHONPATH=pico-llm python3 -m part2.play_model \
  --checkpoint pico-llm/part2/runs/cn-vm-001/checkpoints/transformer_dpo.pt \
  --device cuda:0 \
  --mode oneshot \
  --top_p 0.95 \
  --max_new_tokens 220 \
  --prompt "You are a creative writing assistant.\nWrite a short horror story that follows this story specification.\n\nTitle: The Door That Wasn't There\nSetting: an old apartment building during a winter blackout\nProtagonist: Mina\nSupporting character: Kai\nImportant object: a brass key\nTaboo rule: never answer knocks after 2 a.m.\nTwist: the sound is coming from inside the walls\n\nConstraints:\n- 2 to 4 paragraphs.\n- Keep it suspenseful and eerie, not graphic.\n- End with an unsettling implication.\n\nStory:\n"
```

参数怎么调：
- `--top_p <= 0`：贪心（更稳定但更重复）
- `--top_p 0.9~0.97`：更有创造力
- `--max_new_tokens`：生成长度

### 6.2.3 REPL：交互式反复试 prompt

```bash
PYTHONPATH=pico-llm python3 -m part2.play_model \
  --checkpoint pico-llm/part2/runs/cn-vm-001/checkpoints/transformer_dpo.pt \
  --device cuda:0 \
  --mode repl \
  --top_p 0.95 \
  --max_new_tokens 220
```

输入提示词后回车即可生成；输入 `/exit` 退出。

### 6.2.4 如果你没有 GPU（或想用 CPU）

把 `--device cpu` 即可：

```bash
PYTHONPATH=pico-llm python3 -m part2.play_model --checkpoint ... --device cpu
```

## 7. 后续你最可能要改的地方（把“模板生成”换成“用大模型生成”）

你说 SFT 的 prompt-answer 数据集要由大模型生成；DPO 的 preference 也要由偏好规则生成。  
现在你有两条路：

### 路线 A（推荐）：直接用 ChatGPT API provider

你不需要改代码，只需要：
- DeepSeek（默认）：设置 `DEEPSEEK_API_KEY`，`DATA_PROVIDER=deepseek`
- OpenAI（可选）：设置 `OPENAI_API_KEY`，`DATA_PROVIDER=chatgpt`

数据生成代码在：
- `pico-llm/part2/chatgpt_api.py`
- `pico-llm/part2/make_datasets.py`

### 路线 B：你要换成“你自己的大模型/不同 API”

你可以模仿 `ChatGPTClient` 的接口，写一个你自己的 client：
- 产出 SFT 的 `response`
- 产出 DPO 的 `{chosen, rejected}`

只要输出 JSONL 的字段不变（`input/prompt/response` 与 `input/chosen/rejected`），训练脚本就能直接复用。

---

## 8. 具体：SFT 和 DPO 如何用大模型（ChatGPT API）来“辅助训练”

这里的“用大模型辅助训练”指的是：**用更强的模型生成训练数据**（SFT 的 prompt→answer；DPO 的 chosen/rejected），再用你的 pico-llm transformer 去 finetune / post-train。

### 8.1 SFT：用大模型生成 prompt→answer（训练你的模型写恐怖故事）

数据目标：
- 输入（prompt/input）：故事规格（spec）
- 输出（response）：完整恐怖短故事

本项目已经把“调用 ChatGPT API 生成 SFT response”写好了，位置在：
- `pico-llm/part2/chatgpt_api.py`：`generate_sft_story(prompt)`

它的大致提示词结构（示意）：

```text
System: you write eerie, non-graphic horror stories in English...
User: given this specification, write 2-4 paragraphs, end unsettling, return story only...
```

生成出来会写入一行 JSONL：

```json
{"id":"sft-000123","input":"...spec...","prompt":"...spec...","response":"...story..."}
```

训练时（SFT）只对 `response` 计算 loss（避免模型去“复述 prompt”）：
- 见 `pico-llm/part2/tokenization.py` 的 `loss_mask`
- 见 `pico-llm/part2/losses.py` 的 `masked_next_token_loss`

### 8.2 DPO：用大模型生成 preference pair（chosen/rejected）

数据目标：
- 同一个 prompt/spec 下：
  - `chosen`：更恐怖、更阴森、结尾更不安（但不血腥）
  - `rejected`：更温馨/圆满/化解恐惧（你定义为 reject）

本项目已经把“调用 ChatGPT API 生成 DPO pair”写好了：
- `pico-llm/part2/chatgpt_api.py`：`generate_dpo_pair(prompt)`

它会要求模型返回严格 JSON（chosen/rejected），写入：

```json
{"id":"dpo-000123","input":"...spec...","prompt":"...spec...","chosen":"...","rejected":"..."}
```

DPO 训练时：
- policy 模型从 SFT checkpoint 起步（`transformer_sft.pt`）
- reference 模型冻结（默认就是同一个 checkpoint）
- 优化目标：让 policy 相对 reference 更偏向 chosen 而不是 rejected

核心对比来自 logprob 差（简化）：

```text
advantage = (log π(chosen) - log π(rejected)) - (log π_ref(chosen) - log π_ref(rejected))
loss = -log sigmoid(beta * advantage)
```

### 8.3 质量与成本控制（你远程跑时最关键）

1) **先做 API 检测**：`run_all.sh` 会先跑 `part2.check_openai`，失败自动回退模板数据，避免卡住。
2) **失败不停机**：
   - 默认 `LLM_FALLBACK=template`：某条样本生成失败就用模板补齐（保证数据量）
   - 或设 `LLM_FALLBACK=stop`：直接停止生成，保留部分数据继续训练
3) **预算阈值**：建议你远程必设其一：
   - `LLM_MAX_CALLS`
   - 或 `LLM_MAX_TOTAL_TOKENS`
4) **记录真实用量**：生成数据后看 `runs/.../data/dataset_meta.json` 的 `openai_usage`
