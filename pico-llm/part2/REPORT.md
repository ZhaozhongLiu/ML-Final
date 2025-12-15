# pico-llm part2 报告（SFT + DPO：恐怖故事偏好）

本报告解释两件事：
1) **我改动之前，这个项目是什么状态**（你原本的 repo 能做什么）。  
2) **我新增了哪些 part2 内容**（SFT 数据、DPO 数据、训练与评测脚本），以及每一步的目的、核心代码在做什么。

> 注：本报告只覆盖 **SFT / DPO / Metrics** 流水线；不包含任何 interpretability / 可解释性相关内容。

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

1) **数据生成**：先提供一个不依赖外部大模型的模板式生成器（保证离线可跑）。  
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

目的：
- 先生成可用的训练数据（你后续可以替换为“用大模型生成”的数据）。
- 数据格式对齐你参考图里常见 DPO 格式：支持 `input/chosen/rejected` 字段（同时兼容 `prompt`）。

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

### 常用环境变量

```bash
# 指定设备（云端一般有 cuda:0；本地没 GPU 可以用 cpu）
DEVICE=cuda:0

# 跳过预训练，直接使用你已有的 base ckpt
BASE_CKPT_OVERRIDE=/path/to/transformer_final.pt

# 调整训练轮数
SFT_EPOCHS=1
DPO_EPOCHS=1
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

3) 如果你已经有一个训练好的 `transformer_final.pt`：
```bash
DEVICE=cuda:0 BASE_CKPT_OVERRIDE=/your/transformer_final.pt bash pico-llm/part2/run_all.sh
```

---

## 7. 后续你最可能要改的地方（把“模板生成”换成“用大模型生成”）

你说 SFT 的 prompt-answer 数据集要由大模型生成；DPO 的 preference 也要由偏好规则生成。  
你只需要改一个位置：`pico-llm/part2/story_generators.py`：
- 把 `write_horror_story()` / `write_wholesome_story()` 的实现换成 “调用你的大模型 / API / 本地 LLM” 产出文本即可
- 输出 JSONL 的字段不变（`input/prompt/response` 与 `input/chosen/rejected`），训练脚本就能直接复用

