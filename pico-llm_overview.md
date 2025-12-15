# pico-llm.py 概览

## 文件结构与作用（大白话）
- 参数解析：`parse_args` 收集命令行开关，控制数据源、序列长度、模型类型、训练超参、位置编码、设备、采样/日志等。
- 数据整理：`MixedSequenceDataset`/`seq_collate_fn`/`create_sequence_loader` 混合 TinyStories 和自定义文本，按概率采样，padding 后得到形如 `(seq_len, batch)` 的 token 序列；`split_sequence_pool` 负责 train/val/test 切分。
- 训练辅助：`compute_next_token_loss` 做自回归下一 token 的交叉熵；`save_model_checkpoint` 存权重+配置。
- 模型族1—k-gram MLP：`KGramMLPSeqModel` 看前 k 个 token 的 one-hot 拼接，经多层感知机直接出 logits，`chunk_size` 分块降内存。
- 模型族2—LSTM：`LSTMSeqModel` 是 embedding + LSTM + 线性层的自回归解码器。
- 模型族3—Transformer：`RMSNorm`/`RotaryEmbedding`/`MultiHeadSelfAttention`/`FeedForward`/`TransformerBlock`/`TransformerModel` 组成轻量因果 Transformer，支持 learned/sinusoidal/RoPE/none 位置编码，用上三角 mask 限制看未来，RoPE 预计算 cos/sin。
- 生成与多样性：`nucleus_sampling` (top-p)、`generate_text` 统一自回归生成入口（贪心或 top-p，可接入 monosemantic 标注）、`compute_text_diversity` 粗略统计重复率/类型比例。
- 训练主循环：`train_one_model` 跑 epoch，打印 loss，定时生成三种样例（greedy、top-p=0.95、top-p=1.0），可选记录过拟合日志；`evaluate_model` 做验证/测试 loss。
- 主入口：`main` 读取参数、设随机种子、加载并编码数据、构建数据拆分/loader，实例化所选模型，逐个训练并生成样例，保存 checkpoint，若启用则写过拟合报告，最后打印问候。

## 从输入到输出的链路
- 参数 → 配置：命令行经 `parse_args` 解析，确定数据源、超参、模型、位置编码、设备、采样/日志等。
- 文本 → token 序列：加载 TinyStories/自定义文件，用 `tiktoken` GPT-2 编码成 token id 并截断到 `block_size`；`split_sequence_pool` 切 train/val/test；`MixedSequenceDataset` 按概率混样本，`seq_collate_fn` padding 得到 `(seq_len, batch)`。
- 批数据 → 模型前向：DataLoader 提供 batch，送入所选模型——k-gram MLP（前 k token one-hot→MLP）、LSTM（embedding→LSTM→线性）、Transformer（token+位置编码，多层注意力+FFN，因果 mask）。
- logits → 训练信号：`compute_next_token_loss` 对齐下一 token 算交叉熵，Adam 反传更新；按步打印 loss，按时间跑 `generate_text` 三档采样做质检。
- 评估/日志：`evaluate_model` 算 val/test 平均 loss；若开过拟合日志，记录每 epoch 的 loss、生成文本、多样性指标。
- 输入 prompt → 文本输出：`generate_text` 自回归采样（greedy 或 top-p），循环前向取最后一步 logits 采样新 token，直到 `max_new_tokens`，再解码文本。
- 结果落地：打印最终三种采样文本，`save_model_checkpoint` 存权重+配置到 `checkpoints/<model>_final.pt`；若提供 `overfit_report_path`，写出 JSON 报告。
