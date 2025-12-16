# Part2 流程图（SFT → DPO → Metrics）

下面的流程图对应 `pico-llm/part2/run_all.sh` 的实际执行顺序与关键分支（API fallback、HF mirror、缺数据跳过等）。

## Mermaid

```mermaid
flowchart TD
  A([开始: bash pico-llm/part2/run_all.sh])

  A --> B[创建 runs/<RUN_TAG>/ 目录并开始记录 output.log]
  B --> C[pip install -r pico-llm/part2/requirements.txt]

  C --> D{NETWORK_PROBE=1?}
  D -- 是 --> D1[python -m part2.check_network<br/>-> network_probe.json]
  D -- 否 --> E
  D1 --> E[生成数据集: python -m part2.make_datasets]

  E --> P{DATA_PROVIDER}
  P -- deepseek --> P1{DEEPSEEK_API_KEY 存在且 check_openai 通过?}
  P1 -- 否 --> PT[回退: provider=template]
  P1 -- 是 --> E0[使用 DeepSeek 生成(批量/预算/失败回退)]
  P -- chatgpt --> P2{OPENAI_API_KEY 存在且 check_openai 通过?}
  P2 -- 否 --> PT
  P2 -- 是 --> E1[使用 OpenAI 生成(批量/预算/失败回退)]
  P -- template --> PT[使用模板生成(离线)]
  E0 --> E2[写出 sft_*.jsonl / dpo_*.jsonl + dataset_meta.json]
  E1 --> E2
  PT --> E2
  E2 --> E3{make_datasets 失败?}
  E3 -- 是 --> E4[WARN: 继续后续流程(可能缺 JSONL)]
  E3 -- 否 --> F
  E4 --> F[预训练 base checkpoint]

  F --> G{BASE_CKPT_OVERRIDE 设置?}
  G -- 是 --> G1[复制到 checkpoints/transformer_final.pt]
  G -- 否 --> H[尝试 TinyStories 预训练: python pico-llm/pico-llm.py]
  H --> H1{huggingface.co 可达?}
  H1 -- 否 --> H2[设置 HF_ENDPOINT_TINY=hf-mirror 并重试]
  H1 -- 是 --> H3[用官方/指定 endpoint 训练]
  H2 --> H3
  H3 --> H4{预训练成功?}
  H4 -- 否 --> X([ERROR: 退出 run_all.sh])
  H4 -- 是 --> I[得到 checkpoints/transformer_final.pt]
  G1 --> I

  I --> J{存在 sft_train.jsonl 与 sft_val.jsonl?}
  J -- 是 --> J1[SFT: python -m part2.train_sft<br/>-> transformer_sft.pt + logs_sft.jsonl]
  J -- 否 --> J2[跳过 SFT: 复制 base -> transformer_sft.pt]

  J1 --> K{存在 dpo_train.jsonl 与 dpo_val.jsonl?}
  J2 --> K
  K -- 是 --> K1[DPO: python -m part2.train_dpo<br/>-> transformer_dpo.pt + logs_dpo.jsonl]
  K -- 否 --> K2[跳过 DPO: 复制 sft -> transformer_dpo.pt]

  K1 --> L[评测: python -m part2.evaluate<br/>-> metrics/metrics.json]
  K2 --> L

  L --> M{logs_sft.jsonl 和 logs_dpo.jsonl 都存在?}
  M -- 是 --> M1[画曲线: python -m part2.plot_curves<br/>-> plots/curves.png]
  M -- 否 --> M2[跳过画图]

  M1 --> N([结束: runs/<RUN_TAG>/ 写出 checkpoints/metrics/plots/output.log])
  M2 --> N

  N --> Z[可选: 打包 poster/report<br/>python -m part2.make_part2_bundle --run_tag <RUN_TAG><br/>-> part2_results/<RUN_TAG>/]
```

## Diagram（纯文本）

```diagram
run_all.sh
  |
  +--> 创建 runs/<RUN_TAG>/ + output.log
  |
  +--> pip install requirements.txt
  |
  +--> (可选) NETWORK_PROBE=1 -> check_network -> network_probe.json
  |
  +--> 数据集生成 make_datasets
  |      |
  |      +--> provider=deepseek/chatgpt -> 先 check_openai
  |      |        |
  |      |        +--> 失败/没 API KEY -> fallback=template
  |      |
  |      +--> 输出: sft_{train,val,test}.jsonl, dpo_{train,val,test}.jsonl, dataset_meta.json
  |      +--> (失败也继续) 可能导致后续 SFT/DPO 跳过
  |
  +--> 预训练 base checkpoint
  |      |
  |      +--> BASE_CKPT_OVERRIDE? 是 -> 复制 -> transformer_final.pt
  |      |
  |      +--> 否 -> 用 TinyStories 子集跑 pico-llm.py
  |             |
  |             +--> huggingface.co 不可达 -> 用 HF mirror 重试
  |             +--> 仍失败 -> 直接退出
  |
  +--> SFT (如果 sft_train/val 存在)
  |      |
  |      +--> train_sft -> transformer_sft.pt + logs_sft.jsonl
  |      +--> 否则: base -> transformer_sft.pt
  |
  +--> DPO (如果 dpo_train/val 存在)
  |      |
  |      +--> train_dpo -> transformer_dpo.pt + logs_dpo.jsonl
  |      +--> 否则: sft -> transformer_dpo.pt
  |
  +--> evaluate -> metrics/metrics.json
  |
  +--> plot_curves (如果两份 logs 都存在) -> plots/curves.png
  |
  +--> DONE: runs/<RUN_TAG>/
         - checkpoints/{transformer_final,transformer_sft,transformer_dpo}.pt
         - metrics/metrics.json
         - plots/curves.png (可选)
         - output.log

可选后处理:
  make_part2_bundle --run_tag <RUN_TAG> -> part2_results/<RUN_TAG>/
```
