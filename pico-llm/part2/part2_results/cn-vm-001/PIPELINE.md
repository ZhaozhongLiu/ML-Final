```mermaid
flowchart TD
  %% End-to-end: tokens -> LM -> (pretrain) -> SFT -> DPO -> eval/play

  A[Raw text input] --> B[Tokenizer<br/>(tiktoken GPT-2)]
  B --> C[Input token IDs]

  C --> M{Model type}

  subgraph LM[Language Model Forward Pass]
    direction TB
    M --> T[Transformer LM]
    M --> L[LSTM LM]
    M --> K[k-gram MLP]
    T --> Z[Logits over vocab]
    L --> Z
    K --> Z
  end

  subgraph TRAIN[Training Stages (Part2)]
    direction TB

    P0[TinyStories corpus] --> P1[Pretrain base LM<br/>(next-token prediction)]
    P1 --> S0[Base checkpoint]

    D0[Story spec prompts] --> D1[SFT dataset<br/>(prompt, response)]
    S0 --> S1[SFT fine-tune<br/>(masked next-token loss)]
    D1 --> S1
    S1 --> S2[SFT checkpoint]

    D2[DPO dataset<br/>(prompt, chosen, rejected)] --> D3[DPO post-train<br/>(preference loss)]
    S2 --> D3
    D3 --> F[Final checkpoint]
  end

  subgraph USE[Use / Output]
    direction TB
    F --> E1[Evaluation<br/>(SFT test loss, DPO pref acc)]
    F --> E2[Plots<br/>(training curves, summaries)]
    F --> E3[Play / Inference<br/>(prompt → generated story)]
  end

```
