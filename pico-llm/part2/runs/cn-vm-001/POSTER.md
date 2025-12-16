# Part 2 Poster
## Supervised Fine-Tuning and Direct Preference Optimization for Horror-Story Alignment (Run: cn-vm-001)

![Pipeline overview](flowchart.png)

### Motivation
A small language model can learn to continue text, but that does not automatically mean it can follow an explicit writing specification or consistently produce a desired style. This project turns the task into instruction-following creative writing: the input is a structured story specification, and the output is a short story that follows the constraints. After learning the basic mapping from specification to story, we further align the model toward a target style using preference comparisons rather than a hand-written reward function.

### Task
Each prompt is a story specification describing the title, setting, characters, an important object, a taboo rule the story must respect, and a twist that should be revealed. The output is a two-to-four paragraph story that stays eerie and suspenseful without being graphic, and ends with an unsettling implication. This framing makes it clear what the model must condition on, and it gives a concrete way to build both supervised data and preference data.

### Method overview
The training is intentionally split into two stages with different learning signals.

In supervised fine-tuning, the model is trained to imitate high-quality target stories conditioned on the specification. This stage mainly teaches the model how to follow the format and constraints, how to map the structured fields into coherent narrative content, and how to produce complete multi-paragraph outputs.

In direct preference optimization, the model is trained on pairs of candidate stories for the same specification. One is labeled as preferred because it better matches the desired horror tone and narrative intent, and the other is labeled as less preferred because it is more wholesome or deviates from the target style. This stage shifts probability mass toward preferred outputs without training a separate reward model.

### Data construction
The dataset is built around a common prompt format so that supervised examples and preference pairs share the same input distribution. A random story specification is sampled by choosing values for each field (title, setting, names, object, taboo, twist), then formatting them into a single instruction prompt that ends with a clear marker indicating where the story should begin.

Supervised examples contain one story per specification. Preference examples contain two stories per specification: a preferred story with stronger horror cues and a less preferred story with a more positive tone. For this run, the data is generated with a template-based generator to make the pipeline fully offline and reproducible. The split sizes are small on purpose (train, validation, test for both supervised and preference data), which keeps the run fast but increases the risk of overfitting.

### Training details that matter
The most important detail for both stages is that the prompt is treated purely as context. The training signal is applied only to the story tokens, not to the specification tokens. In other words, the model is not rewarded for predicting the prompt template; it is rewarded only for producing the response given the prompt. This makes the loss reflect the quality of the generated story rather than the fixed structure of the prompt.

The preference stage uses a frozen reference model as a baseline. The reference model is a snapshot of the model before preference optimization, and it is not updated during the preference stage. Training then focuses on how the policy model changes relative to that reference, which improves stability and reduces the chance of drifting away from fluent language.

### Objective functions
Supervised fine-tuning uses a masked next-token cross entropy objective. Let the response tokens be y1 through yT, and let the model define a conditional distribution over tokens. The supervised objective minimizes the average negative log-likelihood of the response tokens given the specification and the preceding response tokens. A binary mask ensures only response tokens contribute to the loss.

Direct preference optimization uses a pairwise objective. For each prompt x, there is a preferred completion y plus and a less preferred completion y minus. We compute the summed log-probability of each completion under the current policy model and under the frozen reference model. The optimization target increases the policy’s relative preference for y plus over y minus compared to the reference, scaled by a temperature-like parameter beta. This run also applies light label smoothing in the preference loss to reduce brittleness when preferences are noisy.

### Results
![Run summary](summary.png)

This run achieves a low supervised test loss and perfect preference accuracy on the held-out preference test set. The reward trends are consistent with preference alignment: the preferred responses receive higher relative scores while the less preferred responses receive lower relative scores. The curves also show that the supervised stage can overfit when trained for too long on a small synthetic dataset, which is expected in this low-data regime.

For qualitative inspection, sampled generations typically follow the specification and maintain an eerie tone. However, some samples show common small-model decoding artifacts such as repetitive tails and visible end-of-sequence markers. These issues affect presentation quality more than the underlying demonstration of the SFT and DPO pipeline, but they are important to address if the goal is a polished creative writing assistant.

### Evaluation and interpretation
The strongest evidence that the pipeline works is the consistent separation between preferred and less preferred outputs under the model after preference training. In a controlled synthetic setting, preference optimization reliably pushes the model toward the target style. The supervised stage provides a necessary foundation by teaching the model how to write stories that match the instruction format; without that foundation, preference optimization alone tends to be unstable or to trade off fluency.

At the same time, these results should be interpreted cautiously. Because the dataset is template-generated, preferred and less preferred stories can differ in easy-to-detect surface patterns. Perfect preference accuracy can therefore overstate real-world alignment ability if the model is mainly learning shortcuts. A stronger test would use harder preference pairs where both candidates are plausible horror stories but one is clearly better along a subtle dimension, such as integrating the taboo rule more naturally or delivering a twist payoff more effectively.

### Limitations
The dataset is small and synthetic, so generalization beyond the prompt style and vocabulary of the generator is not guaranteed. Long supervised training can lead to overfitting and to repetitive generation patterns. Preference accuracy is measured by log-probability comparisons on the same family of prompts, which does not capture creativity, coherence over long narratives, or strict constraint satisfaction.

### Practical improvements
Several straightforward changes would make the system more convincing as a final project.

First, use early stopping or checkpoint selection based on validation performance to avoid overfitting in the supervised stage. Second, improve the preference dataset by generating closer pairs where the distinction is stylistic and semantic rather than lexical. Third, add constraint-based evaluation that checks paragraph count, taboo mention, twist inclusion, and end implication. Finally, apply light post-processing for demos to remove end-of-sequence markers and to mitigate repeated trailing phrases.

### Figure: training curves and rewards
![Curves](curves_updated.png)

### Playing with the model

**Command (one-shot generation)**

```bash
PYTHONPATH=pico-llm python3 -m part2.play_model \
  --checkpoint pico-llm/part2/runs/cn-vm-001/checkpoints/transformer_dpo.pt \
  --device mps \
  --mode oneshot \
  --top_p 0.95 \
  --max_new_tokens 220 \
  --prompt $'You are a creative writing assistant.\nWrite a short horror story that follows this story specification.\n\nTitle: The Door That Wasn\x27t There\nSetting: an old apartment building during a winter blackout\nProtagonist: Mina\nSupporting character: Kai\nImportant object: a brass key\nTaboo rule: never answer knocks after 2 a.m.\nTwist: the sound is coming from inside the walls\n\nConstraints:\n- 2 to 4 paragraphs.\n- Keep it suspenseful and eerie, not graphic.\n- End with an unsettling implication.\n\nStory:\n'
```

**Sample output**

```text
Mina learned the never answer knocks after 2 a.m. the hard way, in an old apartment building during a winter blackout.

When Mina tried to leave, the hallway stretched and the a brass key grew warm in their pocket, as if it was counting something no one could see. Mina insisted it was nothing, yet their smile never reached their eyes. Then it happened: the door open the door that door.

When Mina tried to leave, the hallway stretched and the a brass key grew warm in their pocket, as if it had finally found its owner.
```

### Why the generation can look worse than the metrics

One likely root cause is a **prompt/response tokenization + loss-mask mismatch** in the SFT dataset preparation.

- Earlier data prep concatenated `enc.encode(prompt) + "\n\n" + enc.encode(response)` directly, but did not match the training-time `encode_sft` behavior used in Part 2.
- For long examples, the truncation strategy could drop important response tokens, while the loss mask still assumed a different boundary, causing the model to spend training signal on the wrong tokens (sometimes including parts of the prompt).
- Missing or inconsistent EOS handling can also encourage run-on or repetitive tails during generation.

This can lead to a situation where **loss curves and held-out preference accuracy look strong**, but the model’s *free-form generations* still show repetition, awkward phrasing, or poor adherence to constraints.

**Fix / next step**
- Regenerate SFT token data using `part2.tokenization.encode_sft` (consistent truncation + EOS) and record `loss_start` (completion boundary), then re-train SFT → DPO on the corrected dataset.
