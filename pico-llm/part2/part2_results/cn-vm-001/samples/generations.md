# Samples (same prompts, different checkpoints)

This section shows generations for the *same prompts* using different checkpoints.

## Sample 1 (id=sft-000041)

**Prompt**

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Mirror in the Attic
Setting: a school after everyone has gone home
Protagonist: Mina
Supporting character: Sora
Important object: a child's drawing
Taboo rule: never say the missing person's name out loud
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:

```

### transformer_sft (horror_lexicon_per_100_words=1.5464)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Mirror in the Attic
Setting: a school after everyone has gone home
Protagonist: Mina
Supporting character: Sora
Important object: a child's drawing
Taboo rule: never say the missing person's name out loud
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Mina learned the never say the missing person's name out loud the hard way, in a school after everyone has gone home.

Somewhere nearby, a soft scraping answered every silence, like a patient listener practicing a name. The air felt thinner each time a child's drawing changed hands, as if it was counting something no one could see. At last, Mina understood: the secondary character has been repeating the same day.

When Mina tried to leave, the hallway stretched and the a child's drawing grew warm in their pocket, as if it had finally found its owner.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful
```

### transformer_dpo (horror_lexicon_per_100_words=1.5789)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Mirror in the Attic
Setting: a school after everyone has gone home
Protagonist: Mina
Supporting character: Sora
Important object: a child's drawing
Taboo rule: never say the missing person's name out loud
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Mina only meant to help Sora, but the rule was clear: never say the missing person's name out loud.

Somewhere nearby, a soft scraping answered every silence, like a patient listener practicing a name. Mina insisted it was nothing, yet their smile never reached their eyes. Then it happened: the secondary character has been repeating the same day.

When Mina tried to leave, the hallway stretched and the a child's drawing grew warm in their pocket, as if it had finally found its owner.<|endoftext|> did the last, slow and grateful.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the
```

### transformer_final (horror_lexicon_per_100_words=0.9346)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Mirror in the Attic
Setting: a school after everyone has gone home
Protagonist: Mina
Supporting character: Sora
Important object: a child's drawing
Taboo rule: never say the missing person's name out loud
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Maala was so happy and excited to have her very own recording. She continued to enjoy her writing her story and her drawing instantly.

The end.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## Sample 2 (id=sft-000046)

**Prompt**

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Last Candle
Setting: a rural bus stop at midnight
Protagonist: Iris
Supporting character: Theo
Important object: a child's drawing
Taboo rule: never say the missing person's name out loud
Twist: the protagonist has been the missing person all along

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:

```

### transformer_sft (horror_lexicon_per_100_words=1.5544)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Last Candle
Setting: a rural bus stop at midnight
Protagonist: Iris
Supporting character: Theo
Important object: a child's drawing
Taboo rule: never say the missing person's name out loud
Twist: the protagonist has been the missing person all along

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
In a rural bus stop at midnight, Iris found a child's drawing and wished they hadn't.

The air felt thinner each time a child's drawing changed hands, as if it was counting something no one could see. Theo insisted it was nothing, yet their smile never reached their eyes. Only when the lights failed did the truth land—the protagonist has been the missing person all along.

Iris blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.
```

### transformer_dpo (horror_lexicon_per_100_words=1.0471)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Last Candle
Setting: a rural bus stop at midnight
Protagonist: Iris
Supporting character: Theo
Important object: a child's drawing
Taboo rule: never say the missing person's name out loud
Twist: the protagonist has been the missing person all along

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Somewhere learned the never say the missing person's name out loud the hard way, in a rural bus stop at midnight.

 bus stop, Iris insisted it was nothing, yet their smile never reached their eyes. Somewhere nearby, a soft scraping answered every silence, like a patient listener practicing a name. Only when the lights failed did the truth land—the protagonist has been the missing person all along.

When Iris tried to leave, the hallway stretched and the a child's drawing grew warm in their pocket, as if it had finally found its owner.<|endoftext|><|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last
```

### transformer_final (horror_lexicon_per_100_words=0.0000)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Last Candle
Setting: a rural bus stop at midnight
Protagonist: Iris
Supporting character: Theo
Important object: a child's drawing
Taboo rule: never say the missing person's name out loud
Twist: the protagonist has been the missing person all along

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
 beatenting your decisions is what you have, unless you believe in the end," the lawyer said:

 wrapperAh settled down on and started moving away quickly.

Shall rather than ever being nosy.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## Sample 3 (id=sft-000229)

**Prompt**

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: Borrowed Footsteps
Setting: a mountain motel with a broken sign
Protagonist: Sora
Supporting character: Kai
Important object: a cracked phone
Taboo rule: never answer knocks after 2 a.m.
Twist: the protagonist realizes the sound is coming from inside the walls

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:

```

### transformer_sft (horror_lexicon_per_100_words=3.6082)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: Borrowed Footsteps
Setting: a mountain motel with a broken sign
Protagonist: Sora
Supporting character: Kai
Important object: a cracked phone
Taboo rule: never answer knocks after 2 a.m.
Twist: the protagonist realizes the sound is coming from inside the walls

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Sora learned the never answer knocks after 2 a.m. the hard way, in a mountain motel with a broken sign.

The air felt thinner each time a cracked phone changed hands, as if it was counting something no one could see. Somewhere nearby, a soft scraping answered every silence, like a patient listener practicing a name. Then it happened: the protagonist realizes the sound is coming from inside the walls.

When Sora tried to leave, the hallway stretched and the a cracked phone grew warm in their pocket, as if it had finally found its owner.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last
```

### transformer_dpo (horror_lexicon_per_100_words=4.1237)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: Borrowed Footsteps
Setting: a mountain motel with a broken sign
Protagonist: Sora
Supporting character: Kai
Important object: a cracked phone
Taboo rule: never answer knocks after 2 a.m.
Twist: the protagonist realizes the sound is coming from inside the walls

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Sora learned the never answer knocks after 2 a.m. the hard way, in a mountain motel with a broken sign.

The air felt thinner each time a cracked phone changed hands, as if it was counting something no one could see. Grand insisted it was nothing, yet their smile never reached their eyes. Then it happened: the protagonist realizes the sound is coming from inside the walls.

When Sora tried to leave, the hallway stretched and the a cracked phone grew warm.<|endoftext|> hung from inside the door.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last candle and heard a knock from inside the room
```

### transformer_final (horror_lexicon_per_100_words=3.4483)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: Borrowed Footsteps
Setting: a mountain motel with a broken sign
Protagonist: Sora
Supporting character: Kai
Important object: a cracked phone
Taboo rule: never answer knocks after 2 a.m.
Twist: the protagonist realizes the sound is coming from inside the walls

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
The challenge has a bad ending to our story.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## Sample 4 (id=sft-000278)

**Prompt**

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: A Quiet Room
Setting: a mountain motel with a broken sign
Protagonist: Iris
Supporting character: Nina
Important object: a paper map
Taboo rule: never follow footsteps you can't see
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:

```

### transformer_sft (horror_lexicon_per_100_words=2.6042)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: A Quiet Room
Setting: a mountain motel with a broken sign
Protagonist: Iris
Supporting character: Nina
Important object: a paper map
Taboo rule: never follow footsteps you can't see
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
In a mountain motel with a broken sign, Iris found a paper map and wished they hadn't.

The air felt thinner each time a paper map changed hands, as if it was counting something no one could see. Somewhere nearby, a soft scraping answered every silence, like a patient listener practicing a name. At last, Iris understood: the secondary character has been repeating the same day.

Iris blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|>
```

### transformer_dpo (horror_lexicon_per_100_words=2.1164)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: A Quiet Room
Setting: a mountain motel with a broken sign
Protagonist: Iris
Supporting character: Nina
Important object: a paper map
Taboo rule: never follow footsteps you can't see
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Iris only meant to help Nina, but the rule was clear: never follow footsteps you can't see.

Somewhere nearby, a soft scraping answered every silence, like a patient listener practicing a name.

When Iris tried to leave, the hallway stretched and the a paper map grew warm in their pocket, as if it had finally found its owner.<|endoftext|> insisted it was meant to the protagonist.

When Iris tried to leave, the hallway stretched and the a paper map grew warm in their pocket, as if it had finally found its owner.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last candle
```

### transformer_final (horror_lexicon_per_100_words=1.2500)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: A Quiet Room
Setting: a mountain motel with a broken sign
Protagonist: Iris
Supporting character: Nina
Important object: a paper map
Taboo rule: never follow footsteps you can't see
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
The challenge fast!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## Sample 5 (id=sft-000072)

**Prompt**

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Mirror in the Attic
Setting: a subway platform that feels too long
Protagonist: Theo
Supporting character: Sora
Important object: a child's drawing
Taboo rule: never play the tape past the final click
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:

```

### transformer_sft (horror_lexicon_per_100_words=2.0513)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Mirror in the Attic
Setting: a subway platform that feels too long
Protagonist: Theo
Supporting character: Sora
Important object: a child's drawing
Taboo rule: never play the tape past the final click
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Theo only meant to help Sora, but the rule was clear: never play the tape past the final click.

Somewhere nearby, a soft scraping answered every silence, like a patient listener practicing a name. Sora insisted it was nothing, yet their smile never reached their eyes. At last, Theo understood: the secondary character has been repeating the same day.

When Theo tried to leave, the hallway stretched and the a child's drawing grew warm in their pocket, as if it had finally found its owner.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last candle and heard a knock from inside the room
```

### transformer_dpo (horror_lexicon_per_100_words=1.0309)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Mirror in the Attic
Setting: a subway platform that feels too long
Protagonist: Theo
Supporting character: Sora
Important object: a child's drawing
Taboo rule: never play the tape past the final click
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Theo learned the never play the tape past the final click the hard way, in a subway platform that feels too long.

Somewhere nearby, a soft scraping answered every silence, like a patient listener practicing a name. Mina insisted it was nothing, yet their smile never reached their eyes. At last, Theo understood: the secondary character has been repeating the same day.

When Theo tried to leave, the hallway stretched and the a child's drawing grew warm in their pocket, as if it had finally found its owner.<|endoftext|> did the truth land slow slow, slow and the a child's drawing grew warm in their pocket, as if it had finally found its owner.
```

### transformer_final (horror_lexicon_per_100_words=0.8065)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Mirror in the Attic
Setting: a subway platform that feels too long
Protagonist: Theo
Supporting character: Sora
Important object: a child's drawing
Taboo rule: never play the tape past the final click
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Every day, Kim takes a break from the settling walk to the other side of theJamie's farm adventure. She understood that preparation was that it was important to take good care of her environment.

The next day, everybody appears with curiosity and satisfaction.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## Sample 6 (id=sft-000122)

**Prompt**

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Door That Wasn't There
Setting: a rural bus stop at midnight
Protagonist: Mina
Supporting character: Leah
Important object: a cracked phone
Taboo rule: never follow footsteps you can't see
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:

```

### transformer_sft (horror_lexicon_per_100_words=2.6042)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Door That Wasn't There
Setting: a rural bus stop at midnight
Protagonist: Mina
Supporting character: Leah
Important object: a cracked phone
Taboo rule: never follow footsteps you can't see
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
In a rural bus stop at midnight, Mina found a cracked phone and wished they hadn't.

Leah insisted it was nothing, yet their smile never reached their eyes. The air felt thinner each time a cracked phone changed hands, as if it was counting something no one could see. Then it happened: the secondary character has been repeating the same day.

Mina blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|>mother blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last candle and heard a knock from inside the room, slow and grateful.<|endoftext|> blew out the last
```

### transformer_dpo (horror_lexicon_per_100_words=4.7368)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Door That Wasn't There
Setting: a rural bus stop at midnight
Protagonist: Mina
Supporting character: Leah
Important object: a cracked phone
Taboo rule: never follow footsteps you can't see
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
In a rural bus stop at midnight, Mina found a cracked phone and wished they hadn't.

The air felt thinner each time a cracked phone changed hands, as if it was counting something no one could see. Somewhere nearby, a soft scraping answered every silence, like a patient listener practicing a name. Then it happened: the secondary character has been repeating the same day.

In the mirror-dark, Mina saw Leah behind them—except Leah was still speaking from the other side of the door.<|endoftext|> was still speaking from the other side of the door.<|endoftext|> heard someone knock from inside the room, slow and grateful.<|endoftext|> blew out the last candle and heard a knock
```

### transformer_final (horror_lexicon_per_100_words=2.5316)

```
You are a creative writing assistant.
Write a short horror story that follows this story specification.

Title: The Door That Wasn't There
Setting: a rural bus stop at midnight
Protagonist: Mina
Supporting character: Leah
Important object: a cracked phone
Taboo rule: never follow footsteps you can't see
Twist: the secondary character has been repeating the same day

Constraints:
- 2 to 4 paragraphs.
- Keep it suspenseful and eerie, not graphic.
- End with an unsettling implication.

Story:
Maah!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```
