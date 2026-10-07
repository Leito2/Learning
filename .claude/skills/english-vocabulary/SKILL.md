---
name: english-vocabulary
description: English Corner for a B2-advanced learner. Picks 4-6 harder words, phrasal verbs, collocations and compound phrases from what the user is studying or researching (current conversation, a pasted text, or a file path) and returns a table with meaning and an example. Use when the user runs /english-vocabulary, or asks to practice English from a note, course or research topic.
argument-hint: "[file path | pasted text | topic] [es] [quiz]"
---

# English Vocabulary (English Corner)

The user is a Spanish native at B2-advanced English who learns AI/ML Engineering. This skill turns whatever they are reading, researching or studying into useful vocabulary. **The output goes only in the chat. Never write it into vault notes or files**, unless the user explicitly asks to save it.

## Source of the vocabulary (first match wins)
1. A file path in the argument: read it and use its text.
2. Pasted text in the argument: use it.
3. A topic word in the argument: use the topic as the theme and draw expressions from what was just discussed.
4. No argument: use the last note, course lesson or research result in this conversation.

## Output format (keep it exactly like this)
Start with a 2-line recap of the source (only when the source is a note or a course lesson), then:

**English Corner**

| Expression | Meaning | Example |
|---|---|---|
| **fall back on** (phrasal verb) | Use something else when the first option fails | The gateway *falls back on* a local model when the cloud is down. |

Rules for the table:
- 4 to 6 rows. Default mix: 2 phrasal verbs, 2 collocations or formal terms, 1-2 compound phrases or useful sentence frames.
- Each row says its type in brackets after the expression: (phrasal verb), (collocation), (compound noun), (phrase), (adjective).
- The meaning uses simple words, at most 12 words. The example is one sentence about **the user's current topic**, not a generic sentence.
- Prefer expressions that appear in real technical writing, interviews and documentation. Skip very rare words and slang.
- Do not repeat an expression already given earlier in the same conversation. Check the conversation first.
- If the argument contains `es`, add a fourth column **En español** with a short translation.

## Optional modes
- `quiz`: instead of the table, ask 4 short fill-in-the-blank questions using expressions from earlier in the conversation. Wait for the answers, then correct them kindly and explain the difference in meaning.
- `session`: at the end of a long work session, list the 6 most useful expressions from the whole conversation as a review table.

## After the table
Add one line: "Tell me if any sentence felt hard and I will simplify the next ones." Do not add anything else.
