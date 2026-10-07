---
name: learnenglish
description: English Corner for a B2-advanced learner, plus a live study page (an Artifact named "learnenglish"). Gives 4-6 harder words, phrasal verbs, collocations and compound phrases from what the user is studying or researching, and keeps a minimalist visual sheet with a map, a tiny practice and the vocabulary of each note. Use when the user runs /learnenglish or asks to practice English from a note, course or research topic.
argument-hint: "[file path | pasted text | topic] [es] [quiz] [reset]"
---

# learnenglish

The user is a Spanish native at B2-advanced English who learns AI/ML Engineering. This skill has **two outputs**:

1. **Chat (always):** recap + English Corner table. The chat is never replaced.
2. **Artifact "learnenglish" (a plus):** one private page that grows note by note. It is a shallow, fundamentals-only micro summary, not a second copy of the course.

Never write this vocabulary into vault notes or other files, unless the user asks.

## 1. Source of the content (first match wins)
1. A file path in the argument: read it.
2. Pasted text in the argument.
3. A topic word: use it as the theme, draw from what was just discussed.
4. No argument: the last note, lesson or research result in this conversation.

## 2. Chat output (keep exactly this shape)
For notes and lessons, start with a 2-line recap. Then:

**English Corner**

| Expression | Meaning | Example |
|---|---|---|
| **fall back on** (phrasal verb) | Use something else when the first option fails | The gateway *falls back on* a local model when the cloud is down. |

- 4 to 6 rows: about 2 phrasal verbs, 2 collocations or formal terms, 1-2 compound phrases or sentence frames. The type goes in brackets.
- Meaning in simple words, 12 words at most. The example is one sentence about the **current topic**.
- Real technical English (docs, interviews). No slang, no rare words.
- Never repeat an expression already given in this conversation or already on the page.
- If the argument has `es`, add a fourth column **En español**.
- `quiz`: instead of the table, ask 4 fill-in-the-blank questions from earlier expressions, wait, then correct kindly.
- Close with one line: "Tell me if any sentence felt hard and I will simplify the next ones."

## 3. The artifact (do this after the chat output)
Design source: `.claude/skills/learnenglish/template.html`. It is a finished page for the topic "LLM gateways" and defines the look (tokens, light and dark themes, phone layout). **Keep its CSS and structure; change only content.** Before drawing or editing a diagram, load the `artifact-diagramming` skill. Before writing the page, load `artifact-design`. Use the same file path every time, so the URL stays the same.

Find the page: `Artifact` action `list` and look for the title `learnenglish`. If it does not exist, publish the template once (icon `book`) and keep that URL.

**Page structure (do not change the order):**
1. Header: topic name, course and project chips, "Notes so far".
2. **The map:** one inline SVG that shows the mechanism of the topic (data flow, states, or the difference between two options). Label every arrow. Redraw it only when the new note changes the mechanism; then extend the same drawing instead of adding a second one.
3. **One block per note**, newest at the bottom, each with exactly:
   - **Core idea:** 2-3 bullets, fundamentals only.
   - **Try it:** one tiny practice taken from the note (a calculation or a 4-8 line snippet with the answer in comments). No setup.
   - **Words from this note:** the same 4-6 expressions given in chat.
   - **Quick check:** 2 foldable questions.
4. Footer: next note.

**Update rules**
- **Same topic, new note:** add one block, update "Notes so far", extend the map only if needed, then republish the same path. Do not rewrite old blocks.
- **Topic changed, or the user says `reset`:** start from `template.html`, replace the topic, map and blocks completely, keep the CSS. Republish to the **same URL**. Do not keep old topics.
- Never delete the artifact unless the user asks.
- If the user is researching (not a course), a "note" is one research finding or one source: same block shape.

**Keep it minimal:** no extra sections, no decorative images, no repeated content between map, ideas and words. Every fact once. English in the page; short and plain.

## 4. After publishing
Tell the user in one line that the page was updated and give the link. Do not repeat its content in the chat.
