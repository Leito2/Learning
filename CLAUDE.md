# Learning vault: working rules for Claude

This repo is an Obsidian vault (`SW-ML-AI Engineering/`) of compressed AI/ML Engineering courses. The user (Leandro) is a Spanish native with B2-advanced English. Chat language: Spanish. Note language: English.

## Writing notes and courses (always, without being asked)
- Follow the **Compact Deep Format** in `SW-ML-AI Engineering/Continuity Prompt.md`. Full-path wikilinks only, one Mermaid map, no ASCII art, `Estado 2026` callout, Cheat Sheet, Interview Angle, Recall callouts.
- The live plan is `PLAN - Cierre de Gaps y Proyectos.md`. Pending notes are written in the order of its section 2 and section 5.

## Course workflow: one note per turn
1. Write **one** note, then validate it: line count (150-250), fences, no ASCII, and every wikilink resolves.
2. In the chat (Spanish), after each note give:
   - a **2-line recap**;
   - an **English Corner** table with 4-6 items: `Expression | Meaning | Example`. Mix phrasal verbs, collocations, formal terms and compound phrases from that note. No repeats within a course. The example sentence is about the note's topic.
   - the name of the next note.
3. When the last note of a course (the Bridge to Project) is done: give a **5-line mini summary** (what the course covers, 3 key ideas, the project milestone that applies it), update the Master Index, Continuity Prompt and Skills Tree, and make **one commit per course**: `feat: add <course> (N notes)`, then push.
4. If the user asks to see the structure of what is pending, read it from the plan; never invent titles.

## English practice stays in the chat
The English Corner, recaps and vocabulary live **only in the chat**, never inside vault notes. Notes stay in plain, clear English. The `/english-vocabulary` skill (`.claude/skills/english-vocabulary/`) does the same on demand for any topic or text.

## Windows notes
Paths over 260 characters need `core.longpaths=true`. Never use `:` or `/` in file names.
