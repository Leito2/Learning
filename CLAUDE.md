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

## English practice: chat always, artifact as a plus
The English Corner, recaps and vocabulary live **in the chat**, never inside vault notes, and the chat is never replaced. The `/learnenglish` skill (`.claude/skills/learnenglish/`) adds a live Artifact named `learnenglish` (map, tiny practice and words per note). It is a plus: update it only after the user runs `/learnenglish`. Once the user has run it in a session, keep adding one block to the page after each new note until the topic changes (then rebuild it from the template at the same URL). The same applies when reviewing a course or researching a topic.

## Windows notes
Paths over 260 characters need `core.longpaths=true`. Never use `:` or `/` in file names.
