"""
pages/the_story.py

Narrative essay: From RAG to Orchestration — what building a legal research AI
actually teaches you. Self-contained markdown; no pipeline imports.
"""

import streamlit as st

st.set_page_config(page_title="The Story", page_icon="📖")

_STORY = """
# From RAG to Orchestration: What Building a Legal Research AI Actually Teaches You

*By Stephen Dubay — Azure Cloud Architect transitioning to AI Engineering*

---

## Table of Contents

1. [Why This Project Started](#1-why-this-project-started)
2. [The Original Goals — and the First Assumption That Broke](#2-the-original-goals-and-the-first-assumption-that-broke)
3. [How the Goal Evolved](#3-how-the-goal-evolved)
4. [The Process: Source, Structure, and Chunking](#4-the-process-source-structure-and-chunking)
5. [Key Decisions Along the Way](#5-key-decisions-along-the-way)
6. [Challenges We Worked Through](#6-challenges-we-worked-through)
7. [The Broader AI Lesson](#7-the-broader-ai-lesson)
8. [Current State](#8-current-state)
9. [Final Takeaways](#9-final-takeaways)

---

## 1. Why This Project Started

I've spent 25 years designing and operating cloud infrastructure on Azure. I know how production systems work — not just how they get built, but how they fail, how they scale, and what it takes to operate them honestly over time.

When the AI wave hit in earnest, I wasn't a skeptic. I was curious about something specific: could AI be reliable in a domain where being wrong has real consequences?

Legal research seemed like the right test. Supreme Court opinions are public, structured, foundational, and legally meaningful. If an AI system could answer questions about them accurately — not just plausibly, but accurately, with traceable sources — that would mean something. If it couldn't, I wanted to understand exactly why.

So I started building.

The initial goal was modest: load landmark Supreme Court opinions into a vector store, and see whether a language model could answer questions from them. Basic retrieval-augmented generation. Nothing novel. The point was to learn the stack and stress-test the idea.

What I didn't expect was how quickly the project would force deeper questions — about legal structure, about evaluation, about what it actually means to build a *trustworthy* AI system rather than an impressive-sounding one.

> *"At first, the goal was simple: take Supreme Court opinions, put them into a vector store, and see whether an AI assistant could answer questions from them. But almost immediately, the project forced a deeper question: what does it actually mean for AI to understand legal authority?"*

---

## 2. The Original Goals — and the First Assumption That Broke

**The original goals were straightforward:**

- Build a working RAG system over U.S. Supreme Court cases
- Use publicly available case law as the source corpus
- Retrieve relevant passages accurately
- Allow an LLM to answer questions using those retrieved passages
- Test whether local models could support legal research workflows
- Create a practical foundation for a legal research assistant — not just a demo

**Key early assumptions:**

- Case text could be ingested like ordinary documents
- Chunking the text and embedding it would be enough to support retrieval
- Landmark cases would be a good starting test set
- A local model plus FAISS vector search could provide a low-cost proof of concept

**The first assumption to break: local model viability.**

The earliest prototype used Ollama running locally on my development machine — an i7-11800H with integrated Xe graphics. Capable hardware for most tasks. Not capable enough for local inference at any useful speed. Seven seconds to generate a single word. The local model path closed immediately.

Switching to the Anthropic API solved the latency problem and introduced a useful constraint: cost discipline. That constraint shaped almost every subsequent architecture decision — which model handles which task, how much context to send, how to classify queries before they reach the expensive model.

The final answer was a two-model strategy: **Claude Haiku** handles classification, routing, and grounded research synthesis — fast, cheap, and accurate for structured tasks. **Claude Sonnet** is reserved for cross-case comparative synthesis, where reasoning demands are higher. One shared daily cost cap across all sessions.

> *"Hardware constraints are architectural constraints. The first version of a system isn't a prototype of the final design — it's a list of assumptions waiting to be invalidated."*

---

## 3. How the Goal Evolved

The project shifted from *"build a chatbot over case text"* to something more deliberate:

> **Build a legally aware research platform where the structure, authority, and relationships inside case law are preserved — and then build the governed pipeline that keeps it honest as the corpus grows.**

That evolution happened in four stages:

**Stage 1 — Basic RAG**
*Can I ask a question and get an answer from a Supreme Court opinion?*
Get retrieval working. Get generation working. See what breaks.

**Stage 2 — Structured Legal Retrieval**
*Can the system know which part of the case it's using?*
Distinguish majority opinions from concurrences, dissents, syllabuses, and procedural background. Make opinion type a first-class metadata field — not a label, but a reasoning asset.

**Stage 3 — Legal Research Product**
*Can this become something usable by students, researchers, or public users?*
Add evaluation, cost controls, rate limiting, a real UI, and source citations. Close the gap between "it works on my machine" and "it works reliably for people I've never met."

**Stage 4 — Governed AI Pipeline**
*Can corpus growth and evaluation be made systematic rather than manual?*
This is where the project lives now. Agent 1 recommends new cases using Sonnet with a human approval gate before anything touches the manifest. Future agents handle ingestion, validation, and evaluation question generation. The system governs itself.

---

## 4. The Process: Source, Structure, and Chunking

### Source Selection

Public Supreme Court sources — primarily Justia and CourtListener — expose different fields: HTML opinion text, plain text, syllabus, opinion type, authoring justice, citation data, case metadata. Not all sources expose all fields. Not all fields are consistently populated.

The source of truth matters before a single embedding is computed. In legal AI, source quality affects every downstream result.

### Ingestion and the Embedding Model Decision

The original embedding model was `nomic-embed-text` (768-dimensional, Ollama). When the pipeline moved to Streamlit Cloud, Ollama was no longer viable. The replacement — `BAAI/bge-small-en-v1.5` (384-dimensional, HuggingFace) — was chosen specifically for its balance of retrieval quality and memory efficiency at 200,000-chunk scale, evaluated against `all-MiniLM-L6-v2` and `all-mpnet-base-v2`.

**The cost of that switch:** the entire FAISS index had to be rebuilt from scratch. Every embedding model change invalidates the existing index. That is not a footnote — it's a constraint that shapes every future decision about model selection.

**The payoff:** MRR (Mean Reciprocal Rank) improved from 0.632 to 0.670. The smaller index left headroom for model and application memory at scale.

> *"You don't get to change your embedding model and keep your index. That's a law, not a guideline."*

### Annotation and Metadata

Every chunk in the vector store carries structured metadata:

```json
{
  "case_title": "Mapp v. Ohio",
  "citation": "367 U.S. 643",
  "section_type": "majority_opinion",
  "author": "Justice Clark",
  "chunk_text": "..."
}
```

These are not decorative catalog fields. They are part of the reasoning system. A retrieved passage without its opinion type is a legal statement without its authority. The system knows whether a passage came from the majority, a concurrence, or a dissent — and that knowledge shapes what it can honestly say.

### Chunking Strategy

Generic chunking breaks legal meaning:

- A rule separated from its reasoning
- A holding separated from the facts
- A dissent retrieved as if it were controlling law

Chunking had to become intentional: preserve section boundaries, keep opinion type attached to every chunk, include case metadata, avoid mixing majority and dissenting reasoning, balance context preservation against retrieval precision.

> *"In legal AI, bad chunking is not just a technical flaw. It can produce legally misleading answers."*

---

## 5. Key Decisions Along the Way

### Decision 1: LLM-Native Classification Over Keyword Routing

The system routes incoming questions into four categories:

| Category | Description |
|---|---|
| **META** | Questions about the app itself |
| **OVERVIEW** | Requests for a case summary |
| **COMPARATIVE** | Cross-case doctrinal evolution questions |
| **RESEARCH** | Specific legal text retrieval against a single case |

The first approach was keyword-based routing — scan the question for signal words, route accordingly.

Then someone typed: *"What is your purpose?"*

The keyword router sent it to RESEARCH. No case name, no legal terminology, no comparative signal — just a plain English question about what the app does. Keyword lists are brittle at the edges. They handle the cases you thought of when you wrote them.

The fix: a single Haiku call with structured category definitions, positive and negative examples per category, and explicit tie-break rules for boundary cases. Validated 12 for 12 on a boundary test set.

> *"Route by meaning, not by vocabulary. A cheap LLM call with well-structured instructions is more robust than a keyword list that grows brittle at the edges."*

### Decision 2: FAISS-First Comparative Retrieval

For cross-case questions, the original approach asked Haiku to identify which cases were relevant from the manifest list. The problem: the model's knowledge of corpus contents is not the same as what the retrieval system actually has.

The better approach:

1. Run `similarity_search_with_score(question, k=50)` across the full corpus
2. Group results by case title
3. Normalize hit counts by each case's total chunk count to remove size bias

Without normalization, larger opinions dominate regardless of actual relevance. *Mapp v. Ohio* at 200 chunks would always outrank *Engel v. Vitale* at 80 chunks, even when the question is about school prayer.

> *"Ask the vector store which cases are relevant. Don't ask the LLM to remember. Retrieval is more reliable than recollection."*

### Decision 3: Multi-Model Cost Strategy

Not every task deserves your best model:

- **Haiku:** Classification, overview generation, grounded research synthesis
- **Sonnet:** Cross-case comparative synthesis, orchestration layer case selection
- **Shared daily cost cap:** $2.00 default, tracked per model with a per-model pricing table

The discipline of that constraint turned out to be useful, not just economical. Knowing which tasks genuinely require stronger reasoning — and which ones only feel like they do — is a design skill.

### Decision 4: Include Dissents and Concurrences

Majority opinions alone miss the richness of Supreme Court reasoning. Dissents and concurrences often shape future law. For education and research, they are essential.

But they must be tagged — not mixed with controlling holdings. Retrieving dissenting language as if it were the holding of the Court is not a retrieval error. In a legal context, it's specifically misleading.

### Decision 5: Evaluation as a First-Class System Component

Early on, I evaluated the system by asking it questions and seeing whether the answers sounded right. That's not evaluation. That's an optimism test.

The current framework: **73 human-written test questions** across three evaluation batches, organized by ingestion wave. Each question specifies:

- **Expected terms** — words or phrases that must appear in retrieved context
- **Forbidden terms** — signals of a wrong or hallucinated answer
- **Routing ground truth** — which case this question should resolve to

Four metrics are tracked per run: routing accuracy, MRR, context term coverage, and answer term rate. New cases don't ship to the live corpus until their evaluation questions are written and passing.

> *"Evaluation cannot be an afterthought. It has to be built alongside the system — and every new case requires new evaluation questions before it can be trusted in production."*

---

## 6. Challenges We Worked Through

### Challenge 1: The Brown I / Brown II Routing Bias

*Brown v. Board of Education* appears twice in the corpus. Brown I (1954) holds that racially segregated schools are unconstitutional. Brown II (1955) addresses the pace of implementation — the source of "with all deliberate speed." They are different cases with different holdings.

For a while, questions about *Brown* were consistently routing to Brown II instead of Brown I. Questions about the constitutional holding were returning passages about implementation timelines.

The cause was an arithmetic artifact in the lexical scorer. Brown II's case title has slightly fewer tokens than Brown I's. In the scoring formula, that smaller denominator inflated Brown II's score on any question that didn't explicitly mention "1954" or "Brown I."

The system wasn't confused about constitutional law. It was confused about a division problem.

**The fix:** `_ordinal_ranks()` using real `date_filed` metadata to assign chronological ordinals to same-named sibling decisions. Roman numeral references in questions — "Brown I," "Brown II" — resolve against actual filing dates. Fully general: handles any future same-name pairs without hardcoding any case names.

**Result:** Routing accuracy 29/30 on the second evaluation batch, 18/18 on the first, no regressions.

> *"Legal identity lives in metadata. When the system got the identity of a case wrong, it got the law wrong. The fix required going back to the source of truth — the filing date — and making the system's reasoning explicit in code."*

### Challenge 2: The Dual Sonnet Call Bug

Agent 1 (`case_selector.py`) recommends which cases to add to the corpus next, with a human approval gate before anything is written to the manifest. The original design called Sonnet twice: once to generate recommendations for display, and once to generate the list actually written to the manifest after approval.

Two calls to the same model with the same prompt don't produce the same output. Language models are not deterministic. The human was approving one set of recommendations. The system was writing a different set to the manifest.

**The fix:** Call Sonnet once. Cache the output to `pending_cases.json` before presenting it for review. Write from the cache after approval. One model call. One source of truth.

> *"Cache before the human gate. Any agentic step with a review/approval checkpoint must commit the model's output to disk before presenting it for approval. Re-calling the model after approval is not a retry — it's a different conversation with a potentially different answer."*

This generalizes beyond legal AI. Any workflow where a human reviews AI output before it takes effect has this vulnerability if the handoff between generation and execution isn't made explicit.

### Challenge 3: Hallucination and Source Trust

The system must not invent legal principles or misstate holdings. The response is architectural, not just a prompt instruction:

- Ground every answer in retrieved text
- Preserve source citations in every response
- Use forbidden terms in evaluation to catch wrong-answer patterns
- Separate controlling holdings from persuasive or dissenting reasoning
- A faithfulness guard flags any legal authority cited in an answer that doesn't appear in the retrieved context

### Challenge 4: Moving from Demo to Governed System

A demo answers a few questions. A governed system requires:

- Repeatable ingestion with manifest tracking
- Structured evaluation per ingestion batch
- Cost controls and rate limiting
- Human oversight on corpus changes
- Clear legal-use boundaries

The multi-agent orchestration layer is the answer to this challenge. It's not just about scaling the corpus. It's about making corpus growth something the system can do safely, with every new case passing evaluation before it reaches users.

---

## 7. The Broader AI Lesson

This project reflects a larger shift in where the real AI engineering work lives.

There's a spectrum of AI involvement:

1. Using AI directly through prompts
2. Building prompt workflows and chains
3. Building vector stores and domain-specific corpora
4. Orchestrating multiple models and tools with human oversight
5. Building governed AI systems — where the question isn't just *"does it work?"* but *"can we trust it, and can we prove it?"*

This project moved from level 1 through level 5. The corpus didn't get there by adding cases. It got there by building the system that governs how cases get added.

The specific lesson from both the Brown I/II routing bug and the dual Sonnet call bug is the same: both failures came from implicit assumptions about identity and continuity that weren't guaranteed by the implementation. The fixes required making those assumptions explicit in code — in metadata, in caching, in the handoff between agents.

> *"The future value is not in asking better prompts. It's in building better context, better retrieval, better evaluation, and better oversight — and then making those things systematic."*

That's what production AI engineering actually is. And it turns out that 25 years of thinking about production infrastructure is directly applicable — because the problems are the same. Where are your assumptions? What are the tests that expose them? Where does the system have to be explicit rather than relying on things working out?

---

## 8. Current State

**What the project has proven:**

- A production-quality RAG system over Supreme Court cases is feasible and deployable on a free hosting tier
- Legal document structure must be preserved *before* data enters the vector store — it cannot be recovered after
- Metadata is a reasoning asset, not a catalog field
- LLM-native classification outperforms keyword routing for natural language queries
- FAISS-first comparative retrieval with size normalization outperforms LLM-based case identification
- Evaluation questions per case are necessary to prevent false confidence at scale
- Multi-agent orchestration with human-in-the-loop approval is the right architecture for governed corpus growth
- The cheapest model that solves the task is the right model — but knowing which task requires which model is a design decision, not a default

**What still needs work:**

- Broader case ingestion — current corpus is 19 cases, target is 100
- Agent 4 (Question Generator) to automate evaluation question creation for new ingestion batches
- Citation graph linking related cases across doctrinal lines
- Self-hosted open-source model option on Azure VM (directly relevant to the infrastructure portfolio)
- Stronger UI and clearer legal-use disclaimers for public users
- More systematic testing at scale as the corpus grows

---

## 9. Final Takeaways

- **A capable model is the least of your problems.** The model will generate fluent, confident-sounding text regardless of whether the retrieval was right, the routing was right, or the metadata was right. The hard work is building the context that makes its fluency honest.

- **Metadata is a reasoning asset, not a catalog field.** A retrieved passage without its opinion type is a legal statement without its authority.

- **Route by meaning, not by vocabulary.** Keyword classifiers break on natural language. LLM-native classification with explicit tie-break rules is more robust and the cost difference is negligible.

- **Ask the vector store which cases are relevant. Don't ask the LLM to remember.** Retrieval is more reliable than recollection.

- **Normalize for corpus size bias.** Hit counts in retrieval must be normalized by each case's total chunk count, or larger cases will always dominate results regardless of actual relevance.

- **Cache before the human gate.** Any agentic step with a review/approval checkpoint must commit model outputs to disk before presenting them. Re-calling the model after approval is a different conversation.

- **Evaluation cannot be an afterthought.** Write test questions per case. Track routing accuracy, MRR, context coverage, and answer grounding. False confidence is worse than acknowledged uncertainty.

- **Embedding model changes require index rebuilds.** This is not a footnote. Design around it.

- **Trust is not a feature. It's a system property. And system properties have to be engineered.**

---

*Source code: [github.com/scdubay/ai-SCOTUS](https://github.com/scdubay/ai-SCOTUS)*
*Live app: [ai-scotus.streamlit.app](https://ai-scotus.streamlit.app)*
"""

st.markdown(_STORY, unsafe_allow_html=True)
