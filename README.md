# Topic Modelling — Agentic AI

A research-grade topic modeling workflow following **Braun & Clarke's (2006) six-phase Reflexive Thematic Analysis**, enhanced with BERTopic computational NLP and Mistral LLM. Built as a HuggingFace Space with Gradio UI.

---

## 📋 Overview

This application performs thematic analysis on Scopus academic paper exports using:
- **BERTopic** for sentence-level topic modeling
- **Mistral LLM** for topic labeling and taxonomy mapping
- **Braun & Clarke** reflexive thematic analysis framework (6 phases)
- **PAJAIS/ICIS** research taxonomy for category mapping

The workflow is researcher-in-the-loop: each phase produces output that the researcher reviews and approves before proceeding.

---

## 🏗️ Architecture

The app is organized into **20 blocks** across 5 sections:

| Section | Blocks | Description |
|---------|--------|-------------|
| **B1–B3: Setup** | B1–B3 | Imports, agent creation, visual theme |
| **B4–B10: Helpers** | B4–B10 | Pure Python functions, no UI |
| **B11–B17: UI Layout** | B11–B17 | gr.Blocks with native components |
| **B18: Event Wiring** | B18 | Connect UI to functions |
| **B19: Launch** | B19/20 | Start server |

---

## 🔄 Workflow Phases

The following 6-phase workflow is followed (pausing for researcher approval between phases):

### Phase 1: Familiarization
- Load Scopus CSV, inspect data landscape
- Agent shows paper count, sentence estimates, columns

### Phase 2: Initial Codes
- Run BERTopic discovery (abstract or title config)
- Label topics with Mistral LLM
- Review coded topics in review table

### Phase 3: Searching for Themes
- Consolidate topics into themes
- Researcher groups/renames topics
- Themes recomputed with new centroids

### Phase 4: Reviewing Themes
- Saturation analysis — check all theme pairs
- Determine if further consolidation is possible

### Phase 5: Defining & Naming Themes
- Finalize theme definitions and names
- Second run config (abstract vs title) repeated

### Phase 5.5: Taxonomy Comparison
- Map themes against PAJAIS taxonomy (Jiang et al., 2019)
- Identify MAPPED vs NOVEL themes

### Phase 6: Producing Report
- Generate comparison CSV (abstract vs title)
- Export 500-word narrative for publication

---

## 🛠️ Tools (7 Total)

| # | Tool | Description |
|---|------|-------------|
| 1 | `load_scopus_csv` | Load CSV, show preview, estimate sentence count |
| 2 | `run_bertopic_discovery` | Split→embed→cluster→Plotly charts |
| 3 | `label_topics_with_llm` | Mistral labels topics from evidence |
| 4 | `consolidate_into_themes` | Merge topics into themes |
| 5 | `compare_with_taxonomy` | Map themes to PAJAIS taxonomy |
| 6 | `generate_comparison_csv` | Compare themes across runs |
| 7 | `export_narrative` | Generate 500-word Section 7 narrative |

---

## 🚀 Quick Start

1. **Upload** your Scopus CSV export (must have "Title" and "Abstract" columns)
2. **Review** the data preview and confirm proceeding
3. **Type** `run abstract only` or `run title only` in chat to begin
4. **Follow** the agent's prompts — each phase ends with "**Review the table below. Edit Approve/Rename columns, then click Submit Review to Agent.**"
5. **Progress** is shown via emoji indicator (✅ complete, ⬜ pending)
6. **Download** all outputs from the Download tab when complete

---

## 📦 Requirements

```bash
langchain
langchain-mistralai
langgraph
langchain-core
bertopic
sentence-transformers
numpy
pandas
plotly
kaleido
gradio
python-dotenv
```

See `requirements.txt` for exact versions.

---

## 📁 Output Files

Generated checkpoints are saved to `/tmp/checkpoints/` and `/tmp/`:

| File | Phase |
|------|-------|
| `rq4_abstract_emb.npy` / `rq4_title_emb.npy` | Phase 2 |
| `rq4_abstract_summaries.json` / `rq4_title_summaries.json` | Phase 2 |
| `rq4_abstract_labels.json` / `rq4_title_labels.json` | Phase 2 |
| `rq4_abstract_themes.json` / `rq4_title_themes.json` | Phase 3 |
| `rq4_abstract_taxonomy_map.json` / `rq4_title_taxonomy_map.json` | Phase 5.5 |
| `rq4_comparison.csv` | Phase 6 |
| `rq4_narrative.txt` | Phase 6 |
| `rq4_abstract_*.html` / `rq4_title_*.html` | Visualizations (4 files) |

Download tab shows all files by phase.

---

## 💬 Chat Commands

| Command | Description |
|---------|-------------|
| `run abstract only` | Start Phase 2 with abstract config |
| `run title only` | Start Phase 2 with title config |
| `run abstract` / `run title` | Same as above |
| `change threshold to X` | Adjust clustering threshold |
| `show topic X papers` | List all papers in topic X |
| `show topic X` | Show 5 nearest centroid sentences for topic X |
| `approve` | Approve current table state |
| `re-run 0.65` / `re-run 0.8` | Re-run BERTopic with new threshold |
| `code X looks wrong` | Show why topic was labeled that way |

---

## 🎯 Key Features

- **Researcher-in-the-loop**: Every phase pauses for explicit approval
- **Native Gradio UI**: No custom HTML/CSS, fully themed
- **Braun & Clarke compliance**: 6-phase framework with saturation checks
- **PAJAIS taxonomy mapping**: Compare themes against established IS categories
- **Cross-run comparison**: Abstract vs title theme side-by-side
- **500-word narrative export**: Ready for publication Section 7
- **Download all outputs**: CSVs, JSONs, Plotly charts in one click

---

## 📬 Deployment

This app is designed for **HuggingFace Spaces**. To run locally:

```bash
python app.py
```

Access at `http://localhost:7860`

For HuggingFace Spaces deployment, ensure the Space uses `app.py` as the entry point and the Gradio `theme` and `footer_links` are configured as in `app.py` B20.

---

## 📄 License

Apache 2.0 — see repository for full license terms.

---

## 🙋‍♂️ Support

- Issues: Open a GitHub issue in the repository
- Questions: Check the chat prompts and follow the Braun & Clarke workflow
- Documentation: Each phase is explicitly labeled in agent responses

---

*Version 3.0.0 | April 2026 | Stack: Gradio 5.x + LangGraph + Mistral + BERTopic*