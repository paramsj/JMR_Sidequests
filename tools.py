"""tools.py — Sentence-level BERTopic pipeline + Mistral LLM. Version 3.0.0 | 4 April 2026. ZERO for/while/if.

PIPELINE:
  Paper → split into sentences → each sentence gets paper_id + sent_id + metadata
  → embed sentences (384d) → AgglomerativeClustering cosine → centroid nearest 5 sentences
  → Mistral labels topics from sentence evidence + paper metadata
  → one paper can span MULTIPLE topics
"""
from langchain_core.tools import tool
import os
import json
import re
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════
# DEBUG + STATE + CONSTANTS
# ═══════════════════════════════════════════════
DEBUG = True
debug = {True: print, False: lambda *a, **k: None}[DEBUG]

CHECKPOINT_DIR = "/tmp/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NEAREST_K = 5
SENT_SPLIT_RE = r'(?<=[.!?])\s+(?=[A-Z])'
MIN_SENT_LEN = 30

RUN_CONFIGS = {
    "abstract": ["Abstract"],
    "title": ["Title"],
}

_data = {}


# ═══════════════════════════════════════════════
# HELPER: Split text into sentences (regex, no nltk)
# ═══════════════════════════════════════════════
def _split_sentences(text):
    """Split text on sentence boundaries. Filters short fragments (<30 chars).
    Uses regex: split after .!? followed by uppercase letter."""
    raw = re.split(SENT_SPLIT_RE, str(text))
    return list(filter(lambda s: len(s.strip()) >= MIN_SENT_LEN, raw))


# ═══════════════════════════════════════════════
# TOOL 1: Load Scopus CSV
# ═══════════════════════════════════════════════
@tool
def load_scopus_csv(filepath: str) -> str:
    """Load a Scopus CSV export and show preview. Call this first.

    Args:
        filepath: Path to the uploaded .csv file.

    Returns:
        Row count, column names, and sample data."""
    debug(f"\n>>> TOOL: load_scopus_csv(filepath='{filepath}')")
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    _data["df"] = df
    debug(f">>> Loaded {len(df)} rows, {len(df.columns)} columns")
    target_cols = list(filter(lambda c: c in df.columns, ["Title", "Abstract", "Author Keywords"]))
    sample = df[target_cols].head(3).to_string(max_colwidth=80)
    null_counts = ", ".join(list(map(
        lambda c: f"{c}: {df[c].notna().sum()}/{len(df)}", target_cols)))

    # Estimate sentence counts
    sample_sents = df["Abstract"].head(5).apply(_split_sentences).apply(len)
    avg_abstract_sents = sample_sents.mean()
    est_abstract = int(avg_abstract_sents * len(df))
    title_count = int(df["Title"].notna().sum())

    return (f"📊 **Dataset Statistics:**\n"
            f"- **Papers:** {len(df)}\n"
            f"- **Abstract sentences:** ~{est_abstract} (~{avg_abstract_sents:.0f} per paper)\n"
            f"- **Title sentences:** {title_count} (1 per paper)\n"
            f"- **Non-null:** {null_counts}\n\n"
            f"Columns: {', '.join(list(df.columns)[:15])}\n\n"
            f"Sample:\n{sample}")


# ═══════════════════════════════════════════════
# TOOL 2: Sentence-Level BERTopic Pipeline
# ═══════════════════════════════════════════════
@tool
def run_bertopic_discovery(run_key: str, threshold: float = 0.7) -> str:
    """Sentence-level BERTopic: split papers → embed sentences → cosine similarity clustering → centroid nearest 5 → Plotly charts.
    Each sentence keeps paper_id, sent_id, and metadata. One paper can span multiple topics.
    Uses AgglomerativeClustering with cosine distance — groups sentences by similarity threshold.

    Args:
        run_key: One of 'abstract' or 'title' — selects which columns to split into sentences.
        threshold: Cosine distance threshold (0.0-1.0). Lower = stricter = more topics.
                   0.5 = very strict (~2000 topics), 0.7 = recommended (~100 topics, default), 0.8 = loose (~30 topics), 0.9 = very loose (~10 topics).

    Returns:
        Topic summary with sentence counts, paper counts, and 5 nearest centroid sentences."""
    debug(f"\n>>> TOOL: run_bertopic_discovery(run_key='{run_key}', threshold={threshold})")
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    df = _data["df"].copy()
    cols = RUN_CONFIGS[run_key]
    available = list(filter(lambda c: c in df.columns, cols))
    debug(f">>> Columns: {available}")

    # ── Step 1: Assemble text per paper ──
    df["_text"] = df[available].fillna("").agg(" ".join, axis=1)
    df["_paper_id"] = df.index
    debug(f">>> {len(df)} papers assembled")

    # ── Step 2: Split into sentences — regex, no nltk ──
    debug(">>> Splitting into sentences...")
    df["_sentences"] = df["_text"].apply(_split_sentences)
    debug(f">>> Sentence counts: min={df['_sentences'].apply(len).min()}, "
          f"max={df['_sentences'].apply(len).max()}, "
          f"mean={df['_sentences'].apply(len).mean():.1f}")

    # ── Step 3: Explode to sentence-level DataFrame ──
    meta_cols = ["_paper_id", "Title", "Author Keywords", "_sentences"]
    available_meta = list(filter(lambda c: c in df.columns, meta_cols))
    sent_df = df[available_meta].explode("_sentences").rename(
        columns={"_sentences": "text"}).reset_index(drop=True)
    sent_df = sent_df.dropna(subset=["text"]).reset_index(drop=True)
    sent_df["sent_id"] = sent_df.groupby("_paper_id").cumcount()

    # ── Step 3b: Filter out publisher boilerplate sentences ──
    # Scopus abstracts contain copyright/license noise that clustering picks up as topics.
    # These are NOT research content — remove before embedding.
    debug(">>> Filtering publisher boilerplate...")
    _n_before = len(sent_df)
    boilerplate_patterns = "|".join([
        r"Licensee MDPI",
        r"Published by Informa",
        r"Published by Elsevier",
        r"Taylor & Francis",
        r"Copyright ©",
        r"Creative Commons",
        r"open access article",
        r"Inderscience Enterprises",
        r"All rights reserved",
        r"This is an open access",
        r"distributed under the terms",
        r"The Author\(s\)",
        r"Springer Nature",
        r"Emerald Publishing",
        r"limitations and future",
        r"limitations and implications",
        r"limitations are discussed",
        r"limitations have been discussed",
        r"implications are discussed",
        r"implications were discussed",
        r"implications are presented",
        r"concludes with .* implications",
    ])
    clean_mask = ~sent_df["text"].str.contains(boilerplate_patterns, case=False, regex=True, na=False)
    sent_df = sent_df[clean_mask].reset_index(drop=True)
    sent_df["sent_id"] = sent_df.groupby("_paper_id").cumcount()
    debug(f">>> Filtered: {_n_before} → {len(sent_df)} sentences ({_n_before - len(sent_df)} boilerplate removed)")
    n_sentences = len(sent_df)
    n_papers = len(df)
    debug(f">>> {n_sentences} sentences from {n_papers} papers")

    # ── Step 4: Embed sentences (384d, L2-normalized) ──
    # BERTopic FAQ: "normalize them first to force a cosine-related distance metric"
    # Math: for L2-normalized vectors, euclidean²(a,b) = 2(1 - cos(a,b)) → same clusters as cosine
    debug(">>> Embedding sentences with all-MiniLM-L6-v2 (L2-normalized)...")
    docs = sent_df["text"].tolist()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs, show_progress_bar=False, normalize_embeddings=True)
    debug(f">>> Embeddings: {embeddings.shape}, normalized: True")

    # Save checkpoint
    np.save(f"{CHECKPOINT_DIR}/rq4_{run_key}_emb.npy", embeddings)

    # ── Step 5: Agglomerative Clustering with COSINE similarity threshold ──
    # Groups sentences where cosine_distance < threshold → same cluster
    # No dimension reduction. No density estimation. Pure similarity grouping.
    debug(f">>> AgglomerativeClustering cosine threshold={threshold} on 384d embeddings...")
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.cluster import AgglomerativeClustering
    no_umap = FunctionTransformer()
    cluster_model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )
    topic_model = BERTopic(
        hdbscan_model=cluster_model,
        umap_model=no_umap,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
    n_topics = len(set(topics)) - int(-1 in topics)
    n_outliers = int(np.sum(np.array(topics) == -1))
    debug(f">>> {n_topics} topics, {n_outliers} outlier sentences")

    # Store for later tools
    _data[f"{run_key}_model"] = topic_model
    _data[f"{run_key}_topics"] = np.array(topics)
    _data[f"{run_key}_embeddings"] = embeddings
    _data[f"{run_key}_sent_df"] = sent_df

    # ── Step 6: BERTopic Plotly visualizations (skip charts that need 3+ topics) ──
    debug(f">>> Generating visualizations ({n_topics} topics)...")
    # visualize_topics() uses UMAP internally → crashes with < 3 topics
    (n_topics >= 3) and topic_model.visualize_topics().write_html(
        f"/tmp/rq4_{run_key}_intertopic.html", include_plotlyjs="cdn")
    # barchart works with 1+ topics
    (n_topics >= 1) and topic_model.visualize_barchart(
        top_n_topics=min(10, max(1, n_topics))).write_html(
        f"/tmp/rq4_{run_key}_bars.html", include_plotlyjs="cdn")
    # hierarchy needs 2+ topics
    (n_topics >= 2) and topic_model.visualize_hierarchy().write_html(
        f"/tmp/rq4_{run_key}_hierarchy.html", include_plotlyjs="cdn")
    # heatmap needs 2+ topics
    (n_topics >= 2) and topic_model.visualize_heatmap().write_html(
        f"/tmp/rq4_{run_key}_heatmap.html", include_plotlyjs="cdn")
    debug(f">>> Visualizations saved (skipped charts needing more topics)")

    # ── Step 7: Centroid nearest 5 SENTENCES — COSINE similarity ──
    topics_arr = np.array(topics)
    topic_info = topic_model.get_topic_info()
    valid_rows = list(filter(lambda r: r["Topic"] != -1, topic_info.to_dict("records")))

    def _centroid_nearest(row):
        """Find 5 sentences nearest to topic centroid via cosine similarity."""
        mask = topics_arr == row["Topic"]
        member_idx = np.where(mask)[0]
        member_embs = embeddings[mask]
        centroid = member_embs.mean(axis=0)
        # Cosine distance: 1 - cos_sim. For normalized vectors: cos_sim = dot product
        norms = np.linalg.norm(member_embs, axis=1) * np.linalg.norm(centroid)
        cosine_sim = (member_embs @ centroid) / (norms + 1e-10)
        dists = 1 - cosine_sim
        nearest = np.argsort(dists)[:NEAREST_K]

        # 5 nearest sentences with paper metadata
        nearest_evidence = list(map(lambda i: {
            "sentence": str(sent_df.iloc[member_idx[i]]["text"])[:250],
            "paper_id": int(sent_df.iloc[member_idx[i]]["_paper_id"]),
            "title": str(sent_df.iloc[member_idx[i]].get("Title", ""))[:150],
            "keywords": str(sent_df.iloc[member_idx[i]].get("Author Keywords", ""))[:150],
        }, nearest))

        # Count unique papers in this topic + collect their titles
        topic_papers_df = sent_df.iloc[member_idx].drop_duplicates(subset=["_paper_id"])
        unique_papers = len(topic_papers_df)
        paper_titles = list(map(
            lambda idx: str(topic_papers_df.iloc[idx].get("Title", ""))[:200],
            range(min(50, unique_papers))))  # cap at 50 titles per topic

        return {"topic_id": int(row["Topic"]),
                "sentence_count": int(row["Count"]),
                "paper_count": int(unique_papers),
                "top_words": str(row.get("Name", ""))[:100],
                "nearest": nearest_evidence,
                "paper_titles": paper_titles}

    summaries = list(map(_centroid_nearest, valid_rows))
    json.dump(summaries, open(f"{CHECKPOINT_DIR}/rq4_{run_key}_summaries.json", "w"), indent=2, default=str)
    debug(f">>> {len(summaries)} topics saved ({NEAREST_K} nearest sentences each)")

    # ── Format output ──
    lines = list(map(
        lambda s: f"  Topic {s['topic_id']} ({s['sentence_count']} sentences, {s['paper_count']} papers): {s['top_words']}",
        summaries))
    return (f"[{run_key}] {n_topics} topics from {n_sentences} sentences ({n_papers} papers, {n_outliers} outliers).\n\n"
            + "\n".join(lines)
            + f"\n\nVisualizations: /tmp/rq4_{run_key}_*.html (4 files)"
            + f"\nCheckpoints: {CHECKPOINT_DIR}/rq4_{run_key}_emb.npy + summaries.json")


# ═══════════════════════════════════════════════
# TOOL 3: Label Topics with Mistral (sentence evidence)
# ═══════════════════════════════════════════════
@tool
def label_topics_with_llm(run_key: str) -> str:
    """Send 5 nearest centroid sentences + paper metadata to Mistral for labeling.
    Each sentence shows which paper it came from (title + keywords).

    Args:
        run_key: One of 'abstract' or 'title'.

    Returns:
        Labeled topics with sentence-level evidence."""
    debug(f"\n>>> TOOL: label_topics_with_llm(run_key='{run_key}')")
    from langchain_mistralai import ChatMistralAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    summaries = json.load(open(f"{CHECKPOINT_DIR}/rq4_{run_key}_summaries.json"))
    debug(f">>> Loaded {len(summaries)} topics ({NEAREST_K} sentences each)")

    # Limit to top 50 largest topics — prevents Mistral rate limit on 2000+ topics
    MAX_LABEL_TOPICS = 100
    sorted_summaries = sorted(summaries, key=lambda s: s.get("sentence_count", 0), reverse=True)
    summaries_to_label = sorted_summaries[:MAX_LABEL_TOPICS]
    skipped = max(0, len(summaries) - MAX_LABEL_TOPICS)
    debug(f">>> Labeling top {len(summaries_to_label)} topics (skipped {skipped} small clusters)")

    # Format all topics — show sentence + paper metadata as evidence
    topics_block = "\n\n".join(list(map(
        lambda s: (f"Topic {s['topic_id']} ({s['sentence_count']} sentences from {s['paper_count']} papers):\n"
                   f"  Top words: {s['top_words']}\n"
                   f"  {NEAREST_K} nearest centroid sentences:\n"
                   + "\n".join(list(map(
                       lambda e: (f"    - \"{e['sentence'][:200]}\"\n"
                                  f"      Paper: \"{e['title']}\"\n"
                                  f"      Keywords: {e['keywords']}"),
                       s["nearest"])))),
        summaries_to_label)))

    prompt = PromptTemplate.from_template(
        "You are a research topic classifier for academic papers about Technology and Tourism.\n\n"
        "For EACH topic below, you are given the 5 sentences nearest to the topic centroid,\n"
        "plus the paper title and author keywords each sentence came from.\n\n"
        "Return a JSON ARRAY with one object per topic:\n"
        "- topic_id: integer\n"
        "- label: short descriptive name (3-6 words, specific — NOT generic like 'tourism studies')\n"
        "- category: general research area (e.g., 'technology adoption', 'consumer behavior',\n"
        "    'virtual reality', 'social media marketing', 'sustainability', 'cultural heritage',\n"
        "    'AI and machine learning', 'online reviews', 'destination marketing',\n"
        "    'tourist psychology', 'hotel management', 'sharing economy',\n"
        "    'mobile applications', 'research methodology', 'data analytics')\n"
        "    DO NOT use PACIS/ICIS categories — just plain descriptive research area.\n"
        "- confidence: high, medium, or low\n"
        "- reasoning: 1 sentence explaining WHY you chose this label based on the evidence sentences\n"
        "- niche: true or false (true = very specific sub-area with <20 sentences)\n\n"
        "CRITICAL: be SPECIFIC in labels. Do NOT use broad terms.\n"
        "Return ONLY valid JSON array, no markdown.\n\n"
        "Topics:\n{topics}")

    llm = ChatMistralAI(model="mistral-small-latest", temperature=0, timeout=300)
    chain = prompt | llm | JsonOutputParser()
    debug(">>> Calling Mistral (single call, all topics)...")
    labels = chain.invoke({"topics": topics_block})
    debug(f">>> Got {len(labels)} labels")

    # Merge labels with summaries
    labeled = list(map(lambda pair: {**pair[0], **pair[1]},
                       zip(summaries, (labels + summaries)[:len(summaries)])))
    json.dump(labeled, open(f"{CHECKPOINT_DIR}/rq4_{run_key}_labels.json", "w"), indent=2, default=str)
    debug(f">>> Labels saved: {CHECKPOINT_DIR}/rq4_{run_key}_labels.json")

    # Format — show label + evidence sentences + paper source
    lines = list(map(
        lambda l: (f"  **Topic {l.get('topic_id', '?')}: {l.get('label', '?')}** "
                   f"[{l.get('category', '?')}] conf={l.get('confidence', '?')} "
                   f"({l.get('sentence_count', 0)} sentences, {l.get('paper_count', 0)} papers)\n"
                   + "\n".join(list(map(
                       lambda e: f"    → \"{e['sentence'][:120]}...\" — _{e['title'][:60]}_",
                       l.get("nearest", []))))),
        labeled))
    return f"[{run_key}] {len(labeled)} topics labeled by Mistral:\n\n" + "\n\n".join(lines)


# ═══════════════════════════════════════════════
# TOOL 4: Generate Comparison Table
# ═══════════════════════════════════════════════
@tool
def generate_comparison_csv() -> str:
    """Compare Mistral-labeled topics across completed runs. Includes sentence + paper counts.

    Returns:
        Comparison table + CSV path."""
    debug(f"\n>>> TOOL: generate_comparison_csv()")
    completed = list(filter(
        lambda k: os.path.exists(f"{CHECKPOINT_DIR}/rq4_{k}_labels.json"), RUN_CONFIGS.keys()))
    debug(f">>> Completed runs: {completed}")

    def _load_run(run_key):
        labels = json.load(open(f"{CHECKPOINT_DIR}/rq4_{run_key}_labels.json"))
        return list(map(lambda l: {
            "run": run_key, "topic_id": l.get("topic_id", ""),
            "label": l.get("label", ""), "category": l.get("category", ""),
            "confidence": l.get("confidence", ""), "niche": l.get("niche", ""),
            "sentences": l.get("sentence_count", 0),
            "papers": l.get("paper_count", 0),
            "top_words": l.get("top_words", ""),
        }, labels))

    all_rows = sum(list(map(_load_run, completed)), [])
    df = pd.DataFrame(all_rows)
    path = "/tmp/rq4_comparison.csv"
    df.to_csv(path, index=False)
    debug(f">>> Comparison CSV: {path} ({len(df)} rows)")
    return f"Comparison saved: {path} ({len(completed)} runs, {len(df)} topics)\n\n{df.to_string(index=False)}"


# ═══════════════════════════════════════════════
# TOOL 5: Export 500-Word Narrative
# ═══════════════════════════════════════════════
@tool
def export_narrative(run_key: str) -> str:
    """Generate 500-word narrative for research paper Section 7 via Mistral.

    Args:
        run_key: One of 'abstract' or 'title'.

    Returns:
        500-word narrative + save path."""
    debug(f"\n>>> TOOL: export_narrative(run_key='{run_key}')")
    from langchain_mistralai import ChatMistralAI

    labels = json.load(open(f"{CHECKPOINT_DIR}/rq4_{run_key}_labels.json"))
    topics_text = "\n".join(list(map(
        lambda l: f"- {l.get('label', '?')} ({l.get('sentence_count', 0)} sentences from "
                  f"{l.get('paper_count', 0)} papers, category: {l.get('category', '?')}, "
                  f"confidence: {l.get('confidence', '?')}, niche: {l.get('niche', '?')})",
        labels)))

    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.3, timeout=300)
    result = llm.invoke(
        f"Write exactly 500 words for a research paper Section 7 titled "
        f"'Topic Modeling Results — BERTopic Discovery'.\n\n"
        f"Dataset: 1390 Scopus papers on Tourism and AI.\n"
        f"Method: Sentence-level BERTopic — each abstract split into sentences,\n"
        f"embedded with all-MiniLM-L6-v2 (384d), clustered with AgglomerativeClustering (cosine).\n"
        f"Note: One paper can contribute sentences to MULTIPLE topics.\n"
        f"Run config: '{run_key}' columns.\n\n"
        f"Topics discovered:\n{topics_text}\n\n"
        f"Include: methodology justification for sentence-level approach,\n"
        f"key themes, emerging niches, limitations, future work.")

    path = "/tmp/rq4_narrative.txt"
    open(path, "w", encoding="utf-8").write(result.content)
    debug(f">>> Narrative saved: {path} ({len(result.content)} chars)")
    return f"Narrative saved: {path}\n\n{result.content}"


# ═══════════════════════════════════════════════
# TOOL 6: Consolidate Round 1 Topics into Themes
# ═══════════════════════════════════════════════
@tool
def consolidate_into_themes(run_key: str, theme_map: dict) -> str:
    """ROUND 2: Merge fine-grained Round 1 topics into broader themes.
    Researcher decides which topics to group. Recomputes centroids and evidence.

    Args:
        run_key: 'abstract' or 'title'.
        theme_map: Dict mapping theme names to topic ID lists.
                   Example: {"AI in Tourism": [0, 1, 5], "VR Tourism": [2, 3]}

    Returns:
        Consolidated themes with new 5-nearest sentence evidence per theme."""
    debug(f"\n>>> TOOL: consolidate_into_themes(run_key='{run_key}', {len(theme_map)} themes)")

    topics_arr = _data[f"{run_key}_topics"]
    embeddings = _data[f"{run_key}_embeddings"]
    sent_df = _data[f"{run_key}_sent_df"]

    def _build_theme(item):
        """Merge listed topics into one theme. Recompute centroid + 5 nearest."""
        theme_name, topic_ids = item
        mask = np.isin(topics_arr, topic_ids)
        member_idx = np.where(mask)[0]
        member_embs = embeddings[mask]
        centroid = member_embs.mean(axis=0)
        norms = np.linalg.norm(member_embs, axis=1) * np.linalg.norm(centroid)
        cosine_sim = (member_embs @ centroid) / (norms + 1e-10)
        dists = 1 - cosine_sim
        nearest = np.argsort(dists)[:NEAREST_K]

        nearest_evidence = list(map(lambda i: {
            "sentence": str(sent_df.iloc[member_idx[i]]["text"])[:250],
            "paper_id": int(sent_df.iloc[member_idx[i]]["_paper_id"]),
            "title": str(sent_df.iloc[member_idx[i]].get("Title", ""))[:150],
            "keywords": str(sent_df.iloc[member_idx[i]].get("Author Keywords", ""))[:150],
        }, nearest))

        unique_papers = sent_df.iloc[member_idx]["_paper_id"].nunique()

        # Collect paper titles (up to 50)
        topic_papers_df = sent_df.iloc[member_idx].drop_duplicates(subset=["_paper_id"])
        paper_titles = list(map(
            lambda idx: str(topic_papers_df.iloc[idx].get("Title", ""))[:200],
            range(min(50, len(topic_papers_df)))))

        return {"label": theme_name, "merged_topics": list(topic_ids),
                "sentence_count": int(mask.sum()), "paper_count": int(unique_papers),
                "nearest": nearest_evidence, "paper_titles": paper_titles}

    # Add topic_id to each theme (sequential)
    themes_raw = list(map(_build_theme, theme_map.items()))
    themes = list(map(
        lambda pair: {**pair[1], "topic_id": pair[0]},
        enumerate(themes_raw)))
    json.dump(themes, open(f"{CHECKPOINT_DIR}/rq4_{run_key}_themes.json", "w"), indent=2, default=str)
    debug(f">>> {len(themes)} themes saved: {CHECKPOINT_DIR}/rq4_{run_key}_themes.json")

    # Format — show theme + merged topics + evidence
    lines = list(map(
        lambda t: (f"  **{t['label']}** ({t['sentence_count']} sentences, {t['paper_count']} papers)\n"
                   f"    Merged from topics: {t['merged_topics']}\n"
                   f"    Evidence:\n"
                   + "\n".join(list(map(
                       lambda e: f"      → \"{e['sentence'][:120]}...\" — _{e['title'][:60]}_",
                       t["nearest"])))),
        themes))
    return f"[{run_key}] Round 2: {len(themes)} themes consolidated:\n\n" + "\n\n".join(lines)


# ═══════════════════════════════════════════════
# TOOL 7: Compare Themes with PAJAIS Taxonomy
# ═══════════════════════════════════════════════

# Established IS topic taxonomy from:
# Jiang, Liang & Tsai (2019) "Knowledge Profile in PAJAIS"
# Pacific Asia Journal of the AIS, 11(1), 1-24. doi:10.17705/1pais.11101
PAJAIS_TAXONOMY = [
    "Electronic and Mobile Business / Social Commerce",
    "Human Behavior and IS / Human-Computer Interaction",
    "IS/IT Strategy, Leadership, Governance",
    "Business Intelligence and Data Analytics",
    "Design Science and IS",
    "Enterprise Systems and BPM",
    "IS Implementation, Adoption, and Diffusion",
    "Social Media and Business Impact",
    "Cultural and Global Issues in IS",
    "IS Security and Privacy",
    "IS Smart / IoT",
    "Knowledge Management",
    "ICT / Digital Platform / IT and Work",
    "IS Healthcare",
    "IT Project Management",
    "Service Science and IS",
    "Social and Organizational Aspects of IS",
    "Research Methods and Philosophy",
    "E-Finance / Economics of IS",
    "E-Government",
    "IS Education and Learning",
    "Green IT and Sustainability",
]


@tool
def compare_with_taxonomy(run_key: str) -> str:
    """Compare BERTopic themes against established PAJAIS/PACIS taxonomy
    (Jiang, Liang & Tsai, 2019). Identifies which themes map to known
    categories and which are NOVEL/EMERGING (not in existing taxonomy).
    Researcher reviews mapping and approves new theme consolidation.

    Args:
        run_key: 'abstract' or 'title'.

    Returns:
        Mapping table: BERTopic theme → PAJAIS category (or NOVEL)."""
    debug(f"\n>>> TOOL: compare_with_taxonomy(run_key='{run_key}')")
    from langchain_mistralai import ChatMistralAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    # Load themes (prefer consolidated themes, fall back to labels)
    themes_path = f"{CHECKPOINT_DIR}/rq4_{run_key}_themes.json"
    labels_path = f"{CHECKPOINT_DIR}/rq4_{run_key}_labels.json"
    source_path = (os.path.exists(themes_path) and themes_path) or labels_path
    themes = json.load(open(source_path))
    debug(f">>> Loaded {len(themes)} themes from {source_path}")

    # Format themes for Mistral
    themes_text = "\n".join(list(map(
        lambda t: f"- {t.get('label', '?')} "
                  f"({t.get('paper_count', t.get('count', '?'))} papers)",
        themes)))

    taxonomy_text = "\n".join(list(map(lambda c: f"- {c}", PAJAIS_TAXONOMY)))

    prompt = PromptTemplate.from_template(
        "You are an IS research taxonomy expert.\n\n"
        "Compare each BERTopic theme against the established PAJAIS/PACIS taxonomy.\n"
        "For EACH theme, return a JSON ARRAY with:\n"
        "- label: the BERTopic theme name\n"
        "- pajais_match: closest PAJAIS category (or 'NOVEL' if no match)\n"
        "- match_confidence: high, medium, low, or none\n"
        "- reasoning: why this mapping (1 sentence)\n"
        "- is_novel: true if this theme represents an emerging area not in the taxonomy\n\n"
        "Return ONLY valid JSON array.\n\n"
        "BERTopic Themes:\n{themes}\n\n"
        "PAJAIS Taxonomy (Jiang et al., 2019):\n{taxonomy}")

    llm = ChatMistralAI(model="mistral-small-latest", temperature=0, timeout=300)
    chain = prompt | llm | JsonOutputParser()
    debug(">>> Calling Mistral for taxonomy comparison...")
    mappings = chain.invoke({"themes": themes_text, "taxonomy": taxonomy_text})
    debug(f">>> Got {len(mappings)} mappings")

    # Save mapping
    json.dump(mappings, open(f"{CHECKPOINT_DIR}/rq4_{run_key}_taxonomy_map.json", "w"), indent=2, default=str)

    # Count novel vs mapped
    novel = list(filter(lambda m: m.get("is_novel", False), mappings))
    mapped = list(filter(lambda m: not m.get("is_novel", False), mappings))

    # Format output
    mapped_lines = list(map(
        lambda m: f"  ✅ {m.get('label', '?')} → **{m.get('pajais_match', '?')}** "
                  f"(conf={m.get('match_confidence', '?')}) _{m.get('reasoning', '')}_",
        mapped))
    novel_lines = list(map(
        lambda m: f"  🆕 **{m.get('label', '?')}** → NOVEL "
                  f"_{m.get('reasoning', '')}_",
        novel))

    return (f"[{run_key}] Taxonomy comparison (Jiang et al., 2019):\n\n"
            f"**Mapped to PAJAIS categories ({len(mapped)}):**\n" + "\n".join(mapped_lines) +
            f"\n\n**NOVEL / Emerging themes ({len(novel)}):**\n" + "\n".join(novel_lines) +
            f"\n\nSaved: {CHECKPOINT_DIR}/rq4_{run_key}_taxonomy_map.json")


# ═══════════════════════════════════════════════
# GET ALL TOOLS
# ═══════════════════════════════════════════════
def get_all_tools():
    """Return all 7 tools with error handling enabled."""
    tools = [load_scopus_csv, run_bertopic_discovery, label_topics_with_llm,
             consolidate_into_themes, compare_with_taxonomy,
             generate_comparison_csv, export_narrative]
    list(map(lambda t: setattr(t, 'handle_tool_error', True), tools))
    debug(f">>> tools.py: {len(tools)} tools ready (handle_tool_error=True)")
    list(map(lambda t: debug(f">>>   - {t.name}"), tools))
    return tools
