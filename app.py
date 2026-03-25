import os
import re
import random
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# ---- API KEYS & CLIENT ----
keys = [
    st.secrets["GROQ_API_KEY_1"],
    st.secrets["GROQ_API_KEY_2"],
    st.secrets["GROQ_API_KEY_3"],
    st.secrets["GROQ_API_KEY_4"],
    st.secrets["GROQ_API_KEY_5"],
    st.secrets["GROQ_API_KEY_6"]
]


def choose_client():
    """Pick a random key each time for higher throughput."""
    return Groq(api_key=random.choice(keys))


with open("prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

# ---- EMBEDDING MODEL (CACHED) ----
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ---- SESSION STATE DEFAULTS ----
_defaults = {
    "before_scores": None,
    "optimized_resume": None,
    "after_scores": None,
    "last_resume": None,
    "last_jd": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---- STOP WORDS ----
# Comprehensive list: filters generic English, culture/personality phrases,
# and recruiting jargon so only technical/actionable keywords are scored.
STOP_WORDS = {
    # --- Common English ---
    "the", "and", "for", "with", "that", "this", "from", "have", "will", "are",
    "your", "you", "our", "their", "they", "been", "has", "was", "not", "but",
    "can", "all", "more", "also", "into", "its", "use", "using", "used", "work",
    "team", "help", "make", "well", "both", "who", "what", "when", "where", "how",
    "which", "would", "could", "should", "shall", "may", "might", "must", "does",
    "did", "had", "were", "being", "about", "each", "other", "than", "then",
    "them", "these", "those", "some", "such", "only", "very", "just", "over",
    "after", "before", "between", "through", "during", "under", "above", "below",
    "any", "same", "own", "most", "much", "many", "here", "there", "where",
    "out", "off", "down", "way", "need", "needs", "like", "per", "get", "got",
    "set", "let", "yet", "too", "nor", "via", "etc",
    # --- Generic verbs ---
    "include", "includes", "including", "look", "looking", "find", "finding",
    "meet", "meeting", "allow", "allowing", "ensure", "ensuring", "manage",
    "managing", "maintain", "maintaining", "build", "building", "create",
    "creating", "join", "joining", "shape", "shaping", "take", "taking",
    "give", "giving", "keep", "keeping", "know", "knowing", "want", "wanting",
    "enjoy", "enjoying", "working", "able", "based", "across", "within",
    # --- Recruiting boilerplate ---
    "role", "position", "candidate", "candidates", "ideal", "required",
    "requirements", "responsibilities", "responsibility", "preferred",
    "qualification", "qualifications", "experience", "experiences",
    "opportunity", "opportunities", "company", "organization",
    "looking", "seeking", "hiring", "apply", "application",
    "indicators", "indicator", "person", "right", "strong",
    "ability", "skills", "skill", "years", "year", "week",
    "hour", "hours", "day", "time", "full", "part",
    # --- Culture / personality ---
    "fun", "creative", "engaging", "atmosphere", "brilliant", "jerks",
    "free", "eagerly", "eager", "willing", "open", "comfortable",
    "comfortably", "enjoy", "enjoys", "growth", "massive",
    "collaboration", "collaborative", "collaborating", "collaborate",
    "inside", "large", "small", "future", "values", "value",
    # --- Filler adjectives / adverbs ---
    "new", "relevant", "multiple", "diverse", "growing", "various",
    "best", "high", "good", "great", "excellent", "key", "major",
    "significant", "important", "critical", "specific", "particular",
    "overall", "general", "current", "recent", "available",
    # --- Misc noise ---
    "anyone", "anywhere", "something", "everything", "nothing",
    "whether", "however", "therefore", "although", "though",
    "already", "still", "even", "else", "further", "along",
    "among", "towards", "upon", "around", "especially",
    "few", "dig", "fully", "outside", "built", "designed",
    "ensures", "maintained", "facing", "enhance", "complex",
}


# ---- SCORING FUNCTIONS ----
def keyword_score(resume_text, jd_text):
    jd_words = set(re.findall(r"\b[a-z]{3,}\b", jd_text.lower())) - STOP_WORDS
    resume_words = set(re.findall(r"\b[a-z]{3,}\b", resume_text.lower())) - STOP_WORDS
    matched = jd_words & resume_words
    score = round(len(matched) / len(jd_words) * 100) if jd_words else 0
    missing = jd_words - matched
    return score, matched, missing


def _chunk_text(text, max_words=150):
    """Split text into chunks of ~max_words, respecting paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[\*\-•])", text) if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        words = para.split()
        if current_len + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.extend(words)
        current_len += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks if chunks else [text]


def semantic_score(resume_text, jd_text):
    """Chunk-based semantic similarity to avoid 256-token truncation."""
    resume_chunks = _chunk_text(resume_text)
    jd_chunks = _chunk_text(jd_text)

    all_chunks = resume_chunks + jd_chunks
    embeddings = model.encode(all_chunks, convert_to_tensor=True)

    resume_embs = embeddings[: len(resume_chunks)]
    jd_embs = embeddings[len(resume_chunks) :]

    sim_matrix = util.cos_sim(jd_embs, resume_embs)
    best_per_jd = sim_matrix.max(dim=1).values
    score = best_per_jd.mean().item()

    return round(score * 100, 1)


def compute_scores(resume_text, jd_text):
    """Compute keyword + semantic scores locally (zero LLM tokens)."""
    kw_score, matched, missing = keyword_score(resume_text, jd_text)
    sem_score = semantic_score(resume_text, jd_text)
    return {
        "keyword_score": kw_score,
        "semantic_score": sem_score,
        "matched": matched,
        "missing": missing,
    }


def show_scores(scores, label):
    """Render scores in the UI."""
    kw = scores["keyword_score"]
    sem = scores["semantic_score"]
    matched = scores["matched"]
    missing = scores["missing"]

    st.markdown(f"#### {label}")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Keyword Match", f"{kw}%")
        st.progress(kw / 100)
        if kw >= 85:
            st.success("Strong keyword alignment")
        elif kw >= 70:
            st.warning("Good — can improve")
        else:
            st.error("Needs more keywords")

    with col2:
        st.metric("Semantic Match", f"{sem}%")
        st.progress(sem / 100)
        if sem >= 75:
            st.success("Strong semantic alignment")
        elif sem >= 60:
            st.warning("Good — can improve")
        else:
            st.error("Low contextual alignment")

    with st.expander(f"Matched keywords ({label})"):
        st.write(", ".join(sorted(matched)) if matched else "None")

    with st.expander(f"Missing keywords ({label})"):
        st.write(", ".join(sorted(missing)) if missing else "None")


# ---- SINGLE-PASS LLM OPTIMIZATION ----
def extract_metrics(text):
    """Pull every quantified metric from the resume text so we can pass them
    explicitly to the LLM as a 'do not drop' checklist."""
    # Match patterns like "by 30%", "50,000+", "2 hours", "$1.5M", "35%", etc.
    lines = text.split("\n")
    metrics = []
    for line in lines:
        # Find lines that contain numbers with context
        if re.search(r"\d", line):
            # Extract the metric phrase (number + surrounding context)
            found = re.findall(
                r"(?:improving|reducing|increased|decreased|processed|saving|"
                r"eliminating|enabling|automating|built|handling|serving|"
                r"managing|processing|tracking|flagging|surfacing)"
                r"[^,\n]{0,80}?\d[\d,.]*\+?\s*(?:%|percent|hours?|minutes?|"
                r"million|billion|records|departments|weekly|daily|monthly)?",
                line, re.IGNORECASE
            )
            metrics.extend(found)
            # Also catch standalone patterns like "by X%", "X+ records"
            standalone = re.findall(
                r"(?:by\s+)?\d[\d,.]*\+?\s*(?:%|percent)",
                line, re.IGNORECASE
            )
            for s in standalone:
                if s not in " ".join(metrics):
                    metrics.append(s.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for m in metrics:
        m = m.strip()
        if m and m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def optimize_resume(jd, resume_text, missing_keywords):
    """Single LLM call with full context: JD + resume + missing keywords + metrics."""
    client = choose_client()

    missing_sorted = sorted(missing_keywords)
    missing_list = ", ".join(missing_sorted)

    # Extract metrics as a compact checklist
    metrics = extract_metrics(resume_text)
    metrics_str = " | ".join(metrics) if metrics else "None found"

    user_content = f"""Here is my resume:

{resume_text}

Here is the job description:

{jd}

The following keywords are MISSING from my resume — integrate them:
{missing_list}

These metrics from my original resume MUST appear in the output:
{metrics_str}

Optimize this resume for the job description following the system instructions.
Add all missing keywords to the Skills section and weave them into experience
bullets where the original work supports it. Preserve every original metric.
Return ONLY the optimized resume text."""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=4000,
    )
    return resp.choices[0].message.content


# ---- UI ----
st.set_page_config(page_title="Resume Optimizer", page_icon="📄", layout="wide")
st.title("📄 Resume Optimizer")
st.caption("Powered by Groq + llama-3.3-70b-versatile  •  Single-pass optimization")

RESUME = st.text_area(
    "Paste your resume here",
    height=300,
    placeholder="Copy and paste your resume text here...",
)
jd = st.text_area(
    "Paste Job Description here",
    height=300,
    placeholder="Copy the full job description and paste it here...",
)

# ---- OPTIMIZE BUTTON (single-pass) ----
if st.button("Optimize My Resume", type="primary"):
    if not RESUME.strip():
        st.warning("Please paste your resume first.")
    elif not jd.strip():
        st.warning("Please paste a job description.")
    else:
        st.session_state.last_resume = RESUME
        st.session_state.last_jd = jd

        # Step 1: Score the ORIGINAL resume locally (zero tokens)
        with st.spinner("Analyzing resume against job description..."):
            st.session_state.before_scores = compute_scores(RESUME, jd)

        before = st.session_state.before_scores
        missing = before["missing"]

        # Step 2: Single LLM call — pass JD + resume + missing keywords
        with st.spinner(
            f"Optimizing resume (1 LLM call — {len(missing)} missing keywords identified)..."
        ):
            st.session_state.optimized_resume = optimize_resume(jd, RESUME, missing)

        # Step 3: Score the optimized resume locally (zero tokens)
        with st.spinner("Scoring optimized resume..."):
            st.session_state.after_scores = compute_scores(
                st.session_state.optimized_resume, jd
            )


# ---- SHOW RESULTS ----
if st.session_state.get("optimized_resume"):
    st.markdown("---")

    # Before scores
    st.markdown("### 📊 Before Optimization")
    show_scores(st.session_state.before_scores, "Original Resume vs JD")

    # After scores
    st.markdown("---")
    st.markdown("### 📊 After Optimization")
    show_scores(st.session_state.after_scores, "Optimized Resume vs JD")

    # Score improvement summary
    before = st.session_state.before_scores
    after = st.session_state.after_scores
    kw_delta = after["keyword_score"] - before["keyword_score"]
    sem_delta = round(after["semantic_score"] - before["semantic_score"], 1)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Keyword Improvement", f"{after['keyword_score']}%",
                   delta=f"+{kw_delta}%" if kw_delta > 0 else f"{kw_delta}%")
    with col2:
        st.metric("Semantic Improvement", f"{after['semantic_score']}%",
                   delta=f"+{sem_delta}%" if sem_delta > 0 else f"{sem_delta}%")

    # Optimized resume output
    st.markdown("---")
    st.markdown("### 📝 Optimized Resume")
    st.text_area(
        "Copy your optimized resume below:",
        value=st.session_state.optimized_resume,
        height=400,
        key="optimized_text",
    )
    st.download_button(
        label="Download as .txt",
        data=st.session_state.optimized_resume,
        file_name="optimized_resume.txt",
        mime="text/plain",
    )