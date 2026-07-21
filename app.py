"""
DocAuditor - Streamlit UI.

Run with:   streamlit run app.py

This is the interface layer over the pipeline developed in DocAuditor.ipynb.
The pipeline functions are restated here so the app runs standalone, without
importing from the notebook.

Everything below uses BERT-family encoder models only. Nothing is generative.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

st.set_page_config(page_title="DocAuditor", page_icon="[]", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resolve relative to this file, not the working directory. A relative path
# silently yields an empty sample list whenever the app is launched from
# anywhere other than the project root - which is exactly how it gets run.
DATA = Path(__file__).resolve().parent / "data" / "test_documents"

MODELS = {
    "pii":       "iiiorg/piiranha-v1-detect-personal-information",
    "retriever": "sentence-transformers/all-MiniLM-L6-v2",
    "qa":        "deepset/roberta-base-squad2",
    "mlm":       "bert-base-uncased",
    "nli":       "cross-encoder/nli-deberta-v3-base",
}

ROUTER_LABELS = {
    "contract": "a commercial contract or legal agreement between parties",
    "medical":  "a patient medical record or clinical report",
    "hr_email": "an internal HR or employee grievance email",
}

PII_POLICY = {
    "contract": {"boost": ["ACCOUNTNUM", "CREDITCARDNUMBER", "TAXNUM"],
                 "note": "Commercial terms are expected; banking detail is high risk."},
    "medical":  {"boost": ["SOCIALNUM", "DATEOFBIRTH", "IDCARDNUM", "FULLNAME",
                           "GIVENNAME", "SURNAME"],
                 "note": "Treated as PHI: identifiers combined with clinical content."},
    "hr_email": {"boost": ["FULLNAME", "GIVENNAME", "SURNAME", "EMAIL",
                           "TELEPHONENUM", "STREET"],
                 "note": "Complainant identity is the sensitive asset."},
}

VAGUE_TERMS = ("reasonable", "reasonably", "promptly", "timely", "material",
               "materially", "substantial", "substantially", "appropriate",
               "adequate", "satisfactory", "good faith", "best efforts",
               "commercially reasonable", "from time to time",
               "as soon as possible", "where practicable", "significant",
               "customary", "necessary", "periodically")

REGEX_PII = {
    "EMAIL":            re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    # Lookarounds prevent matching the numeric tail of identifiers such as
    # "EC-2019-0884", which would otherwise be reported as a phone number.
    "TELEPHONENUM":     re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){7,14}\d(?![\w-])"),
    "CREDITCARDNUMBER": re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
    "SOCIALNUM":        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

SECTION_RE = re.compile(r"(?:(?<=^)|(?<=[.\s;:]))(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+(?=[A-Z\"(])")
SENTENCE_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z\"(])")


# --------------------------------------------------------------------------
# Cached model loaders
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading PII model ...")
def get_pii_pipe():
    from transformers import pipeline
    return pipeline("token-classification", model=MODELS["pii"],
                    aggregation_strategy="simple",
                    device=0 if DEVICE == "cuda" else -1)


@st.cache_resource(show_spinner="Loading QA model ...")
def get_qa_model():
    """Loaded directly rather than via pipeline().

    transformers v5 removed the "question-answering" pipeline task, so we
    decode the start/end logits ourselves in extract_span() below.
    """
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["qa"])
    mdl = AutoModelForQuestionAnswering.from_pretrained(MODELS["qa"]).to(DEVICE).eval()
    return tok, mdl


@st.cache_resource(show_spinner="Loading sentence encoder ...")
def get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODELS["retriever"], device=DEVICE)


@st.cache_resource(show_spinner="Loading NLI model ...")
def get_nli():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODELS["nli"], device=DEVICE)


@st.cache_resource(show_spinner="Loading masked LM ...")
def get_mlm():
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["mlm"])
    mdl = AutoModelForMaskedLM.from_pretrained(MODELS["mlm"]).to(DEVICE).eval()
    return tok, mdl


def softmax(a):
    a = np.atleast_2d(np.asarray(a, dtype=float))
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def is_section_marker(m, text):
    """Reject numbers that merely look like section markers.

    "at least 90 (ninety) days' prior written notice" was being split at "90",
    inventing a clause whose reference was literally "90". A bare number
    followed by "(" is a quantity, not a heading; dotted numbers always are.
    """
    nxt = text[m.end():m.end() + 1]
    return ("." in m.group(1)) or nxt.isupper() or nxt == '"'


@st.cache_data(show_spinner=False)
def segment_clauses(text, min_chars=60, max_chars=1200):
    cuts = [(0, "")] + [(m.start(), m.group(1)) for m in SECTION_RE.finditer(text)
                        if is_section_marker(m, text)]
    seen, ordered = set(), []
    for pos, lbl in sorted(cuts):
        if pos not in seen:
            seen.add(pos)
            ordered.append((pos, lbl))

    raw = [(p, ordered[i + 1][0] if i + 1 < len(ordered) else len(text), l)
           for i, (p, l) in enumerate(ordered)]
    raw = [(s, e, l) for s, e, l in raw if e > s]

    pieces = []
    for s, e, lbl in raw:
        block = text[s:e]
        if len(block) <= max_chars:
            pieces.append((s, e, lbl))
            continue
        cur = s
        for part in SENTENCE_RE.split(block):
            if not part:
                continue
            i = text.find(part, cur, e)
            i = cur if i == -1 else i
            pieces.append((i, i + len(part), lbl))
            cur, lbl = i + len(part), ""

    merged = []
    for s, e, lbl in pieces:
        if not text[s:e].strip():
            continue
        if merged and (e - s) < min_chars:
            merged[-1][1] = e
        else:
            merged.append([s, e, lbl])

    out = []
    for s, e, lbl in merged:
        body = text[s:e].strip()
        if len(body) < 20:
            continue
        i = text.find(body, s, e + 1)
        i = s if i == -1 else i
        out.append({"index": len(out), "text": body, "start": i,
                    "end": i + len(body), "ref": lbl or f"#{len(out)}"})
    return out


def representative_excerpt(text, max_chars=2000):
    if len(text) <= max_chars:
        return text
    head = text[:max_chars // 2]
    mid0 = max(0, len(text) // 2 - max_chars // 4)
    return head + "\n...\n" + text[mid0:mid0 + max_chars // 2]


def route(text):
    ce = get_nli()
    keys = list(ROUTER_LABELS)
    pairs = [(representative_excerpt(text), f"This document is {ROUTER_LABELS[k]}.")
             for k in keys]
    ent = softmax(np.asarray(ce.predict(pairs, show_progress_bar=False)))[:, 1]
    scores = {k: float(v) / float(ent.sum()) for k, v in zip(keys, ent)}
    best = max(scores, key=scores.get)
    return {"doc_type": best, "confidence": scores[best], "scores": scores,
            "policy": PII_POLICY[best]}


def luhn_ok(s):
    d = [int(c) for c in re.sub(r"\D", "", s)]
    if len(d) < 13:
        return False
    tot, par = 0, len(d) % 2
    for i, n in enumerate(d):
        if i % 2 == par:
            n *= 2
            if n > 9:
                n -= 9
        tot += n
    return tot % 10 == 0


def windows(text, size=1800, overlap=300):
    if len(text) <= size:
        yield 0, text
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            nudge = text.rfind(" ", start + size - overlap, end)
            if nudge > start:
                end = nudge
        yield start, text[start:end]
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def dedupe(items):
    items = sorted(items, key=lambda f: (f["start"], -(f["end"] - f["start"]), -f["score"]))
    kept = []
    for f in items:
        clash = next((k for k in kept
                      if f["start"] < k["end"] and k["start"] < f["end"]), None)
        if clash is None:
            kept.append(f)
        elif (f["end"] - f["start"]) > (clash["end"] - clash["start"]):
            kept[kept.index(clash)] = f
        elif f["source"] != clash["source"]:
            clash["source"] = "model+regex"
            clash["score"] = max(clash["score"], f["score"])
    return sorted(kept, key=lambda f: f["start"])


def scan_pii(text, policy=None, min_score=0.5):
    pipe = get_pii_pipe()
    found = []
    for off, win in windows(text):
        for e in pipe(win):
            if float(e.get("score", 0)) < min_score:
                continue
            lbl = str(e.get("entity_group") or e.get("entity") or "")
            lbl = lbl.replace("B-", "").replace("I-", "").upper()
            s, en = int(e["start"]) + off, int(e["end"]) + off
            # Trim whitespace swept into the span so redaction does not eat
            # newlines and mangle the layout of the redacted document.
            while s < en and text[s].isspace():
                s += 1
            while en > s and text[en - 1].isspace():
                en -= 1
            if en > s:
                found.append({"start": s, "end": en, "text": text[s:en],
                              "label": lbl, "score": float(e["score"]),
                              "source": "model"})
    for lbl, pat in REGEX_PII.items():
        for m in pat.finditer(text):
            v = m.group().strip()
            if lbl == "CREDITCARDNUMBER" and not luhn_ok(v):
                continue
            if lbl == "TELEPHONENUM" and len(re.sub(r"\D", "", v)) < 8:
                continue
            found.append({"start": m.start(), "end": m.start() + len(v), "text": v,
                          "label": lbl, "score": 1.0, "source": "regex"})
    out = dedupe(found)
    boost = set((policy or {}).get("boost", []))
    for f in out:
        f["severity"] = "HIGH" if f["label"] in boost else "normal"
    return out


def redact(text, findings):
    slots, out, cur = {}, [], 0
    for f in sorted(findings, key=lambda x: x["start"]):
        if f["start"] < cur:
            continue
        s = slots.setdefault(f["label"], {})
        if f["text"] not in s:
            s[f["text"]] = f"[{f['label']}_{len(s) + 1}]"
        out.append(text[cur:f["start"]])
        out.append(s[f["text"]])
        cur = f["end"]
    out.append(text[cur:])
    return "".join(out)


def extract_span(question, context, max_answer_tokens=30):
    """Decode an answer span from the QA head's start/end logits.

    The output is a pair of indices into the input - which is precisely why
    this model cannot invent an answer that is not in the document.
    """
    tok, mdl = get_qa_model()
    enc = tok(question, context, return_tensors="pt", truncation="only_second",
              max_length=384, return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    seq_ids = enc.sequence_ids(0)
    inputs = {k: v.to(DEVICE) for k, v in enc.items() if k != "offset_mapping"}

    with torch.no_grad():
        out = mdl(**inputs)
    start_p = torch.softmax(out.start_logits[0], -1).cpu().numpy()
    end_p = torch.softmax(out.end_logits[0], -1).cpu().numpy()

    ctx = [i for i, s in enumerate(seq_ids) if s == 1]
    if not ctx:
        return "", 0.0
    best_score, best_span = 0.0, None
    for i in ctx:
        for j in range(i, min(i + max_answer_tokens, ctx[-1] + 1)):
            sc = float(start_p[i] * end_p[j])
            if sc > best_score:
                best_score, best_span = sc, (i, j)
    if best_span is None:
        return "", 0.0
    i, j = best_span
    return context[offsets[i][0]:offsets[j][1]].strip(), best_score


def ask(clauses, emb, question, top_k=5, min_conf=0.15):
    enc = get_encoder()
    q = enc.encode([question], convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    order = np.argsort(-(emb @ q[0]))[:top_k]
    best = None
    for i in order:
        c = clauses[int(i)]
        ans, score = extract_span(question, c["text"])
        if best is None or score > best["score"]:
            best = {"answer": ans, "score": score,
                    "ref": c["ref"], "context": c["text"]}
    if not best or best["score"] < min_conf or not best["answer"]:
        return {"answer": "Not stated in document.",
                "score": best["score"] if best else 0.0,
                "ref": "-", "context": "", "abstained": True}
    best["abstained"] = False
    return best


def mlm_entropy(sentence, term):
    tok, mdl = get_mlm()
    masked = term_pattern(term).sub(tok.mask_token, sentence, count=1)
    enc = tok(masked, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        return 0.0, []
    with torch.no_grad():
        logits = mdl(**enc).logits[0, pos[0]]
    p = torch.softmax(logits, dim=-1)
    H = float(-(p * torch.log(p + 1e-12)).sum())
    top = torch.topk(p, 5)
    preds = [(tok.decode([i]).strip(), round(float(v), 3))
             for i, v in zip(top.indices.tolist(), top.values.tolist())]
    return H, preds


def both_readings(sentence):
    ce = get_nli()
    strict = "This imposes a strict, precisely defined obligation."
    loose = "This allows flexibility and leaves the requirement open to interpretation."
    ent = softmax(np.asarray(ce.predict([(sentence, strict), (sentence, loose)],
                                        show_progress_bar=False)))[:, 1]
    return bool(ent[0] > 0.35 and ent[1] > 0.35)


def term_pattern(term):
    """Whole-word matcher for a vague term.

    Substring matching is wrong here: "material" as a legal standard means
    *significant*, while "materials" is an ordinary noun meaning physical
    goods. Matching the former inside the latter yields a confident finding
    about a clause containing no vague language at all.
    """
    return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)


def sentence_with(text, term):
    pat = term_pattern(term)
    for part in re.split(r"(?<=[.;])\s+", text):
        if pat.search(part):
            return part.strip()
    return text[:400].strip()


def ambiguity_report(clauses, top_n=12, entropy_floor=3.0):
    out = []
    for c in clauses:
        for t in VAGUE_TERMS:
            m = term_pattern(t).search(c["text"])
            if not m:
                continue
            s = sentence_with(c["text"], t)
            actual = m.group()
            H, preds = mlm_entropy(s, actual)
            if H > 0:
                out.append({"clause": c["ref"], "term": actual, "entropy": H,
                            "sentence": s, "top_predictions": preds})
    out.sort(key=lambda r: -r["entropy"])
    out = out[:top_n]
    for r in out:
        dbl = both_readings(r["sentence"])
        r["both_readings"] = dbl
        r["verdict"] = ("AMBIGUOUS - strict AND loose reading" if dbl and r["entropy"] >= entropy_floor
                        else "VAGUE - context does not constrain" if r["entropy"] >= entropy_floor
                        else "acceptable in context")
    return out


def detect_contradictions(clauses, min_sim=0.45, max_sim=0.995,
                          min_score=0.95, max_pairs=400, max_clause_chars=250):
    """Two-stage contradiction detection.

    max_clause_chars is the main precision lever: NLI models are trained on
    short everyday sentence pairs and degrade badly on long, heavily-qualified
    legal prose, producing confident false contradictions between clauses that
    are merely scoped to different parties. Without the cap this returns ~116
    findings on the sample contract; with it, ~10.
    """
    emb = get_encoder().encode([c["text"] for c in clauses], convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
    sim = emb @ emb.T
    n = len(clauses)
    cands = [(i, j, float(sim[i, j])) for i in range(n) for j in range(i + 1, n)
             if min_sim <= sim[i, j] <= max_sim
             and max(len(clauses[i]["text"]), len(clauses[j]["text"])) <= max_clause_chars]
    cands.sort(key=lambda x: -x[2])
    cands = cands[:max_pairs]

    stats = {"clauses": n, "possible_pairs": n * (n - 1) // 2,
             "pairs_judged": len(cands)}
    stats["reduction"] = (1 - stats["pairs_judged"] / stats["possible_pairs"]
                          if stats["possible_pairs"] else 0)
    if not cands:
        return [], stats

    ce = get_nli()
    pairs = []
    for i, j, _ in cands:
        pairs += [(clauses[i]["text"], clauses[j]["text"]),
                  (clauses[j]["text"], clauses[i]["text"])]
    probs = softmax(np.asarray(ce.predict(pairs, show_progress_bar=False)))[:, 0]

    res = []
    for k, (i, j, s) in enumerate(cands):
        score = max(float(probs[2 * k]), float(probs[2 * k + 1]))
        if score >= min_score:
            res.append({"a_ref": clauses[i]["ref"], "b_ref": clauses[j]["ref"],
                        "similarity": s, "contradiction": score,
                        "clause_a": clauses[i]["text"], "clause_b": clauses[j]["text"]})
    res.sort(key=lambda r: -r["contradiction"])
    return res, stats


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("DocAuditor")
st.caption("Encoder-only document intelligence for legal, HR and compliance review. "
           "Five BERT-family heads. No generative model.")

with st.sidebar:
    st.header("Document")
    samples = sorted(p.name for p in DATA.glob("*.txt")) if DATA.exists() else []
    choice = st.selectbox("Sample document", ["-- upload my own --"] + samples)
    uploaded = st.file_uploader("Upload a .txt document", type=["txt"])

    st.divider()
    st.caption(f"Device: **{DEVICE}**")
    if DEVICE == "cuda":
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    st.caption("Models load on first use.")

text = None
if uploaded is not None:
    text = uploaded.read().decode("utf-8", errors="replace")
elif choice and choice != "-- upload my own --":
    text = (DATA / choice).read_text(encoding="utf-8")

if not text:
    st.info("Select a sample document or upload a .txt file to begin.")
    st.stop()

clauses = segment_clauses(text)

c1, c2, c3 = st.columns(3)
c1.metric("Characters", f"{len(text):,}")
c2.metric("Clauses", len(clauses))
c3.metric("Approx. BERT tokens", f"{int(len(text) / 4.6):,}",
          help="BERT reads 512 at a time - hence retrieval and chunking.")

tabs = st.tabs(["Router", "PII Scanner", "Ask the Document",
                "Ambiguity Meter", "Contradictions", "Document"])

# --- Router -------------------------------------------------------------
with tabs[0]:
    st.subheader("Document Auto-Router")
    st.caption("Zero-shot NLI classification. The verdict selects the PII policy "
               "used by the scanner, so routing changes downstream behaviour.")
    if st.button("Classify document", key="btn_route"):
        r = route(text)
        st.session_state["route"] = r
    r = st.session_state.get("route")
    if r:
        st.success(f"**{r['doc_type']}**  ({r['confidence']:.0%} confidence)")
        st.progress(min(1.0, r["confidence"]))
        st.dataframe(pd.DataFrame(
            [{"type": k, "score": v} for k, v in sorted(r["scores"].items(),
                                                        key=lambda x: -x[1])]),
            hide_index=True, use_container_width=True)
        st.info(f"**Policy applied:** {r['policy']['note']}\n\n"
                f"Escalated labels: `{', '.join(r['policy']['boost'])}`")

# --- PII ----------------------------------------------------------------
with tabs[1]:
    st.subheader("PII Scanner")
    st.caption("Token classification with an overlapping sliding window, plus a "
               "deterministic regex layer for structurally regular identifiers.")
    if st.button("Scan for PII", key="btn_pii"):
        policy = (st.session_state.get("route") or {}).get("policy")
        st.session_state["pii"] = scan_pii(text, policy)
    findings = st.session_state.get("pii")
    if findings:
        high = sum(1 for f in findings if f.get("severity") == "HIGH")
        a, b, c = st.columns(3)
        a.metric("PII spans", len(findings))
        b.metric("High severity", high)
        c.metric("Found by regex only",
                 sum(1 for f in findings if f["source"] == "regex"))
        df = pd.DataFrame(findings)[["text", "label", "score", "source", "severity"]]
        st.dataframe(df, hide_index=True, use_container_width=True)
        with st.expander("Redacted document"):
            st.text(redact(text, findings)[:6000])
        st.download_button("Download redacted document",
                           redact(text, findings), "redacted.txt")

# --- QA -----------------------------------------------------------------
with tabs[2]:
    st.subheader("Ask the Document")
    st.caption("Retrieval with a bi-encoder, then extractive span QA. Answers are "
               "copied verbatim from the text - never composed.")
    q = st.text_input("Question",
                      "How many days notice is required to terminate the agreement?")
    if st.button("Ask", key="btn_qa") and q.strip():
        emb = get_encoder().encode([c["text"] for c in clauses],
                                   convert_to_numpy=True,
                                   normalize_embeddings=True,
                                   show_progress_bar=False)
        st.session_state["qa"] = ask(clauses, emb, q)
    a = st.session_state.get("qa")
    if a:
        if a["abstained"]:
            st.warning("**Not stated in document.**\n\nNo span cleared the confidence "
                       "floor. Abstaining is deliberate: a compliance tool should not "
                       "guess.")
        else:
            st.success(f"**{a['answer']}**")
            st.caption(f"Confidence {a['score']:.2f} - from clause {a['ref']}")
            with st.expander("Source clause"):
                st.write(a["context"])

# --- Ambiguity ----------------------------------------------------------
with tabs[3]:
    st.subheader("Ambiguity Meter")
    st.caption("Masked-language-model entropy plus NLI double-entailment. High "
               "entropy means the context does not pin the term down; entailing "
               "both a strict and a loose reading means genuine ambiguity.")
    own = st.text_area("Test your own sentence (optional)", "")
    if st.button("Analyse", key="btn_amb"):
        if own.strip():
            hits = [t for t in VAGUE_TERMS if term_pattern(t).search(own)]
            if not hits:
                st.warning("No vague terms detected in that sentence. Try wording "
                           "such as 'reasonable efforts' or 'promptly'.")
            else:
                rows = []
                for t in hits:
                    m = term_pattern(t).search(own)
                    H, preds = mlm_entropy(own, m.group())
                    rows.append({"term": m.group(), "entropy": H,
                                 "both_readings": both_readings(own),
                                 "top_predictions": preds})
                st.session_state["amb"] = rows
        else:
            st.session_state["amb"] = ambiguity_report(clauses)
    rows = st.session_state.get("amb")
    if rows:
        # top_predictions is a list of (token, probability) tuples. Streamlit
        # serialises dataframes through Arrow, which cannot type a column of
        # mixed tuples and raises ArrowTypeError - so flatten it to a string.
        def _fmt(r):
            out = {k: v for k, v in r.items() if k != "sentence"}
            if isinstance(out.get("top_predictions"), (list, tuple)):
                out["top_predictions"] = ", ".join(
                    f"{w} ({p})" for w, p in out["top_predictions"])
            if isinstance(out.get("entropy"), float):
                out["entropy"] = round(out["entropy"], 2)
            return out

        st.dataframe(pd.DataFrame([_fmt(r) for r in rows]),
                     hide_index=True, use_container_width=True)
        if rows and "sentence" in rows[0]:
            st.markdown("**Most ambiguous clause**")
            st.info(rows[0]["sentence"])
            st.caption(f"Entropy {rows[0]['entropy']:.2f} nats - "
                       f"BERT's top predictions for the masked slot: "
                       + ", ".join(f"`{w}` ({p})" for w, p in rows[0]["top_predictions"]))

# --- Contradictions -----------------------------------------------------
with tabs[4]:
    st.subheader("Cross-Clause Contradiction Detector")
    st.caption("Two-stage: a bi-encoder filters candidate pairs by topic, then the "
               "NLI cross-encoder judges only the survivors, in both directions.")
    col1, col2, col3 = st.columns(3)
    min_sim = col1.slider("Similarity floor (candidate filter)", 0.20, 0.90, 0.45, 0.05)
    min_score = col2.slider("Contradiction threshold", 0.50, 0.99, 0.95, 0.01)
    max_len = col3.slider("Max clause length (chars)", 150, 1500, 250, 50,
                          help="NLI degrades on long legal prose. Raising this "
                               "finds more but returns far more false positives.")
    if st.button("Detect contradictions", key="btn_contra"):
        res, stats = detect_contradictions(clauses, min_sim=min_sim,
                                           min_score=min_score,
                                           max_clause_chars=max_len)
        st.session_state["contra"] = (res, stats)
    got = st.session_state.get("contra")
    if got:
        res, stats = got
        a, b, c = st.columns(3)
        a.metric("Possible pairs", f"{stats['possible_pairs']:,}")
        b.metric("Pairs judged", f"{stats['pairs_judged']:,}",
                 delta=f"-{stats['reduction']:.0%} filtered")
        c.metric("Contradictions", len(res))
        if res:
            st.dataframe(pd.DataFrame(res)[["a_ref", "b_ref", "similarity",
                                            "contradiction"]],
                         hide_index=True, use_container_width=True)
            for r in res[:5]:
                with st.expander(f"[{r['a_ref']}] vs [{r['b_ref']}]  "
                                 f"- contradiction {r['contradiction']:.2f}"):
                    st.markdown("**A**"); st.write(r["clause_a"])
                    st.markdown("**B**"); st.write(r["clause_b"])
        else:
            st.info("No contradictions above the threshold. Try lowering it.")

# --- Raw document -------------------------------------------------------
with tabs[5]:
    st.subheader("Document")
    st.text(text[:20000])
    if len(text) > 20000:
        st.caption(f"... truncated for display ({len(text):,} chars total)")
