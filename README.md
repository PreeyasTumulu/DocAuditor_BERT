# DocAuditor

**Six heads, one encoder, zero hallucinations.**

A document-intelligence assistant for legal, HR, and compliance teams, built
**entirely from BERT-family encoder models**. No generative LLM is used at any
point in the pipeline.

## Thesis

Every feature in this application is a different *head* on the same family of
bidirectional encoders. The result is a **glass-box auditor**: every output
traces back to a concrete span, a probability, or a cosine distance. Nothing is
generated, so nothing can be hallucinated — which is precisely why a compliance
team could deploy it.

## Architecture

A document is parsed, segmented into clauses and encoded once. The resulting
clause set then fans out to five independent task heads:

```
                              +--> [1] PII Scanner       token classification
                              |
  document                    +--> [2] Document Q&A      retrieval + span QA
     |                        |
     v                        +--> [3] Ambiguity Meter   MLM entropy + NLI
  parse -> segment -> encode -+
     |                        +--> [4] Contradiction     embed-filter + NLI
     v                        |        Detector
  [5] Auto-Router ------------+
   (routes to the
    correct policy)
```

The router runs first and selects the PII policy and clause conventions used
by the remaining heads (PHI rules for medical records, clause rules for
contracts).

### The governing constraint

The prepared contract is **28,009 characters, roughly 6,100 word-piece tokens
-- about 12x BERT's 512-token limit**. Every design decision below follows from
that single fact:

- Q&A cannot read the document, so it **retrieves** candidate passages first.
- Contradiction detection cannot compare everything to everything, so it
  **filters candidate pairs by embedding similarity** before invoking the
  expensive cross-encoder.
- The router cannot see the whole document, so it classifies from a
  **representative excerpt**.

## Features

| # | Feature | Model | Head |
|---|---------|-------|------|
| 1 | PII Scanner | `iiiorg/piiranha-v1-detect-personal-information` | Token classification |
| 2 | Document Q&A | `all-MiniLM-L6-v2` + `deepset/roberta-base-squad2` | Bi-encoder + span QA |
| 3 | Ambiguity Meter | `bert-base-uncased` + `cross-encoder/nli-deberta-v3-base` | MLM entropy + NLI |
| 4 | Cross-Clause Contradiction Detector | `all-MiniLM-L6-v2` + `cross-encoder/nli-deberta-v3-base` | Bi-encoder filter + NLI |
| 5 | Document Auto-Router | `cross-encoder/nli-deberta-v3-base` | Zero-shot classification |

Features 1-3 are required by the brief; 4 and 5 are the free-choice features.

**Why these two.** The contradiction detector uses the NLI model's *native*
`contradiction` output rather than repurposing a model for an unintended task,
and it is a genuine compliance workflow currently done by hand. The router
turns the three-document test corpus into a designed capability rather than
three separate demos. Both reuse models already loaded for feature 3, so the
marginal cost is close to zero.

### Known limitation

Extractive QA can only return spans that **literally appear** in the document.
Asked "what are the risks here?", it will fail, because that answer is written
nowhere. This is an inherent property of encoder-only architecture, not a bug.
It is mitigated with a confidence floor that returns *"not stated in
document"* rather than a low-confidence guess -- an honest abstention, which
for a compliance tool is the correct behaviour.

## Model family note

All models used are bidirectional encoders in the BERT family — RoBERTa,
DeBERTa-v3, MiniLM and MPNet are all BERT derivatives (better pretraining,
disentangled attention, or distillation respectively), not new architectures.
Model IDs are centralised in a single config dict so the entire stack can be
swapped to vanilla `bert-base-*` checkpoints if required.

## Test data provenance

| Document | Source | Licence |
|----------|--------|---------|
| Commercial contract (primary) | [CUAD](https://www.atticusprojectai.org/cuad) — real SEC EDGAR contracts | CC BY 4.0 |
| Medical report | [MTSamples](https://www.mtsamples.com/) — de-identified transcriptions | CC0 |
| HR grievance email | Synthetic (authored for this project) | — |

Real documents provide authentic language; synthetic PII is then injected at
known offsets so the PII scanner can be scored against a ground-truth answer
key rather than merely demonstrated.

The HR thread is deliberately synthetic: real grievance correspondence is not
public for good reason, and available real-email corpora contain un-consented
personal data.

### Prepared corpus

Regenerate with `python scripts/prepare_data.py` (fixed seed, reproducible).

| Document | Chars | PII spans | Contradictions |
|----------|------:|----------:|---------------:|
| `contract.txt` | 28,009 | 25 | 3 |
| `medical_report.txt` | 3,149 | 12 | 0 |
| `hr_grievance_email.txt` | 1,369 | 17 | 0 |
| **Total** | **32,527** | **54** | **3** |

54 PII spans across 15 label types. Offsets are exact **by construction** --
recorded while the document is assembled, not searched for afterwards -- and
the prep script asserts every offset against the written file before exiting,
so a broken answer key fails loudly rather than silently.

The corpus was built assuming the CUAD source contained no PII of its own,
since SEC filings redact sensitive values as `[ * * * ]`. Those markers were
used as the injection anchors.

**That assumption was wrong, and the scanner disproved it.** The redactions
cover commercially sensitive commercial terms, not personal data: real names,
e-mail addresses, telephone numbers and street addresses survive untouched in
the notice blocks of the published filing. What the first evaluation scored as
false positives were all correct detections, and the answer key had to be
extended after manually verifying each one.

A tool built to find personal data in documents found personal data that a
regulatory filing process had missed — on real, public data. That is the
compliance use case, demonstrated by accident.

Note therefore that `data/test_documents/contract.txt` contains real (though
publicly filed) personal data, inherited from the SEC source.

Two of the three planted contradictions were **verified against the source
text** rather than assumed. The original agreement independently specifies a
15-day payment window and a 90-day termination notice, so the planted 30-day
and 60-day clauses conflict with real contract language, not merely with each
other.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
./.venv/Scripts/python.exe -m pip install transformers sentence-transformers streamlit pandas scikit-learn seqeval pymupdf accelerate
```

Verified on Python 3.12 with torch 2.5.1+cu121, transformers 5.14.1,
CUDA enabled on a 4 GB RTX 3050.

## Results

Measured on the prepared corpus; every figure is reproducible by running the
notebook top to bottom.

**PII scanner** (span-overlap scoring, adjudicated key)

| Document | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `contract.txt` | 32 | 7 | 5 | 82.1% | 86.5% | 84.2% |
| `medical_report.txt` | 9 | 7 | 3 | 56.3% | 75.0% | 64.3% |
| `hr_grievance_email.txt` | 13 | 4 | 4 | 76.5% | 76.5% | 76.5% |
| **Overall** | **54** | **18** | **12** | **75.0%** | **81.8%** | **78.3%** |

**Auto-router** — all three documents classified correctly
(contract 95%, HR email 99%, medical 65%).

**Contradiction detector** — 67 clauses, 2,211 possible pairs, **97% filtered
out** by the bi-encoder before the cross-encoder runs; 9 findings at the
default setting, including the planted governing-law conflict.

### Honest limitations

- **Contradiction precision is weak.** 9 findings contain roughly one true
  defect. The shortlist is reviewable, but this is a triage aid, not an
  oracle. The cause is domain mismatch: NLI is trained on short everyday
  sentences, not hedged legal prose.
- **One planted defect is missed** at the default setting. Its source language
  sits in a 1,145-character clause, above the length cap that makes the output
  reviewable. Raising the cap recovers it and returns ~102 findings instead
  of 9. The notebook measures this trade-off rather than hiding it.
- **Extractive QA cannot answer "summarise the risks"** — only spans that
  literally exist. It abstains below a confidence floor instead of guessing.
- **Requiring symmetric contradiction was tried and rejected.** It cut false
  positives but scored every true defect at 0.007–0.067; these contradictions
  are strongly asymmetric in practice.

## Files

```
DocAuditor.ipynb                    the full pipeline, with saved outputs
DocAuditor_Architecture.pdf         architecture documentation
app.py                              Streamlit UI (streamlit run app.py)
requirements.txt                    pinned, verified versions
data/test_documents/                3 documents + ground_truth.json
scripts/prepare_data.py             corpus preparation, fixed seed
scripts/build_architecture_pdf.py   regenerates the PDF
scripts/make_submission.py          builds the submission zip
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Streamlit app
.venv/Scripts/python.exe -m streamlit run app.py

# Re-execute the notebook (register the venv as a kernel first, otherwise
# Jupyter silently uses the system Python and every import fails)
.venv/Scripts/python.exe -m ipykernel install --user --name docauditor
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=docauditor DocAuditor.ipynb
```

**Input formats:** TXT, PDF and DOCX. DOCX tables are flattened row by row,
since contract schedules and HR forms keep much of their sensitive detail in
tables. Scanned PDFs are rejected with an explanatory message — this pipeline
reads text, not images, and OCR is out of scope.

The `.venv` directory is intentionally not committed and not shipped in the
submission zip: it is several gigabytes of platform-specific binaries that
would not run on another machine anyway. `requirements.txt` reproduces it.

`app.py` restates the pipeline functions rather than importing them, so each
file runs standalone. The duplication is deliberate but real: a threshold
changed in one must be changed in the other.

## Deliverables

- [ ] Pipeline architecture documentation (`.pdf`)
- [x] Documented working pipeline (`.ipynb`) — 36 cells, executes end to end
- [x] Test documents
- [x] Streamlit application
