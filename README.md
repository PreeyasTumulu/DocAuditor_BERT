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

Note that the CUAD source contains essentially no PII of its own: SEC filings
are pre-redacted, with sensitive values replaced by `[ * * * ]`. Those markers
are used as the injection anchors, placing synthetic PII exactly where real
PII was removed.

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

## Deliverables

- [ ] Pipeline architecture documentation (`.pdf`)
- [ ] Documented working pipeline (`.ipynb`)
- [ ] Test documents
- [ ] Streamlit application
