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

Documents flow through a shared preprocessing stage (parse -> segment into
clauses -> encode), then fan out to independent task heads:

```
                       +--> [1] PII Scanner        (token classification)
                       |
  document --> parse --+--> [2] Document Q&A       (retrieval + extractive QA)
              segment  |
              encode   +--> [3] Ambiguity Meter    (MLM entropy + NLI)
                       |
                       +--> [4] TBD
                       |
                       +--> [5] TBD
```

## Feature status

| # | Feature | Model | Head | Status |
|---|---------|-------|------|--------|
| 1 | PII Scanner | `iiiorg/piiranha-v1-detect-personal-information` | Token classification | Planned |
| 2 | Document Q&A | `all-MiniLM-L6-v2` + `deepset/roberta-base-squad2` | Bi-encoder + span QA | Planned |
| 3 | Ambiguity Meter | `bert-base-uncased` + `cross-encoder/nli-deberta-v3-base` | MLM entropy + NLI | Planned |
| 4 | *(to be selected)* | — | — | Not chosen |
| 5 | *(to be selected)* | — | — | Not chosen |

Features 1-3 are required by the assignment brief. Features 4-5 are the two
free-choice features and have not been selected yet.

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
