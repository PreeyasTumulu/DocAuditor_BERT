"""
Test-corpus preparation for DocAuditor.

Strategy
--------
Real documents give authentic language; synthetic PII gives a ground-truth
answer key. We combine both:

  1. Take REAL source documents (CUAD commercial contracts from SEC EDGAR,
     MTSamples de-identified medical transcriptions).
  2. Inject SYNTHETIC PII at anchor points, recording the exact character
     offset of every injected span as we build the document.
  3. Plant a small number of deliberate logical contradictions.

Because offsets are recorded during assembly rather than searched for
afterwards, the resulting answer key is exact by construction. This lets the
PII scanner be *scored* (precision / recall / F1) rather than merely
demonstrated.

Ethical note
------------
The HR grievance thread is authored synthetically. Real grievance
correspondence is not public for good reason, and the widely-used real email
corpora contain personal data whose subjects never consented to its release.

Outputs (data/test_documents/)
------------------------------
  contract.txt              real CUAD services agreement + injected PII
  medical_report.txt        real MTSamples transcription + injected PHI
  hr_grievance_email.txt    fully synthetic
  ground_truth.json         PII answer key + planted contradictions
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "test_documents"

SEED = 20260721  # fixed so the corpus is reproducible


# --------------------------------------------------------------------------
# Document assembler
# --------------------------------------------------------------------------

@dataclass
class PIISpan:
    start: int
    end: int
    text: str
    label: str

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end,
                "text": self.text, "label": self.label}


@dataclass
class DocumentBuilder:
    """Accumulates text while tracking the offsets of injected PII.

    Using an explicit builder (rather than string.replace + a later search)
    means every recorded offset is correct by construction, even when the same
    value appears more than once in the document.
    """

    parts: list[str] = field(default_factory=list)
    spans: list[PIISpan] = field(default_factory=list)
    _len: int = 0

    def add(self, text: str) -> "DocumentBuilder":
        """Append ordinary, non-sensitive text."""
        self.parts.append(text)
        self._len += len(text)
        return self

    def add_pii(self, value: str, label: str) -> "DocumentBuilder":
        """Append a PII value and record its exact span."""
        self.spans.append(PIISpan(self._len, self._len + len(value), value, label))
        self.parts.append(value)
        self._len += len(value)
        return self

    @property
    def text(self) -> str:
        return "".join(self.parts)


# --------------------------------------------------------------------------
# Synthetic PII pools
# --------------------------------------------------------------------------
# Deliberately fictitious. Names are invented; the phone numbers use reserved
# ranges; the email domains are RFC 2606 reserved names.

GIVEN_NAMES = ["Rohan", "Priya", "Marcus", "Aisha", "Devika", "Thomas", "Neha"]
SURNAMES = ["Kulkarni", "Fernandes", "Whitfield", "Rahman", "Iyer", "Okonkwo"]
CITIES = ["Pune", "Bengaluru", "Singapore", "Rotterdam"]
STREETS = ["Vishrantwadi Road", "Marine Parade Central", "Hosur Main Road"]


def _rng() -> random.Random:
    return random.Random(SEED)


# --------------------------------------------------------------------------
# 1. Contract  (real CUAD text + injected PII + planted contradictions)
# --------------------------------------------------------------------------

CUAD_FILE = ("CUAD_v1/full_contract_txt/Part_I/"
             "ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt")


def build_contract() -> tuple[str, list[PIISpan], list[dict]]:
    path = hf_hub_download("theatticusproject/cuad", CUAD_FILE, repo_type="dataset")
    source = Path(path).read_text(encoding="utf-8", errors="replace")

    b = DocumentBuilder()

    # --- Real contract body ------------------------------------------------
    # SEC filings redact sensitive values as "[ * * * ]". Those markers are the
    # natural anchor points: we re-populate them with synthetic identities,
    # putting fake PII exactly where real PII was removed.
    segments = source.split("[ * * * ]")
    identities = [
        ("Nordvale Systems Pte. Ltd.", "ORGANISATION"),
        ("Rohan Kulkarni", "FULLNAME"),
        ("Priya Fernandes", "FULLNAME"),
    ]
    for i, seg in enumerate(segments):
        b.add(seg)
        if i < len(segments) - 1:
            value, label = identities[i % len(identities)]
            b.add_pii(value, label)

    # --- Planted contradiction #1: payment terms --------------------------
    # The real agreement states its payment window in Section 2. We append a
    # later clause that conflicts with it, so the contradiction detector has a
    # verifiable positive to find.
    b.add("\n\n2.6 Payment Terms. Recipient shall pay each undisputed invoice "
          "within thirty (30) days of receipt.\n")
    contradiction_a_start = b._len

    # --- Notices section: dense, realistic PII ----------------------------
    b.add("\n8. NOTICES\n\nAll notices under this Agreement shall be delivered "
          "to the following representatives:\n\nFor the Provider:\n")
    b.add("    Attention: ").add_pii("Rohan Kulkarni", "FULLNAME")
    b.add(", Director of Operations\n    ")
    b.add_pii("14", "BUILDINGNUM").add(" ").add_pii("Marine Parade Central", "STREET")
    b.add(", ").add_pii("Singapore", "CITY").add(" ").add_pii("449269", "ZIPCODE")
    b.add("\n    Email: ").add_pii("r.kulkarni@nordvale.example.com", "EMAIL")
    b.add("\n    Telephone: ").add_pii("+65 6555 0142", "TELEPHONENUM")
    b.add("\n    Company Tax Reference: ").add_pii("200912345K", "TAXNUM")
    b.add("\n\nFor the Recipient:\n")
    b.add("    Attention: ").add_pii("Priya Fernandes", "FULLNAME")
    b.add(", General Counsel\n    ")
    b.add_pii("221", "BUILDINGNUM").add(" ").add_pii("Vishrantwadi Road", "STREET")
    b.add(", ").add_pii("Pune", "CITY").add(" ").add_pii("411015", "ZIPCODE")
    b.add("\n    Email: ").add_pii("priya.fernandes@telcostar.example.org", "EMAIL")
    b.add("\n    Telephone: ").add_pii("+91 20 5550 8823", "TELEPHONENUM")
    b.add("\n\n")

    # --- Banking details: high-severity PII -------------------------------
    b.add("9. PAYMENT DETAILS\n\nAll sums shall be remitted to the following "
          "account:\n    Account Name: Nordvale Systems Pte. Ltd.\n")
    b.add("    Account Number: ").add_pii("6710044829", "ACCOUNTNUM")
    b.add("\n    Corporate Card on File: ").add_pii("4024 0071 4412 8899", "CREDITCARDNUMBER")
    b.add("\n    Portal Username: ").add_pii("nordvale_billing", "USERNAME")
    b.add("\n\n")

    # --- Planted contradiction #2: governing law ---------------------------
    b.add("10. GOVERNING LAW\n\n10.1 This Agreement shall be governed by the "
          "laws of the Republic of Singapore.\n\n")
    b.add("10.2 Any dispute arising under this Agreement shall be governed "
          "exclusively by the laws of the State of Delaware.\n\n")

    # --- Planted contradiction #3: termination notice ----------------------
    b.add("11. TERMINATION\n\n11.1 Either Party may terminate this Agreement "
          "upon sixty (60) days written notice.\n\n")
    b.add("11.4 Notwithstanding the foregoing, this Agreement may not be "
          "terminated by either Party on less than ninety (90) days written "
          "notice.\n\n")

    # --- Deliberately ambiguous clauses -----------------------------------
    # Natural vagueness already exists in the source ("good faith",
    # "reasonable request"); these add clear-cut cases for the ambiguity meter.
    b.add("12. SERVICE LEVELS\n\n12.1 Provider shall use reasonable efforts to "
          "respond to support requests promptly and shall restore service "
          "within a commercially reasonable period.\n\n")
    b.add("12.2 Material changes to the Services shall be communicated to "
          "Recipient in a timely manner where practicable.\n")

    # Verified against the source text, not assumed. Each entry below was
    # confirmed by locating the conflicting language in the real contract.
    contradictions = [
        {"id": "payment_terms",
         "clause_a": "2.6 Recipient shall pay each undisputed invoice within "
                     "thirty (30) days of receipt",
         "clause_b": "payments pursuant to this Agreement shall be made within "
                     "fifteen (15) days after the date of receipt",
         "origin": "clause_a planted; clause_b is ORIGINAL source text",
         "note": "Conflicting payment windows: 30 days vs 15 days."},
        {"id": "governing_law",
         "clause_a": "10.1 governed by the laws of the Republic of Singapore",
         "clause_b": "10.2 governed exclusively by the laws of the State of Delaware",
         "origin": "both planted",
         "note": "Two mutually exclusive governing-law provisions."},
        {"id": "termination_notice",
         "clause_a": "11.1 terminate upon sixty (60) days written notice",
         "clause_b": "11.4 may not be terminated on less than ninety (90) days "
                     "notice; the source text separately requires 'at least 90 "
                     "(ninety) days prior written notice'",
         "origin": "clause_a planted; clause_b planted AND corroborated by "
                   "original source text",
         "note": "The 60-day termination right is void under the 90-day floor."},
    ]
    _ = contradiction_a_start  # retained for future span-level annotation

    return b.text, b.spans, contradictions


# --------------------------------------------------------------------------
# 2. Medical report  (real MTSamples transcription + injected PHI)
# --------------------------------------------------------------------------

def build_medical() -> tuple[str, list[PIISpan]]:
    import pandas as pd

    path = hf_hub_download(
        "DataFog/medical-transcription-instruct",
        "datafog-medical-transcription-instruct.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(path, nrows=400)

    # Pick a text column heuristically - the schema is not guaranteed stable.
    text_col = next(
        (c for c in ("transcription", "output", "text", "response")
         if c in df.columns),
        df.columns[-1],
    )
    body = next(
        (t for t in df[text_col].dropna().astype(str)
         if 1500 < len(t) < 6000),
        str(df[text_col].dropna().iloc[0]),
    )

    b = DocumentBuilder()
    b.add("CONFIDENTIAL PATIENT RECORD\nMeridian Valley Medical Centre\n")
    b.add("=" * 60 + "\n\n")
    b.add("Patient Name:      ").add_pii("Devika Iyer", "FULLNAME").add("\n")
    b.add("Date of Birth:     ").add_pii("14 March 1979", "DATEOFBIRTH").add("\n")
    b.add("Patient ID:        ").add_pii("MVMC-4471982", "IDCARDNUM").add("\n")
    b.add("Social Security:   ").add_pii("412-88-7390", "SOCIALNUM").add("\n")
    b.add("Address:           ").add_pii("221 Vishrantwadi Road", "STREET")
    b.add(", ").add_pii("Pune", "CITY").add(" ").add_pii("411015", "ZIPCODE").add("\n")
    b.add("Contact Number:    ").add_pii("+91 98220 55017", "TELEPHONENUM").add("\n")
    b.add("Email:             ").add_pii("d.iyer@example.net", "EMAIL").add("\n")
    b.add("Referring Doctor:  Dr. ").add_pii("Marcus Whitfield", "FULLNAME").add("\n")
    b.add("Insurance Policy:  ").add_pii("HLTH-99120445", "ACCOUNTNUM").add("\n")
    b.add("\n" + "=" * 60 + "\n\nCLINICAL TRANSCRIPTION\n\n")
    b.add(body.strip())
    b.add("\n\n" + "=" * 60 + "\n")
    b.add("Record released to ").add_pii("d.iyer@example.net", "EMAIL")
    b.add(" on request. Handling of this record is subject to applicable "
          "data-protection law.\n")

    return b.text, b.spans


# --------------------------------------------------------------------------
# 3. HR grievance email thread  (fully synthetic - see ethical note above)
# --------------------------------------------------------------------------

def build_hr_email() -> tuple[str, list[PIISpan]]:
    b = DocumentBuilder()

    b.add("From: ").add_pii("aisha.rahman@example-corp.com", "EMAIL")
    b.add("\nTo: ").add_pii("hr.grievance@example-corp.com", "EMAIL")
    b.add("\nDate: 2 July 2026\nSubject: Formal grievance - conduct in the "
          "Rotterdam project team\n\n")
    b.add("Dear HR Team,\n\nI am writing to raise a formal grievance regarding "
          "conduct I have experienced while working on the Rotterdam "
          "engagement.\n\nMy details are as follows:\n")
    b.add("    Name:            ").add_pii("Aisha Rahman", "FULLNAME").add("\n")
    b.add("    Employee ID:     ").add_pii("EC-2019-0884", "IDCARDNUM").add("\n")
    b.add("    Mobile:          ").add_pii("+31 6 5550 2277", "TELEPHONENUM").add("\n")
    b.add("    Home Address:    ").add_pii("48", "BUILDINGNUM").add(" ")
    b.add_pii("Weena Zuid", "STREET").add(", ").add_pii("Rotterdam", "CITY")
    b.add(" ").add_pii("3012 NC", "ZIPCODE").add("\n\n")

    # Ambiguous / hedged language - material for the ambiguity meter.
    b.add("On several occasions my line manager, ").add_pii("Thomas Whitfield", "FULLNAME")
    b.add(", made comments in team meetings that I found inappropriate. I "
          "raised this informally some time ago and was told the matter would "
          "be handled appropriately, but as far as I am aware nothing much has "
          "changed since then.\n\n")
    b.add("I would appreciate it if this could be looked into reasonably "
          "quickly. I understand these things take time, but the situation is "
          "becoming difficult to manage.\n\n")
    b.add("You may contact me on my personal email at ")
    b.add_pii("aisha.r.personal@example.net", "EMAIL")
    b.add(" if that is easier.\n\nKind regards,\n")
    b.add_pii("Aisha Rahman", "FULLNAME").add("\n")

    b.add("\n---\n\nFrom: ").add_pii("hr.grievance@example-corp.com", "EMAIL")
    b.add("\nTo: ").add_pii("aisha.rahman@example-corp.com", "EMAIL")
    b.add("\nDate: 3 July 2026\nSubject: RE: Formal grievance\n\n")
    b.add("Dear ").add_pii("Aisha", "FULLNAME")
    b.add(",\n\nThank you for your message. We will review this shortly and "
          "aim to respond within a reasonable timeframe. A case reference has "
          "been assigned: ")
    b.add_pii("GRV-2026-0417", "IDCARDNUM")
    b.add(".\n\nRegards,\n").add_pii("Neha Okonkwo", "FULLNAME")
    b.add("\nHR Business Partner\n")

    return b.text, b.spans


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth: dict = {
        "seed": SEED,
        "note": "PII offsets are exact by construction (recorded during "
                "document assembly, not searched afterwards).",
        "documents": {},
    }

    print("Building contract from CUAD ...")
    contract, c_spans, contradictions = build_contract()
    (OUT_DIR / "contract.txt").write_text(contract, encoding="utf-8")
    ground_truth["documents"]["contract.txt"] = {
        "source": "CUAD (CC BY 4.0) - real SEC EDGAR services agreement",
        "doc_type": "contract",
        "chars": len(contract),
        "pii": [s.as_dict() for s in c_spans],
        "planted_contradictions": contradictions,
    }

    print("Building medical report from MTSamples ...")
    medical, m_spans = build_medical()
    (OUT_DIR / "medical_report.txt").write_text(medical, encoding="utf-8")
    ground_truth["documents"]["medical_report.txt"] = {
        "source": "MTSamples via DataFog mirror (CC0) - real de-identified "
                  "transcription, synthetic PHI header",
        "doc_type": "medical",
        "chars": len(medical),
        "pii": [s.as_dict() for s in m_spans],
        "planted_contradictions": [],
    }

    print("Building HR grievance thread (synthetic) ...")
    hr, h_spans = build_hr_email()
    (OUT_DIR / "hr_grievance_email.txt").write_text(hr, encoding="utf-8")
    ground_truth["documents"]["hr_grievance_email.txt"] = {
        "source": "Synthetic - authored for this project (see ethical note)",
        "doc_type": "hr_email",
        "chars": len(hr),
        "pii": [s.as_dict() for s in h_spans],
        "planted_contradictions": [],
    }

    (OUT_DIR / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8")

    # --- Verification: every recorded offset must match the written file ---
    print("\nVerifying offsets against written files ...")
    ok = True
    for fname, meta in ground_truth["documents"].items():
        text = (OUT_DIR / fname).read_text(encoding="utf-8")
        for span in meta["pii"]:
            actual = text[span["start"]:span["end"]]
            if actual != span["text"]:
                ok = False
                print(f"  MISMATCH in {fname}: expected {span['text']!r}, "
                      f"found {actual!r}")
        print(f"  {fname:26s} {meta['chars']:7,d} chars  "
              f"{len(meta['pii']):3d} PII spans")

    print("\nOFFSET VERIFICATION:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
