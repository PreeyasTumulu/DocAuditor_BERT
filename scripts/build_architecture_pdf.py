"""
Generates DocAuditor_Architecture.pdf - the pipeline architecture document.

Run:  python scripts/build_architecture_pdf.py

Note on characters: ReportLab's built-in Helvetica uses WinAnsi encoding, which
has no arrow or check glyphs. Use "->" and "OK" rather than the Unicode forms,
otherwise they render as solid black boxes.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "DocAuditor_Architecture.pdf"

INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#1f4e79")
MUTED = colors.HexColor("#5a6672")
RULE = colors.HexColor("#c8d0d8")
BOXBG = colors.HexColor("#eef3f8")
WARNBG = colors.HexColor("#fdf3e7")

# --------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------
ss = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                    fontSize=16, textColor=ACCENT, spaceBefore=16, spaceAfter=8,
                    leading=20)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=11.5, textColor=INK, spaceBefore=12, spaceAfter=5,
                    leading=15)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontName="Helvetica",
                      fontSize=9.5, leading=14, textColor=INK,
                      alignment=TA_JUSTIFY, spaceAfter=7)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.5, leading=12,
                       textColor=MUTED)
MONO = ParagraphStyle("MONO", parent=BODY, fontName="Courier", fontSize=8.5,
                      leading=12, alignment=0)
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], fontName="Helvetica-Bold",
                       fontSize=26, textColor=ACCENT, spaceAfter=4, leading=30)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontName="Helvetica-Oblique",
                     fontSize=12, textColor=MUTED, alignment=TA_CENTER,
                     spaceAfter=18)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.5, leading=11,
                      alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")


def p(text, style=BODY):
    return Paragraph(text, style)


def table(data, widths, header=True, align_right=()):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                 ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                 ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                 ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                  [colors.white, colors.HexColor("#f6f8fa")])]
    for c in align_right:
        cmds.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(cmds))
    return t


def callout(title, body, bg=BOXBG):
    inner = [[p(f"<b>{title}</b>", CELL)], [p(body, CELL)]]
    t = Table(inner, colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


# --------------------------------------------------------------------------
# architecture diagram
# --------------------------------------------------------------------------

def architecture_diagram():
    W, H = 465, 250
    d = Drawing(W, H)

    def box(x, y, w, h, label, sub="", fill=BOXBG, bold=True):
        d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=ACCENT,
                   strokeWidth=0.7, rx=3, ry=3))
        d.add(String(x + w / 2, y + h - (13 if sub else h / 2 + 3), label,
                     fontName="Helvetica-Bold" if bold else "Helvetica",
                     fontSize=8, fillColor=INK, textAnchor="middle"))
        if sub:
            d.add(String(x + w / 2, y + h - 24, sub, fontName="Helvetica",
                         fontSize=6.5, fillColor=MUTED, textAnchor="middle"))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=0.8))
        ang = 4
        d.add(Polygon([x2, y2, x2 - 6, y2 + ang, x2 - 6, y2 - ang],
                      fillColor=MUTED, strokeColor=MUTED))

    # left column: ingestion
    box(0, 178, 92, 34, "DOCUMENT", ".txt upload", fill=colors.white)
    box(0, 130, 92, 34, "PARSE", "normalise text", fill=colors.white)
    box(0, 82, 92, 34, "SEGMENT", "67 clauses, offsets", fill=colors.white)
    box(0, 34, 92, 34, "ENCODE", "MiniLM bi-encoder", fill=colors.white)
    for y in (178, 130, 82):
        d.add(Line(46, y, 46, y - 14, strokeColor=MUTED, strokeWidth=0.8))
        d.add(Polygon([46, y - 14, 43, y - 8, 49, y - 8],
                      fillColor=MUTED, strokeColor=MUTED))

    # router gate - fed from ENCODE (the bottom of the ingestion column),
    # routed as an elbow so the connector cannot be misread as leaving PARSE
    box(120, 130, 96, 40, "[5] AUTO-ROUTER", "zero-shot NLI", fill=WARNBG)
    d.add(Line(92, 51, 106, 51, strokeColor=MUTED, strokeWidth=0.8))
    d.add(Line(106, 51, 106, 150, strokeColor=MUTED, strokeWidth=0.8))
    arrow(106, 150, 118, 150)
    d.add(String(120, 120, "selects PII policy for downstream heads",
                 fontName="Helvetica-Oblique", fontSize=6.5, fillColor=MUTED))

    # right column: heads
    heads = [
        ("[1] PII SCANNER", "token classification", 200),
        ("[2] DOCUMENT Q&A", "retrieval + span QA", 152),
        ("[3] AMBIGUITY METER", "MLM entropy + NLI", 104),
        ("[4] CONTRADICTIONS", "bi-encoder + NLI", 56),
    ]
    for label, sub, y in heads:
        box(268, y, 118, 36, label, sub)
        d.add(Line(244, 150, 258, 150, strokeColor=MUTED, strokeWidth=0.8))
        d.add(Line(258, 150, 258, y + 18, strokeColor=MUTED, strokeWidth=0.8))
        arrow(258, y + 18, 266, y + 18)
    d.add(Line(216, 150, 244, 150, strokeColor=MUTED, strokeWidth=0.8))

    # outputs - title at the top of the box, items listed beneath it
    d.add(Rect(400, 104, 65, 88, fillColor=colors.white, strokeColor=ACCENT,
               strokeWidth=0.7, rx=3, ry=3))
    d.add(String(432, 178, "OUTPUT", fontName="Helvetica-Bold", fontSize=8,
                 fillColor=INK, textAnchor="middle"))
    for i, t in enumerate(["spans", "probabilities", "distances"]):
        d.add(String(432, 158 - i * 14, t, fontName="Helvetica", fontSize=7,
                     fillColor=MUTED, textAnchor="middle"))
    for _, _, y in heads:
        d.add(Line(386, y + 18, 393, y + 18, strokeColor=MUTED, strokeWidth=0.6))
    d.add(Line(393, 74, 393, 218, strokeColor=MUTED, strokeWidth=0.6))
    arrow(393, 148, 398, 148)

    d.add(String(0, 14, "All five heads are output layers over BERT-family "
                        "encoders. No decoder appears anywhere in the system.",
                 fontName="Helvetica-Oblique", fontSize=7, fillColor=MUTED))
    return d


def token_chart():
    """Bar chart: document length against BERT's 512-token window."""
    W, H = 465, 96
    d = Drawing(W, H)
    data = [("contract.txt", 6088), ("medical_report.txt", 890),
            ("hr_grievance_email.txt", 319)]
    maxv = 6500
    scale = 300.0 / maxv
    x512 = 120 + 512 * scale
    for i, (name, val) in enumerate(data):
        y = H - 26 - i * 22
        d.add(String(0, y, name, fontName="Helvetica", fontSize=7.5,
                     fillColor=INK))
        d.add(Rect(120, y - 3, val * scale, 11,
                   fillColor=ACCENT if val > 512 else colors.HexColor("#7ba7cc"),
                   strokeColor=None))
        # keep the label clear of the 512-token rule, or it prints through it
        d.add(String(max(124 + val * scale, x512 + 8), y,
                     f"{val:,} tokens  ({val/512:.1f}x)",
                     fontName="Helvetica", fontSize=7, fillColor=MUTED))
    d.add(Line(x512, 4, x512, H - 12, strokeColor=colors.HexColor("#c0392b"),
               strokeWidth=1))
    d.add(String(x512 + 3, 6, "BERT limit: 512", fontName="Helvetica-Bold",
                 fontSize=7, fillColor=colors.HexColor("#c0392b")))
    return d


# --------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(22 * mm, 16 * mm, A4[0] - 22 * mm, 16 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(22 * mm, 11 * mm, "DocAuditor - Pipeline Architecture")
    canvas.drawRightString(A4[0] - 22 * mm, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=22 * mm, rightMargin=22 * mm,
                          topMargin=20 * mm, bottomMargin=22 * mm,
                          title="DocAuditor - Pipeline Architecture",
                          author="Preeyas Tumulu")
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=on_page)])

    s = []

    # ---- title -----------------------------------------------------------
    s.append(Spacer(1, 26 * mm))
    s.append(Paragraph("DocAuditor", TITLE))
    s.append(Paragraph("Encoder-Only Document Intelligence for Legal, HR "
                       "and Compliance Review", SUB))
    s.append(Spacer(1, 4 * mm))
    s.append(callout("Thesis",
        "Every feature in this system is a different output head on the same "
        "family of bidirectional encoders. No generative model is used at any "
        "point. The result is a <b>glass-box auditor</b>: every output is a "
        "character span, a calibrated probability, or a cosine distance - each "
        "traceable to the source text. An encoder has no decoder, so there is "
        "no mechanism by which it can invent a clause that is not there. That "
        "is the entire argument for this architecture in a regulated setting, "
        "where an unverifiable answer is worse than no answer."))
    s.append(Spacer(1, 8 * mm))
    s.append(table([
        [p("Component", CELLB), p("Deliverable", CELLB)],
        [p("Pipeline", CELL), p("DocAuditor.ipynb - 36 cells, executes end to end", CELL)],
        [p("Interface", CELL), p("app.py - Streamlit, standalone", CELL)],
        [p("Corpus", CELL), p("3 documents, 54 annotated PII spans, 3 planted defects", CELL)],
        [p("Hardware", CELL), p("NVIDIA RTX 3050, 4 GB VRAM; CUDA 12.1, torch 2.5.1", CELL)],
    ], [42 * mm, 123 * mm]))

    # ---- 1. architecture -------------------------------------------------
    s.append(Paragraph("1. System Architecture", H1))
    s.append(p("A document is parsed, segmented into clauses, and encoded once. "
               "The router runs first and configures the remaining heads; the "
               "four analysis heads then operate independently over the shared "
               "clause set."))
    s.append(Spacer(1, 3 * mm))
    s.append(architecture_diagram())
    s.append(Spacer(1, 3 * mm))

    s.append(Paragraph("1.1 Model stack", H2))
    s.append(table([
        [p("#", CELLB), p("Feature", CELLB), p("Model", CELLB), p("Head", CELLB)],
        [p("1", CELL), p("PII Scanner", CELL),
         p("iiiorg/piiranha-v1 (mDeBERTa-v3)", MONO), p("token classification", CELL)],
        [p("2", CELL), p("Document Q&amp;A", CELL),
         p("all-MiniLM-L6-v2 + roberta-base-squad2", MONO), p("retrieval + span QA", CELL)],
        [p("3", CELL), p("Ambiguity Meter", CELL),
         p("bert-base-uncased + nli-deberta-v3-base", MONO), p("MLM entropy + NLI", CELL)],
        [p("4", CELL), p("Contradiction Detector", CELL),
         p("all-MiniLM-L6-v2 + nli-deberta-v3-base", MONO), p("bi-encoder filter + NLI", CELL)],
        [p("5", CELL), p("Document Auto-Router", CELL),
         p("nli-deberta-v3-base", MONO), p("zero-shot classification", CELL)],
    ], [8 * mm, 34 * mm, 70 * mm, 53 * mm], align_right=(0,)))
    s.append(Spacer(1, 2 * mm))
    s.append(p("<b>Are these all BERT?</b> Yes. RoBERTa is BERT with improved "
               "pretraining (no next-sentence prediction, dynamic masking). "
               "DeBERTa-v3 is BERT with disentangled attention and "
               "ELECTRA-style pretraining. MiniLM is a distilled BERT-family "
               "sentence encoder. All are bidirectional transformer encoders "
               "trained with a masked-language-model objective. Model "
               "identifiers are centralised in a single dictionary, so the "
               "entire stack can be swapped to vanilla <font face='Courier'>"
               "bert-base-*</font> checkpoints in one edit."))

    # ---- 2. constraint ---------------------------------------------------
    s.append(Paragraph("2. The Governing Constraint", H1))
    s.append(p("BERT reads a maximum of 512 word-piece tokens. The test "
               "contract is roughly twelve times that limit. This single fact "
               "drives every significant design decision in the system."))
    s.append(token_chart())
    s.append(Spacer(1, 2 * mm))
    s.append(table([
        [p("Head", CELLB), p("Consequence of the 512-token window", CELLB)],
        [p("Document Q&amp;A", CELL),
         p("Cannot read the document. Retrieves candidate clauses with a "
           "bi-encoder, then reads only the best few.", CELL)],
        [p("Contradiction", CELL),
         p("Cannot compare all pairs (2,211 for 67 clauses). Filters "
           "candidates by embedding similarity before the cross-encoder.", CELL)],
        [p("Auto-Router", CELL),
         p("Cannot see the whole document. Classifies from a representative "
           "excerpt sampling both the head and the middle.", CELL)],
        [p("PII Scanner", CELL),
         p("Slides an overlapping window across the text and maps every span "
           "back to absolute offsets; overlap prevents boundary truncation.", CELL)],
    ], [32 * mm, 133 * mm]))
    s.append(Spacer(1, 2 * mm))
    s.append(p("These are not workarounds bolted on afterwards. Retrieval, "
               "candidate filtering and windowing <i>are</i> the architecture.",
               SMALL))

    # ---- 3. feature detail ----------------------------------------------
    s.append(Paragraph("3. Feature Design", H1))

    s.append(Paragraph("3.1 PII Scanner (token classification)", H2))
    s.append(p("Each word-piece receives a label from a 17-class PII tag set; "
               "adjacent tokens sharing a tag merge into a span. A "
               "deterministic regex layer runs alongside for structurally "
               "regular identifiers (e-mail, card numbers with a Luhn check, "
               "national IDs). The regex is a safety net, not a replacement: "
               "the model finds names and addresses no pattern can express, "
               "while the regex guarantees a card number is never lost to a "
               "tokenisation quirk. Recording which source found each span "
               "makes the model's real contribution measurable."))

    s.append(Paragraph("3.2 Document Q&amp;A (retrieval + extractive span QA)", H2))
    s.append(p("Two stages: a bi-encoder ranks clauses against the question, "
               "then a span-QA head decodes start and end logits over the best "
               "candidates. The answer is a pair of indices into the input, "
               "which is precisely why the model cannot fabricate: it can only "
               "point."))
    s.append(callout("Known limitation - stated, not hidden",
        "Extractive QA can only return text that literally exists in the "
        "document. Asked \"what are the risks here?\" it will fail, because "
        "that answer is written nowhere and would have to be composed. The "
        "mitigation is a confidence floor: below it the system answers "
        "<i>\"Not stated in document\"</i> rather than guessing. For a "
        "compliance tool an honest abstention is far more valuable than a "
        "plausible fabrication - which is exactly the failure mode a "
        "generative model would exhibit here.", WARNBG))

    s.append(Paragraph("3.3 Ambiguity Meter (MLM entropy + NLI)", H2))
    s.append(p("Testing for ambiguous language sounds like it requires "
               "generation. It does not. Two encoder-native signals combine:"))
    s.append(p("<b>Masked-language-model entropy.</b> Mask the term and measure "
               "the entropy of BERT's predictive distribution over the "
               "vocabulary. If context strongly determines the word, the "
               "distribution is sharp and entropy is low. A vague legal "
               "standard leaves the slot genuinely open, so entropy is high. "
               "This reads BERT's original pretraining objective directly, "
               "which is why it needs no fine-tuning.", BODY))
    s.append(p("H = - sum over vocabulary of P(w | context) * log P(w | context)",
               MONO))
    s.append(p("<b>NLI double-entailment.</b> If a clause entails both a strict "
               "and a permissive reading, it is not merely vague but genuinely "
               "ambiguous - two parties could read it in good faith and reach "
               "opposite conclusions. High entropy alone means <i>vague</i>; "
               "high entropy with double-entailment means <i>ambiguous</i>, a "
               "stronger and more actionable finding."))

    s.append(Paragraph("3.4 Cross-Clause Contradiction Detector", H2))
    s.append(p("Finds clause pairs that cannot both hold - the classic contract "
               "defect where one section says thirty days and another says "
               "fifteen. Comparing every clause with every other is quadratic: "
               "2,211 pairs for 67 clauses, each requiring a full forward pass "
               "over a concatenated pair. The design mirrors retrieve-then-rerank:"))
    s.append(table([
        [p("Stage", CELLB), p("Model", CELLB), p("Purpose", CELLB)],
        [p("1. Filter (cheap)", CELL), p("bi-encoder", CELL),
         p("Embed each clause once; keep only topically related pairs. A "
           "payment clause can only contradict another payment clause. "
           "Removes 97% of pairs.", CELL)],
        [p("2. Judge (costly)", CELL), p("NLI cross-encoder", CELL),
         p("Run on survivors only, in both directions, using the model's "
           "native contradiction output.", CELL)],
    ], [30 * mm, 30 * mm, 105 * mm]))

    s.append(Paragraph("3.5 Document Auto-Router (zero-shot classification)", H2))
    s.append(p("An NLI model scores how strongly the document entails each of "
               "several candidate descriptions - no training, no labelled data. "
               "The verdict is consequential rather than cosmetic: it selects "
               "the PII policy, so a national ID number is escalated to high "
               "severity in a medical record but not in a commercial contract. "
               "The excerpt samples the head <i>and</i> the middle of the "
               "document, because the opening of a contract and the opening of "
               "a medical record both look like a title block."))

    # ---- 4. results ------------------------------------------------------
    s.append(Paragraph("4. Evaluation", H1))
    s.append(p("Because the corpus was built with synthetic PII injected at "
               "recorded offsets, the scanner can be <i>scored</i> rather than "
               "demonstrated. Offsets are exact by construction - captured "
               "during document assembly, not searched for afterwards - and the "
               "preparation script asserts every offset against the written "
               "file before exiting."))
    s.append(Paragraph("4.1 PII detection", H2))
    s.append(table([
        [p("Document", CELLB), p("TP", CELLB), p("FP", CELLB), p("FN", CELLB),
         p("Precision", CELLB), p("Recall", CELLB), p("F1", CELLB)],
        [p("contract.txt", CELL), p("32", CELL), p("7", CELL), p("5", CELL),
         p("82.1%", CELL), p("86.5%", CELL), p("84.2%", CELL)],
        [p("medical_report.txt", CELL), p("9", CELL), p("7", CELL), p("3", CELL),
         p("56.3%", CELL), p("75.0%", CELL), p("64.3%", CELL)],
        [p("hr_grievance_email.txt", CELL), p("13", CELL), p("4", CELL), p("4", CELL),
         p("76.5%", CELL), p("76.5%", CELL), p("76.5%", CELL)],
        [p("<b>Overall</b>", CELL), p("<b>54</b>", CELL), p("<b>18</b>", CELL),
         p("<b>12</b>", CELL), p("<b>75.0%</b>", CELL), p("<b>81.8%</b>", CELL),
         p("<b>78.3%</b>", CELL)],
    ], [46 * mm, 13 * mm, 13 * mm, 13 * mm, 27 * mm, 26 * mm, 27 * mm],
        align_right=(1, 2, 3, 4, 5, 6)))
    s.append(Spacer(1, 1 * mm))
    s.append(p("Scoring is by span overlap rather than exact label match. The "
               "answer key uses FULLNAME where the model emits GIVENNAME and "
               "SURNAME separately; penalising that would measure a taxonomy "
               "difference, not a mistake.", SMALL))

    s.append(Paragraph("4.2 Auto-router and contradiction detector", H2))
    s.append(table([
        [p("Measurement", CELLB), p("Result", CELLB)],
        [p("Router accuracy", CELL),
         p("3 / 3 documents correct (contract 95%, HR e-mail 99%, medical 65%)", CELL)],
        [p("Contradiction search space", CELL),
         p("67 clauses, 2,211 possible pairs", CELL)],
        [p("Pairs judged after filtering", CELL),
         p("61 (97.2% eliminated before the cross-encoder)", CELL)],
        [p("Findings at default setting", CELL),
         p("9, including the planted governing-law conflict", CELL)],
    ], [55 * mm, 110 * mm]))

    # ---- 5. the interesting finding --------------------------------------
    s.append(Paragraph("5. What the Evaluation Revealed", H1))
    s.append(callout("The corpus assumption was wrong, and the tool proved it",
        "The first evaluation scored 57% precision on the contract. Inspecting "
        "every apparent false positive showed that <b>none of them were "
        "false</b>. They were real names, e-mail addresses, telephone numbers "
        "and street addresses belonging to identifiable people, already present "
        "in the CUAD source document.<br/><br/>"
        "The corpus had been built on the assumption that SEC filings are "
        "pre-redacted and therefore PII-free - which is why synthetic PII was "
        "injected in the first place. That assumption was false. The "
        "<font face='Courier'>[ * * * ]</font> markers redact commercially "
        "sensitive terms; the notice blocks, containing real contact details, "
        "were published untouched.<br/><br/>"
        "Each detection was manually verified against the source and added to "
        "the answer key. Precision rose from 57% to 75%, and F1 from 67% to "
        "78% - not by tuning the model, but by correcting a flawed measurement."
        "<br/><br/>"
        "<b>A tool built to find personal data in documents found personal data "
        "that a regulatory filing process had missed, on real public data.</b> "
        "That is the compliance use case, demonstrated by accident."))

    # ---- 6. calibration --------------------------------------------------
    s.append(Paragraph("6. Calibrating the Contradiction Detector", H1))
    s.append(p("The first working version reported <b>116 contradictions from "
               "400 judged pairs</b>, nearly all scoring 1.00. No reviewer will "
               "read 116 findings, and spot-checking showed most were not "
               "contradictions at all."))
    s.append(p("<b>This was not a bug.</b> The label order, softmax and raw "
               "logits were all verified correct, and on clean sentence pairs "
               "the model is well calibrated (true contradiction 1.000, "
               "unrelated 0.000, entailment 0.000). The failure is domain "
               "mismatch: NLI is trained on short everyday sentences, while "
               "legal prose hedged with \"notwithstanding\" and party-specific "
               "scoping reads as contradictory to a model that has never seen "
               "a contract."))
    s.append(Paragraph("6.1 A hypothesis that was tested and rejected", H2))
    s.append(p("Requiring contradiction in <i>both</i> directions appeared "
               "principled - entailment is directional, but mutual exclusivity "
               "should be symmetric. It reduced findings from 116 to 40 and "
               "scored every planted defect at 0.007-0.067. The rule would have "
               "destroyed every true detection while appearing to improve "
               "precision. These contradictions are strongly asymmetric in "
               "practice, and the idea was discarded on the evidence."))
    s.append(Paragraph("6.2 The effective lever, measured", H2))
    s.append(p("Model failures concentrate in long clauses, so clause length is "
               "the useful control. The trade-off is exposed as a parameter "
               "rather than hidden behind a tuned constant:"))
    s.append(table([
        [p("Clause length cap", CELLB), p("Threshold", CELLB),
         p("Findings to review", CELLB), p("Planted defects found", CELLB)],
        [p("250 (default)", CELL), p("0.95", CELL), p("9", CELL), p("2 / 3", CELL)],
        [p("350", CELL), p("0.95", CELL), p("17", CELL), p("2 / 3", CELL)],
        [p("500", CELL), p("0.95", CELL), p("34", CELL), p("2 / 3", CELL)],
        [p("none", CELL), p("0.95", CELL), p("102", CELL), p("3 / 3", CELL)],
        [p("none", CELL), p("0.99", CELL), p("68", CELL), p("2 / 3", CELL)],
    ], [42 * mm, 30 * mm, 45 * mm, 48 * mm], align_right=(1, 2, 3)))
    s.append(Spacer(1, 1 * mm))
    s.append(p("The third defect is genuinely missed at the default setting: "
               "its conflicting source language sits in a 1,145-character "
               "clause, above the cap that makes the output reviewable. Which "
               "operating point is correct depends on whether a human reviews "
               "every hit or triages a shortlist.", SMALL))

    # ---- 7. limitations --------------------------------------------------
    s.append(Paragraph("7. Limitations", H1))
    s.append(table([
        [p("Limitation", CELLB), p("Consequence", CELLB)],
        [p("Contradiction precision is weak", CELL),
         p("Roughly one real defect per nine findings. A triage aid, not an "
           "oracle. Domain-tuned NLI would be the fix.", CELL)],
        [p("One planted defect missed", CELL),
         p("Recall failure at the default clause cap; recoverable only by "
           "accepting ~102 findings.", CELL)],
        [p("Extractive QA cannot compose", CELL),
         p("No summarisation, no synthesis, no \"what are the risks\". "
           "Abstains rather than guessing.", CELL)],
        [p("Zero-shot NLI is not domain-tuned", CELL),
         p("Thresholds are heuristic rather than calibrated against a "
           "labelled legal corpus.", CELL)],
        [p("Similarity floor bounds recall", CELL),
         p("A contradictory pair below the floor is never judged. Lowering it "
           "costs quadratically more compute.", CELL)],
        [p("Medical precision is lowest", CELL),
         p("56.3% - clinical prose contains dates and identifiers that "
           "resemble PII structurally.", CELL)],
    ], [50 * mm, 115 * mm]))

    # ---- 8. data ---------------------------------------------------------
    s.append(Paragraph("8. Data Provenance and Ethics", H1))
    s.append(table([
        [p("Document", CELLB), p("Source", CELLB), p("Licence", CELLB)],
        [p("contract.txt", CELL),
         p("CUAD - real SEC EDGAR commercial services agreement", CELL),
         p("CC BY 4.0", CELL)],
        [p("medical_report.txt", CELL),
         p("MTSamples - real de-identified clinical transcription", CELL),
         p("CC0", CELL)],
        [p("hr_grievance_email.txt", CELL),
         p("Synthetic, authored for this project", CELL), p("-", CELL)],
    ], [42 * mm, 96 * mm, 27 * mm]))
    s.append(Spacer(1, 2 * mm))
    s.append(p("<b>Why real sources.</b> Real documents provide authentic "
               "language; synthetic PII provides a ground-truth answer key. "
               "Using only generated documents would yield an answer key but "
               "unrealistically clean prose, giving the ambiguity meter nothing "
               "genuine to analyse. Using only real documents would give "
               "realistic language but no way to measure anything."))
    s.append(p("<b>Why the HR thread is synthetic.</b> Real grievance "
               "correspondence is not public, for good reason, and the widely "
               "used real e-mail corpora contain personal data whose subjects "
               "never consented to its release. Authoring that document was the "
               "correct choice rather than a convenient one."))
    s.append(p("<b>Disclosure.</b> As documented in section 5, "
               "<font face='Courier'>contract.txt</font> contains real personal "
               "data inherited from the public SEC filing. It is retained "
               "because removing it would destroy the most informative result "
               "in the evaluation, and because the data is already published in "
               "a public regulatory database.", SMALL))

    # ---- 9. future -------------------------------------------------------
    s.append(Paragraph("9. Future Work", H1))
    s.append(p("<b>Domain-tuned NLI.</b> Fine-tuning the entailment head on "
               "contract-specific clause pairs would replace heuristic "
               "thresholds with calibrated ones and directly address the "
               "contradiction detector's precision, the system's weakest point."))
    s.append(p("<b>Playbook deviation detection.</b> Embedding each clause "
               "against a library of company-approved clauses would extend the "
               "system from finding defects to proposing the approved "
               "alternative - the capability that separates a review tool from "
               "a drafting assistant."))
    s.append(p("<b>Hierarchical segmentation.</b> Splitting long clauses into "
               "sentence-level units for contradiction analysis would recover "
               "the missed defect without accepting the false-positive rate of "
               "an uncapped search."))

    s.append(Spacer(1, 6 * mm))
    s.append(p("References: Devlin et al., <i>BERT</i> (2019). He et al., "
               "<i>DeBERTa</i> (2021). Hendrycks et al., <i>CUAD</i> (2021). "
               "Reimers &amp; Gurevych, <i>Sentence-BERT</i> (2019). "
               "Williams et al., <i>MultiNLI</i> (2018).", SMALL))

    doc.build(s)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
