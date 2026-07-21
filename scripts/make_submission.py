"""
Builds the submission zip.

Run:  python scripts/make_submission.py

Deliberately excludes .venv, model caches and git history. The virtual
environment is several gigabytes of platform-specific binaries that would not
work on the marker's machine anyway; requirements.txt reproduces it instead.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDENT = "PreeyasTumulu"
OUT = ROOT / f"{STUDENT}-GenerativeAI-UT01-DocAuditor.zip"

# Explicit allow-list. An exclude-list would silently ship whatever new junk
# happens to be sitting in the folder on the day.
INCLUDE_FILES = [
    "DocAuditor.ipynb",                 # the pipeline (with saved outputs)
    "DocAuditor_Architecture.pdf",      # architecture documentation
    "app.py",                           # Streamlit interface
    "README.md",
    "requirements.txt",
    "scripts/prepare_data.py",
    "scripts/build_architecture_pdf.py",
    "scripts/make_submission.py",
    "scripts/fetch_sample_documents.py",
]
INCLUDE_GLOBS = ["data/test_documents/*", "data/sample_documents/*"]


def main() -> None:
    files: list[Path] = []
    missing: list[str] = []

    for rel in INCLUDE_FILES:
        path = ROOT / rel
        (files if path.is_file() else missing).append(path if path.is_file() else rel)

    for pattern in INCLUDE_GLOBS:
        matched = sorted(p for p in ROOT.glob(pattern) if p.is_file())
        if not matched:
            missing.append(pattern)
        files.extend(matched)

    if missing:
        raise SystemExit("Missing expected files:\n  " + "\n  ".join(map(str, missing)))

    # Guard: a notebook without saved outputs shows the marker nothing.
    import json
    nb = json.loads((ROOT / "DocAuditor.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    no_out = [c for c in code if not c.get("outputs")]
    errored = [c for c in code
               if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    if no_out:
        raise SystemExit(
            f"REFUSING TO PACKAGE: {len(no_out)}/{len(code)} code cells have no saved "
            "output. Run:\n  jupyter nbconvert --to notebook --execute --inplace "
            "--ExecutePreprocessor.kernel_name=docauditor DocAuditor.ipynb")
    if errored:
        raise SystemExit(f"REFUSING TO PACKAGE: {len(errored)} cell(s) contain errors.")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            z.write(path, Path(STUDENT + "-DocAuditor") / path.relative_to(ROOT))

    total = sum(f.stat().st_size for f in files)
    print(f"{OUT.name}")
    print(f"  files      : {len(files)}")
    print(f"  raw size   : {total/1e6:.2f} MB")
    print(f"  zipped     : {OUT.stat().st_size/1e6:.2f} MB")
    print(f"  notebook   : {len(code)} code cells, all with saved outputs")
    print("\ncontents:")
    for path in files:
        print(f"   {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
