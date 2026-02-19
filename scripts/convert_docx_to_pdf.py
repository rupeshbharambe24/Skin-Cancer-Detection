import os
from pathlib import Path


def main():
    docx = Path("Results_and_Discussion_Report.docx").resolve()
    pdf = Path("Results_and_Discussion_Report.pdf").resolve()

    # Keep temp vars explicit (Word/COM may still use other caches, but this helps).
    temp = os.environ.get("TEMP") or os.environ.get("TMP")
    print("DOCX:", docx)
    print("PDF :", pdf)
    print("TEMP:", temp)
    print("TMP :", os.environ.get("TMP"))

    if not docx.exists():
        raise SystemExit(f"Missing: {docx}")

    try:
        from docx2pdf import convert
    except Exception as e:
        raise SystemExit(f"docx2pdf import failed: {e}")

    convert(str(docx), str(pdf))
    print("Wrote:", pdf)


if __name__ == "__main__":
    main()

