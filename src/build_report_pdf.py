"""report/REPORT.md -> report/report.pdf (Markdown -> HTML -> headless Chrome print).

Usage: python src/build_report_pdf.py [--chrome PATH]
"""
import argparse
import subprocess
from pathlib import Path

import markdown

REP = Path(__file__).resolve().parents[1] / "report"
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; margin: 0 0 4pt; } h2 { font-size: 12.5pt; margin: 14pt 0 4pt; border-bottom: 1px solid #ddd; }
p, li { margin: 3pt 0; } ul, ol { margin: 2pt 0 4pt; padding-left: 18pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8.6pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 2.5pt 4pt; text-align: left; vertical-align: top; } th { background: #f3f3f1; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
code { font-size: 8.6pt; background: #f3f3f1; padding: 0 2pt; }
h2, h3 { break-after: avoid; } p:has(+ table), p:has(+ p > img) { break-after: avoid; }
img, p:has(> img) { break-after: avoid; } img + p, p:has(> img) + p { break-before: avoid; } th { break-after: avoid; }
"""
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def md_fix(text):
    """Python-Markdown needs a blank line before a list and 4-space nesting; the report uses CommonMark style."""
    out, prev = [], ""
    for line in text.splitlines():
        stripped = line.lstrip()
        ind = len(line) - len(stripped)
        is_item = stripped[:2] in ("- ", "* ") or stripped[:1].isdigit() and stripped[1:3] == ". "
        if is_item:
            line = " " * (ind * 2) + stripped
            if prev.strip() and not prev.lstrip()[:2] in ("- ", "* ") and not prev.lstrip()[1:3] == ". " and ind == 0:
                out.append("")
        elif ind and prev.lstrip()[:2] in ("- ", "* "):
            line = " " * (ind * 2) + stripped
        out.append(line)
        prev = line
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default=CHROME)
    ap.add_argument("--src", default="REPORT.md")
    ap.add_argument("--out", default="report.pdf")
    a = ap.parse_args()
    body = markdown.markdown(md_fix((REP / a.src).read_text()), extensions=["tables"])
    html = REP / "report.html"
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
    subprocess.run([a.chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={REP / a.out}", html.as_uri()], check=True, capture_output=True)
    html.unlink()
    print("wrote", REP / a.out)


if __name__ == "__main__":
    main()
