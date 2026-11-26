"""Re-export doom-rl-docs/BRIEF.pdf from BRIEF.md (A4, 2 pages max). Needs: pip markdown + Chrome or Edge.
usage (repo root):  python training/results/t3_checks/export_brief.py"""
import re, shutil, subprocess, sys, tempfile
from pathlib import Path
import markdown
root = Path(__file__).resolve().parents[3]
md, pdf = root / "doom-rl-docs" / "BRIEF.md", root / "doom-rl-docs" / "BRIEF.pdf"
css = ("@page{size:A4;margin:11mm 13mm}body{font-family:Calibri,Arial,sans-serif;font-size:9.6pt;line-height:1.28}"
       "h1{font-size:15pt;margin:0 0 2px}h2{font-size:11pt;margin:7px 0 3px;border-bottom:1px solid #999}p,ul,ol{margin:3px 0}"
       "ul,ol{padding-left:18px}blockquote{margin:4px 0;padding:3px 8px;border-left:3px solid #555;background:#f3f3f3}"
       "pre{font-size:6.9pt;line-height:1.15;background:#f6f6f6;padding:5px;white-space:pre}"
       "table{border-collapse:collapse;font-size:8.8pt}th,td{border:1px solid #aaa;padding:1px 5px}")
html = Path(tempfile.mkdtemp()) / "brief.html"
html.write_text(f"<!doctype html><meta charset=utf-8><style>{css}</style>"
                + markdown.markdown(md.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"]), encoding="utf-8")
browsers = [shutil.which(b) for b in ("chrome", "msedge", "google-chrome", "chromium")] + [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
exe = next((b for b in browsers if b and Path(b).exists()), None)
if not exe:
    sys.exit("no Chrome/Edge found")
subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.as_uri()],
               check=True, capture_output=True)
pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
print(f"{pdf} written, {pages} page(s)")
assert pages <= 2, "over 2 pages: shorten BRIEF.md"
