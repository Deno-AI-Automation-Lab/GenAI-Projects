"""Generate samples/sample_schemes.pdf - a small bilingual PDF (English + Tamil,
with a table) used by the tests, the eval set and for a first demo.
All content is fictional sample data.

Run:  python scripts/make_sample_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hybrid_rag.config import TAMIL_FONT  # noqa: E402

OUT = ROOT / "samples" / "sample_schemes.pdf"

INTRO = (
    "Sample Welfare Schemes Handbook 2026 (fictional data for testing).\n\n"
    "This handbook describes support schemes run by state departments. Each scheme lists "
    "eligibility, benefits and the issuing authority. Applications are submitted through the "
    "e-Services portal or at the nearest block office.\n\n"
    "Green Farm Mission is issued by the Department of Agriculture. Small and marginal farmers "
    "holding up to 5 acres are eligible. The scheme provides a subsidy of Rs. 25,000 per acre for "
    "drip irrigation. The last date to apply is 31 March 2027.\n\n"
    "Solar Pump Support is issued by the Department of Energy. It gives a 70 percent subsidy on "
    "solar water pumps up to 10 HP. Farmers with an existing borewell are eligible.\n\n"
    "Coastal Livelihood Grant is issued by the Department of Fisheries. Registered fisherfolk "
    "families receive Rs. 6,000 per year during the fishing ban period."
)
TAMIL = ("பசுமை பண்ணை திட்டம் வேளாண்மைத் துறையால் செயல்படுத்தப்படுகிறது. "
         "சிறு மற்றும் குறு விவசாயிகளுக்கு சொட்டு நீர் பாசனத்திற்கு மானியம் வழங்கப்படுகிறது.")
TABLE = [
    ["Scheme", "Department", "Benefit", "Eligibility"],
    ["Green Farm Mission", "Agriculture", "Rs. 25,000 per acre drip subsidy", "Farmers up to 5 acres"],
    ["Solar Pump Support", "Energy", "70% subsidy on solar pumps", "Farmers with borewell"],
    ["Coastal Livelihood Grant", "Fisheries", "Rs. 6,000 per year", "Registered fisherfolk"],
]


def make_pdf(out: Path = OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 545, 520), INTRO, fontsize=11, fontname="helv")
    page = doc.new_page()
    page.insert_font(fontname="tamil", fontfile=str(TAMIL_FONT))  # bundled font
    page.insert_textbox(pymupdf.Rect(50, 50, 545, 160), TAMIL, fontsize=12, fontname="tamil")
    # Draw a ruled table so PyMuPDF's table finder can detect it.
    x0, y0, widths, h = 50, 200, [140, 90, 160, 110], 28
    for r, row in enumerate(TABLE):
        x = x0
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x, y0 + r * h, x + widths[c], y0 + (r + 1) * h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page.insert_textbox(rect + (3, 3, -3, -3), cell, fontsize=8, fontname="helv")
            x += widths[c]
    doc.save(out)
    doc.close()
    return out


if __name__ == "__main__":
    print(make_pdf())
