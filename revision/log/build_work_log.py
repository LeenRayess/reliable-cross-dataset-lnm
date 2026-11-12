"""Build revision/IEEE_revision_work_log.docx from log_content.py."""

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import log_content as C  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "IEEE_revision_work_log.docx"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)


def add_runs(par, text):
    for i, part in enumerate(re.split(r"\*\*", text)):
        if part:
            r = par.add_run(part)
            r.bold = i % 2 == 1


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def add_table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        add_runs(c.paragraphs[0], "**" + h + "**")
        shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            add_runs(cells[i].paragraphs[0], str(v))
    for row in t.rows:
        for i, c in enumerate(row.cells):
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
            if widths:
                c.width = Cm(widths[i])
    doc.add_paragraph()


def render_blocks(doc, blocks):
    for kind, payload in blocks:
        if kind == "h":
            doc.add_heading(payload, level=3)
        elif kind == "p":
            add_runs(doc.add_paragraph(), payload)
        elif kind == "b":
            for item in payload:
                add_runs(doc.add_paragraph(style="List Bullet"), item)
        elif kind == "t":
            header, rows = payload
            add_table(doc, header, rows)


def main():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    for lvl, size in [(1, 16), (2, 13), (3, 11.5)]:
        hs = doc.styles["Heading {}".format(lvl)]
        hs.font.name = "Calibri"
        hs.font.size = Pt(size)
        hs.font.color.rgb = ACCENT
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.2)

    doc.add_heading(C.TITLE, level=0)
    add_runs(doc.add_paragraph(), C.SUBTITLE)
    render_blocks(doc, C.INTRO)

    doc.add_heading("Reviewer-comment tracker", level=1)
    add_table(doc, C.TRACKER[0], C.TRACKER[1:], widths=[3.5, 8.0, 1.6, 3.4])
    render_blocks(doc, C.NOTES)

    doc.add_heading("Work log", level=1)
    for s in C.STEPS:
        doc.add_heading(s["title"], level=2)
        add_runs(doc.add_paragraph(), "**Date:** {}    **Answers:** {}".format(s["date"], s["answers"]))
        render_blocks(doc, s["blocks"])

    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
