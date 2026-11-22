"""Build docs/viva/LungAI_Viva_Guide.docx from docs/viva/LungAI_Viva_Guide.md.

Run:  D:\\LungAI\\.venv-audit\\Scripts\\python.exe docs\\viva\\build_viva_guide.py
The Markdown file is the single source; this script only converts it
(headings, bullets, numbered lists, checkboxes, simple tables, code blocks,
bold/inline code) and adds page numbers in the footer.
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SRC = HERE / "LungAI_Viva_Guide.md"
OUT = HERE / "LungAI_Viva_Guide.docx"

BODY_FONT = "Calibri"
MONO_FONT = "Consolas"


def set_font(style, name, size, bold=None, color=None):
    style.font.name = name
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = RGBColor(*color)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)


def add_runs(par, text, bold_all=False):
    """Add text with **bold** and `code` inline markup."""
    parts = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = MONO_FONT
            r._element.rPr.rFonts.set(qn("w:eastAsia"), MONO_FONT)
            r.font.size = Pt(9.5)
            if bold_all:
                r.bold = True
        else:
            r = par.add_run(part)
            if bold_all:
                r.bold = True


def add_page_number(section):
    par = section.footer.paragraphs[0]
    par.alignment = 1
    run = par.add_run("LungAI Viva Guide, page ")
    run.font.size = Pt(9)
    run2 = par.add_run()
    run2.font.size = Pt(9)
    for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        run2._r.append(el)


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def flush_table(doc, rows):
    if not rows:
        return
    header = rows[0]
    body = [r for r in rows[1:] if not all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in r)]
    ncols = len(header)
    table = doc.add_table(rows=1, cols=ncols)
    table.style = "Table Grid"
    table.autofit = True
    for i, h in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = ""
        add_runs(cell.paragraphs[0], h.strip(), bold_all=True)
        shade(cell, "D9E2F3")
        for r in cell.paragraphs[0].runs:
            r.font.size = Pt(10)
    for row in body:
        cells = table.add_row().cells
        for i in range(ncols):
            txt = row[i].strip() if i < len(row) else ""
            cells[i].text = ""
            add_runs(cells[i].paragraphs[0], txt)
            for r in cells[i].paragraphs[0].runs:
                r.font.size = Pt(10)
    # repeat header row
    trpr = table.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trpr.append(th)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def build():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    add_page_number(sec)

    set_font(doc.styles["Normal"], BODY_FONT, 11)
    doc.styles["Normal"].paragraph_format.space_after = Pt(6)
    set_font(doc.styles["Heading 1"], BODY_FONT, 18, True, (0x1F, 0x3A, 0x5F))
    set_font(doc.styles["Heading 2"], BODY_FONT, 14, True, (0x2F, 0x54, 0x96))
    set_font(doc.styles["Heading 3"], BODY_FONT, 12, True, (0x2F, 0x54, 0x96))
    set_font(doc.styles["Title"], BODY_FONT, 26, True, (0x1F, 0x3A, 0x5F))
    for name in ("List Bullet", "List Number"):
        set_font(doc.styles[name], BODY_FONT, 11)

    table_rows = []
    in_code = False
    first_h1_seen = False
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.left_indent = Cm(0.6)
            r = p.add_run(line)
            r.font.name = MONO_FONT
            r.font.size = Pt(9.5)
            continue
        if line.startswith("|"):
            cells = line.strip().strip("|").split("|")
            table_rows.append(cells)
            continue
        elif table_rows:
            flush_table(doc, table_rows)
            table_rows = []
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "---":
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1 and not first_h1_seen:
                first_h1_seen = True
                doc.add_paragraph(text, style="Title")
                continue
            if level == 1:
                doc.add_page_break()
            doc.add_heading(text, level=level)
            continue
        if stripped.startswith("- [ ] "):
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, "[  ] " + stripped[6:])
            continue
        if stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, stripped[2:])
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            add_runs(p, f"{m.group(1)}. {m.group(2)}")
            continue
        # question line: starts with **Q or **H -> keep with next, bold
        p = doc.add_paragraph()
        if re.match(r"^\*\*[QH]\d+\.", stripped) and stripped.endswith("**"):
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
        add_runs(p, stripped)
    if table_rows:
        flush_table(doc, table_rows)
    doc.core_properties.title = "LungAI Viva Guide"
    doc.core_properties.author = "Katkam Sathwik"
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    out = build()
    check = Document(out)
    words = sum(len(p.text.split()) for p in check.paragraphs)
    for t in check.tables:
        for row in t.rows:
            for c in row.cells:
                words += len(c.text.split())
    print(f"wrote {out}: {len(check.paragraphs)} paragraphs, {len(check.tables)} tables, ~{words} words")
    sys.exit(0)
