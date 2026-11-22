"""Build LungAI_MTech_Presentation.pptx from slides.md, results.json and docs/figures/*.png.

Usage (low CPU):  D:\\LungAI\\.venv-audit\\Scripts\\python.exe build_presentation.py [--verify-only]
"""
import json
import math
import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
SLIDES_MD = HERE / "slides.md"
RESULTS = DOCS / "manuscripts" / "shared" / "results.json"
FIG_DIR = DOCS / "figures"
OUT = HERE / "LungAI_MTech_Presentation.pptx"

NAVY = RGBColor(0x1B, 0x2F, 0x4E)
TEAL = RGBColor(0x1F, 0x7A, 0x8C)
LIGHT = RGBColor(0xEE, 0xF3, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x22, 0x2B, 0x36)
GREY = RGBColor(0x5F, 0x6B, 0x7A)
RED = RGBColor(0xC0, 0x1E, 0x1E)
BODY_FONT, HEAD_FONT = "Calibri", "Cambria"
SW, SH = 13.333, 7.5
FOOTER = "KUCE&T (Kakatiya University College of Engineering and Technology), Kakatiya University  |  M.Tech CSE  |  2026"

AUTHOR = [
    ("Katkam Sathwik", True),
    ("Roll No. 24567T7604  |  M.Tech, Computer Science and Engineering", False),
    ("Department of Computer Science and Engineering", False),
    ("KUCE&T (Kakatiya University College of Engineering and Technology)", False),
    ("Kakatiya University, Warangal, Telangana, India", False),
]
GUIDE = [
    ("Guide: Prof. Nagavelli Ramana, Principal and Head of the Department (CSE & IT), KUCE&T", False),
    ("Supporting faculty: Dr. P. Tirupathi, Assistant Professor, CSE, JNTUH University College of Engineering, Manthani", False),
    ("Dr. K. Srinivas, Assistant Professor, CSE, Swathi Reddy Institute of Technology for Women, Warangal", False),
]

PENDING_RE = re.compile(r"\[(PENDING[^\]]*|AUTHORS TO CONFIRM|TO BE PROVIDED|TO BE CONFIRMED)\]")


# ----------------------------------------------------------------- parsing
def load_results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


MISSING = set()


def subst(text, res):
    def rep(m):
        k = m.group(1)
        if k in res:
            return str(res[k])
        MISSING.add(k)
        return "[PENDING: " + k + "]"
    return re.sub(r"\{\{(\w+)\}\}", rep, text)


def parse_slides(res):
    raw = SLIDES_MD.read_text(encoding="utf-8")
    lines = [l for l in raw.splitlines() if not l.startswith("# ")]
    blocks, cur = [], []
    for l in lines:
        if l.strip() == "---":
            blocks.append(cur)
            cur = []
        else:
            cur.append(l)
    blocks.append(cur)
    slides = []
    for b in blocks:
        s = {"bullets": [], "table": [], "notes": []}
        for l in b:
            if not l.strip():
                continue
            if l.startswith("> "):
                s["notes"].append(subst(l[2:], res))
            elif l.startswith("- "):
                s["bullets"].append(subst(l[2:], res))
            elif l.startswith("| "):
                row = [c.strip() for c in subst(l, res).strip().strip("|").split("|")]
                if row == ["---"]:
                    s["table2"] = []
                elif "table2" in s:
                    s["table2"].append(row)
                else:
                    s["table"].append(row)
            else:
                m = re.match(r"(\w+):\s*(.*)", l)
                if m:
                    s[m.group(1)] = subst(m.group(2), res)
        if s.get("layout"):
            slides.append(s)
    return slides


# ----------------------------------------------------------------- helpers
def est_lines(text, width_in, size_pt):
    cpl = max(1, int(width_in * 72 / (size_pt * 0.50)))
    return max(1, math.ceil(len(text) / cpl))


def fit_size(items, width_in, height_in, hi=24, lo=18, gap_pt=8):
    for size in range(hi, lo - 1, -1):
        total = 0.0
        for t in items:
            total += est_lines(t, width_in - 0.45, size) * size * 1.2 + gap_pt
        if total / 72 <= height_in:
            return size, True
    return lo, False


def set_run(run, size, color=INK, bold=False, italic=False, font=BODY_FONT):
    run.font.size = Pt(size)
    run.font.name = font
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def add_text(slide, x, y, w, h, text, size, color=INK, bold=False, font=BODY_FONT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]
    p.alignment = align
    add_runs(p, text, size, color, bold, italic, font)
    return tb


def add_runs(p, text, size, color=INK, bold=False, italic=False, font=BODY_FONT):
    """Add runs, drawing [PENDING ...] / [AUTHORS TO CONFIRM] spans in red."""
    pos = 0
    for m in PENDING_RE.finditer(text):
        if m.start() > pos:
            r = p.add_run(); r.text = text[pos:m.start()]; set_run(r, size, color, bold, italic, font)
        r = p.add_run(); r.text = m.group(0); set_run(r, size, RED, True, italic, font)
        pos = m.end()
    if pos < len(text):
        r = p.add_run(); r.text = text[pos:]; set_run(r, size, color, bold, italic, font)


def bullet_para(p, level=0):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(0.3 + 0.3 * level))))
    pPr.set("indent", str(int(-Inches(0.3))))
    for tag in ("a:buNone", "a:buChar", "a:buClr"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    clr = etree.SubElement(pPr, qn("a:buClr"))
    c = etree.SubElement(clr, qn("a:srgbClr")); c.set("val", "1F7A8C")
    bu = etree.SubElement(pPr, qn("a:buChar")); bu.set("char", "\u25A0" if level == 0 else "\u2013")


def add_bullets(slide, x, y, w, h, items, name="Body"):
    size, _ = fit_size(items, w, h)
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        bullet_para(p)
        add_runs(p, t, size)
    return tb


def footer(slide, n, dark=False):
    col = WHITE if dark else GREY
    add_text(slide, 0.5, 7.02, 11.3, 0.35, FOOTER, 11, col, name="Footer")
    tb = slide.shapes.add_textbox(Inches(12.0), Inches(7.02), Inches(0.85), Inches(0.35))
    tb.name = "Slide number"
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    fld = etree.SubElement(p._p, qn("a:fld"))
    fld.set("id", "{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}")
    fld.set("type", "slidenum")
    rpr = etree.SubElement(fld, qn("a:rPr")); rpr.set("lang", "en-US"); rpr.set("sz", "1200")
    sf = etree.SubElement(rpr, qn("a:solidFill")); c = etree.SubElement(sf, qn("a:srgbClr"))
    c.set("val", "FFFFFF" if dark else "5F6B7A")
    t = etree.SubElement(fld, qn("a:t")); t.text = str(n)


def title_bar(slide, title):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SW), Inches(1.05))
    bar.name = "Title bar"
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    size = 32 if len(title) <= 48 else 28
    add_text(slide, 0.5, 0.1, 12.3, 0.85, title, size, WHITE, True, HEAD_FONT,
             anchor=MSO_ANCHOR.MIDDLE, name="Title")


def bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def place_picture(slide, stem, x, y, w, h, missing):
    path = FIG_DIR / (stem + ".png")
    if path.exists():
        from PIL import Image
        with Image.open(path) as im:
            iw, ih = im.size
        sc = min(w / iw, h / ih)
        pw, ph = iw * sc, ih * sc
        pic = slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                       Inches(pw), Inches(ph))
        pic.name = stem
        return
    missing.append(stem)
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = "Placeholder " + stem
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = GREY; box.line.dash_style = 4; box.line.width = Pt(1.5)
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = "Figure placeholder: " + stem + ".png"; set_run(r, 20, GREY, True)
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r = p2.add_run(); r.text = "docs/figures/ (not yet generated)"; set_run(r, 16, GREY)


def add_table(slide, rows, x, y, w, colw=None, max_h=4.6, name="Table"):
    nr, nc = len(rows), len(rows[0])
    longest = max(len(c) for r in rows for c in r)
    size = 16 if nr <= 7 else 14
    if nr > 7 and longest > 60:
        size = 14
    rh = 0.5 if size == 16 else 0.42
    gs = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(rh * nr))
    gs.name = name
    tbl = gs.table
    weights = colw or [1] * nc
    tot = sum(weights)
    for i, wt in enumerate(weights):
        tbl.columns[i].width = Inches(w * wt / tot)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(rh)
        for ci in range(nc):
            cell = tbl.cell(ri, ci)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if ri == 0 else (LIGHT if ri % 2 == 0 else WHITE)
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            if ci > 0 and re.fullmatch(r"[\d.,%/ =()\-A-Za-z]*\d[\d.,%/ =()\-A-Za-z]*", row[ci]) and nc >= 4:
                p.alignment = PP_ALIGN.CENTER
            add_runs(p, row[ci], size, WHITE if ri == 0 else INK, bold=(ri == 0 or ci == 0))
    return gs


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = " ".join(s["notes"])


# ----------------------------------------------------------------- build
def build():
    res = load_results()
    slides = parse_slides(res)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    blank = prs.slide_layouts[6]
    missing_figs = []
    for idx, s in enumerate(slides, 1):
        sl = prs.slides.add_slide(blank)
        lay = s["layout"]
        if lay in ("title", "closing"):
            bg(sl, NAVY)
            if lay == "title":
                add_text(sl, 0.8, 0.6, 11.7, 1.7, s["title"], 40, WHITE, True, HEAD_FONT,
                         anchor=MSO_ANCHOR.MIDDLE, name="Title")
                add_text(sl, 0.8, 2.25, 11.7, 0.6, s.get("subtitle", ""), 22, RGBColor(0xBF, 0xDD, 0xE6),
                         italic=True, name="Subtitle")
                tb = sl.shapes.add_textbox(Inches(0.8), Inches(3.05), Inches(11.7), Inches(2.0))
                tb.name = "Student"
                tf = tb.text_frame; tf.word_wrap = True
                for i, (t, b) in enumerate(AUTHOR):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    r = p.add_run(); r.text = t; set_run(r, 24 if b else 18, WHITE, b)
                tb = sl.shapes.add_textbox(Inches(0.8), Inches(5.2), Inches(11.7), Inches(1.7))
                tb.name = "Guide"
                tf = tb.text_frame; tf.word_wrap = True
                for i, (t, b) in enumerate(GUIDE):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    r = p.add_run(); r.text = t; set_run(r, 16, RGBColor(0xDD, 0xE8, 0xEE), b)
            else:
                add_text(sl, 0.8, 2.3, 11.7, 1.4, s["title"], 54, WHITE, True, HEAD_FONT,
                         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name="Title")
                add_text(sl, 0.8, 3.8, 11.7, 0.8, s.get("subtitle", ""), 28, RGBColor(0xBF, 0xDD, 0xE6),
                         align=PP_ALIGN.CENTER, name="Subtitle")
            footer(sl, idx, dark=True)
            notes(sl, s)
            continue

        bg(sl, WHITE)
        title_bar(sl, s["title"])
        top, bot = 1.3, 6.9
        H = bot - top
        cap = s.get("caption", "")
        if lay == "bullets":
            add_bullets(sl, 0.6, top + 0.1, 12.1, H - 0.1, s["bullets"])
        elif lay == "refs":
            half = math.ceil(len(s["bullets"]) / 2)
            for k, part in enumerate((s["bullets"][:half], s["bullets"][half:])):
                tb = sl.shapes.add_textbox(Inches(0.6 + 6.15 * k), Inches(top + 0.1), Inches(5.95), Inches(H - 0.6))
                tb.name = "References %d" % (k + 1)
                tf = tb.text_frame; tf.word_wrap = True
                for i, t in enumerate(part):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    p.space_after = Pt(4)
                    key, _, rest = t.partition(": ")
                    r = p.add_run(); r.text = key + ": "; set_run(r, 14, TEAL, True)
                    r = p.add_run(); r.text = rest; set_run(r, 14, INK)
            add_text(sl, 0.6, 6.4, 12.1, 0.4, "Full verified list: docs/manuscripts/shared/references.bib", 14, GREY, italic=True)
        elif lay == "fig":
            extra = 0.55 if s["bullets"] else 0
            place_picture(sl, s["figure"], 0.6, top, 12.1, H - 0.5 - extra, missing_figs)
            add_text(sl, 0.6, bot - 0.45 - extra, 12.1, 0.4, cap, 14, GREY, italic=True, align=PP_ALIGN.CENTER, name="Caption")
            if s["bullets"]:
                add_text(sl, 0.6, bot - 0.5, 12.1, 0.5, s["bullets"][0], 18, INK, name="Body")
        elif lay == "figs2":
            a, b = [t.strip() for t in s["figure"].split(",")]
            place_picture(sl, a, 0.5, top, 6.1, 4.55, missing_figs)
            place_picture(sl, b, 6.75, top, 6.1, 4.55, missing_figs)
            add_text(sl, 0.6, 5.9 - 0.0, 12.1, 0.4, cap, 14, GREY, italic=True, align=PP_ALIGN.CENTER, name="Caption")
            if s["bullets"]:
                add_text(sl, 0.6, 6.3, 12.1, 0.55, s["bullets"][0], 18, INK, name="Body")
        elif lay == "table":
            nrows = len(s["table"])
            colw = [float(c) for c in s["colw"].split(",")] if s.get("colw") else None
            add_table(sl, s["table"], 0.6, top + 0.1, 12.1, colw)
            if s["bullets"]:
                th = (0.5 if nrows <= 7 else 0.42) * nrows
                y = min(top + 0.1 + th + 0.35, 5.6)
                add_bullets(sl, 0.6, y, 12.1, bot - y, s["bullets"])
        elif lay == "tables2":
            colw = [float(c) for c in s["colw"].split(",")] if s.get("colw") else None
            add_table(sl, s["table"], 0.5, top + 0.1, 6.1, colw)
            add_table(sl, s["table2"], 6.8, top + 0.1, 6.0, colw)
            if s["bullets"]:
                y = top + 0.1 + 0.5 * len(s["table"]) + 0.3
                add_bullets(sl, 0.5, y, 12.3, bot - y, s["bullets"])
        elif lay == "table_fig":
            colw = [float(c) for c in s["colw"].split(",")] if s.get("colw") else None
            add_table(sl, s["table"], 0.5, top + 0.1, 6.4, colw)
            place_picture(sl, s["figure"], 7.1, top, 5.8, 4.7, missing_figs)
            add_text(sl, 7.1, top + 4.75, 5.8, 0.4, cap, 14, GREY, italic=True, align=PP_ALIGN.CENTER, name="Caption")
            if s["bullets"]:
                add_bullets(sl, 0.5, 5.4, 12.3, 1.5, s["bullets"])
        elif lay == "placeholder":
            box = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(top + 0.1), Inches(11.7), Inches(4.6))
            box.name = "Demo placeholder"
            box.fill.solid(); box.fill.fore_color.rgb = LIGHT
            box.line.color.rgb = GREY; box.line.dash_style = 4; box.line.width = Pt(1.5)
            tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = "[PENDING: " + cap + "]"; set_run(r, 22, RED, True)
        footer(sl, idx)
        notes(sl, s)
    prs.core_properties.title = "LungAI DenseNet-121: Six-Class Chest X-Ray Classifier (M.Tech presentation)"
    prs.core_properties.author = "Katkam Sathwik"
    prs.save(OUT)
    print("saved", OUT, "slides:", len(slides))
    if MISSING:
        print("UNRESOLVED result keys:", sorted(MISSING))
    print("missing figures:", sorted(set(missing_figs)))


# ----------------------------------------------------------------- verify
def verify():
    prs = Presentation(OUT)
    print("slide count:", len(prs.slides))
    issues, pend = [], []
    for i, sl in enumerate(prs.slides, 1):
        title = next((sh.text_frame.text for sh in sl.shapes if sh.name == "Title"), "")
        print("%2d  %s" % (i, title))
        if not sl.has_notes_slide or not sl.notes_slide.notes_text_frame.text.strip():
            issues.append("slide %d: no speaker notes" % i)
        for sh in sl.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                tf = sh.text_frame
                w = Emu(sh.width).inches - 0.15
                need = 0.0
                for p in tf.paragraphs:
                    txt = "".join(r.text for r in p.runs)
                    sz = max([r.font.size.pt for r in p.runs if r.font.size] or [18])
                    sp = (p.space_after.pt if p.space_after else 0)
                    ind = 0.3 if p._p.pPr is not None and p._p.pPr.get("marL") else 0
                    need += est_lines(txt, w - ind, sz) * sz * 1.2 + sp
                have = Emu(sh.height).inches * 72
                if need > have * 1.02 and sh.name not in ("Footer",):
                    issues.append("slide %d: '%s' may overflow (need %.0fpt, have %.0fpt)" % (i, sh.name, need, have))
                for m in PENDING_RE.finditer(tf.text):
                    pend.append((i, m.group(0)))
                if tf.text.strip().startswith("[PENDING"):
                    pend.append((i, tf.text.strip()[:60]))
            if sh.has_table:
                for r in sh.table.rows:
                    for c in r.cells:
                        for m in PENDING_RE.finditer(c.text_frame.text):
                            pend.append((i, m.group(0)))
                est = sum(Emu(r.height).inches for r in sh.table.rows)
                if Emu(sh.top).inches + est > 6.95:
                    issues.append("slide %d: table may run into footer" % i)
    print("\nOverflow / structure issues:", len(issues))
    for x in issues:
        print("  -", x)
    print("\nPending markers (slide, text):", len(pend))
    seen = set()
    for x in pend:
        if x not in seen:
            seen.add(x); print("  ", x)


if __name__ == "__main__":
    if "--verify-only" not in sys.argv:
        build()
    verify()
