#!/usr/bin/env python
"""Build the LungAI M.Tech project report (.docx) from Markdown sources.

Usage (from anywhere):
    D:\\LungAI\\.venv-audit\\Scripts\\python.exe docs\\thesis\\build_report.py [--no-update-prompt]

Inputs
    docs/thesis/content/*.md                  one file per chapter / front-matter section
    docs/manuscripts/shared/results.json      result tokens  {{TestAcc}}
    docs/manuscripts/shared/references.bib    bibliography   [@key]
    docs/figures/<stem>.png                   figures        ![stem](Caption)
Output
    docs/thesis/LungAI_MTech_Project_Report.docx

Markdown dialect (small, deliberate):
    # Title                 chapter (CHAPTER n) / appendix (APPENDIX A) / unnumbered front-matter heading
    ## Section              numbered n.m ;  ### Subsection  n.m.k
    paragraphs, "- " bullets, "1. " numbered items, ``` code blocks
    Table: Caption {#tab:id}   (line immediately above a pipe table)
    ![fig_stem](Caption) {#fig:id}
    [[tab:id]] [[fig:id]]  cross references  ->  "Table 5.2"
    [@key] or [@a; @b]     IEEE citations
    {{Key}}                value from results.json
    **bold** *italic* `code`
    [PENDING FINAL RESULTS: ...], [TO BE PROVIDED], [STUDENT TO PERSONALISE], [AUTHORS TO CONFIRM] are highlighted
    @center[size=16,bold]: text   @left: text   @spacer: 24   @sig: Label   @pagebreak
"""
import json
import re
import sys
import unicodedata
from collections import OrderedDict
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
CONTENT = HERE / "content"
FIGS = DOCS / "figures"
SHARED = DOCS / "manuscripts" / "shared"
OUT = HERE / "LungAI_MTech_Project_Report.docx"

SHORT_TITLE = "LungAI: Six-Class Chest X-Ray Classifier | M.Tech Project Report"
FONT = "Times New Roman"
TEXT_WIDTH_IN = 8.27 - 1.5 - 1.0  # A4 width minus margins

# (file, kind) in document order.  kinds: cover, front, special:toc|lof|lot, chapter, refs, appendix
PLAN = [
    ("00_cover.md", "cover"),
    ("01_title.md", "cover"),
    ("02_certificate.md", "front"),
    ("03_declaration.md", "front"),
    ("04_supervisors.md", "front"),
    ("05_acknowledgement.md", "front"),
    ("06_abstract.md", "front"),
    (None, "toc"),
    (None, "lof"),
    (None, "lot"),
    ("07_abbreviations.md", "front"),
    ("10_ch1_introduction.md", "chapter"),
    ("11_ch2_literature_review.md", "chapter"),
    ("12_ch3_problem_objectives.md", "chapter"),
    ("13_ch4_methodology.md", "chapter"),
    ("14_ch5_dataset.md", "chapter"),
    ("15_ch6_implementation.md", "chapter"),
    ("16_ch7_experiments_results.md", "chapter"),
    ("17_ch8_validation_discussion.md", "chapter"),
    ("18_ch9_conclusion.md", "chapter"),
    (None, "refs"),
    ("30_appendices.md", "appendix"),
]

MARKER_RE = r"\[(?:PENDING[^\]]*|TO BE PROVIDED|TO BE CONFIRMED|STUDENT TO PERSONALISE[^\]]*|AUTHORS TO CONFIRM[^\]]*|Figure pending[^\]]*|CHECK[^\]]*)\]"
INLINE_RE = re.compile(
    r"(?P<bold>\*\*(?P<b>.+?)\*\*)"
    r"|(?P<ital>\*(?!\s)(?P<i>.+?)(?<!\s)\*)"
    r"|(?P<code>`(?P<c>[^`]+)`)"
    r"|(?P<cite>\[@(?P<keys>[^\]]+)\])"
    r"|(?P<tok>\{\{(?P<t>\w+)\}\})"
    r"|(?P<ref>\[\[(?P<r>[\w:\-]+)\]\])"
    r"|(?P<mark>" + MARKER_RE + r")"
)

LOG = {"markers": [], "missing_tokens": set(), "missing_cites": set(), "missing_figs": [], "figs": [],
       "refs_notes": [], "uncited_bib": []}

# ----------------------------------------------------------------------------- results tokens
try:
    RESULTS = {k: v for k, v in json.load(open(SHARED / "results.json", encoding="utf-8")).items()
               if not k.startswith("_")}
except Exception as e:  # noqa
    RESULTS = {}
    print("WARNING: results.json unreadable:", e)


# ----------------------------------------------------------------------------- CXR-gate tokens (artifacts/app_checks)
def load_gate_tokens():
    ck = DOCS.parent / "artifacts" / "app_checks"
    try:
        c = json.load(open(ck / "cxr_gate_calibration_v6_build_002.json", encoding="utf-8"))
        r = json.load(open(ck / "cxr_gate_rejection_v6_build_001.json", encoding="utf-8"))
    except Exception as e:  # noqa
        print("WARNING: gate artifacts unreadable:", e)
        return
    pct = lambda x: "%.1f" % (100 * x)
    t = RESULTS
    t["GateN"] = "{:,}".format(c["n"])
    t["GateStage1Rej"] = str(c["stage1_rejected"])
    t["GateTarget"] = "%d" % round(100 * c["target_pass_rate"])
    t["GateQuantile"] = "%.3f" % c["threshold_from_quantile"]
    t["GateHighNeg"] = "%.4f" % c["highest_negative_score_passing_stage1"]
    t["GateThr"] = "%.3f" % c["heuristic_threshold"]
    t["GatePassNew"] = pct(c["val_pass_rate_at_threshold"])
    t["GatePassOld"] = pct(c["val_pass_rate_at_0.83"])
    t["GateOldThr"] = "0.83"
    t["GateNeg"] = str(len(c["synthetic_negatives"]))
    t["GateNegRej"] = str(sum(1 for v in c["synthetic_negatives"].values() if v["rejected_at_threshold"]))
    for k, v in c["val_pass_rate_by_threshold"].items():
        t["GatePass_" + k.replace(".", "")] = pct(v)
    for k, v in c["by_source"].items():
        t["GateSrc_" + k] = pct(v["pass_at_threshold"])
        t["GateSrcN_" + k] = str(v["n"])
    for k, v in c["synthetic_negatives"].items():
        t["GateNegScore_" + k] = "%.3f" % v["score"]
    tt, ee = r["total"]["test"], r["total"]["external"]
    t["GateRejTestK"], t["GateRejTestN"], t["GateRejTestPct"] = "{:,}".format(tt["rejected"]), "{:,}".format(tt["n"]), pct(tt["rejected"] / tt["n"])
    t["GateRejExtK"], t["GateRejExtN"], t["GateRejExtPct"] = str(ee["rejected"]), str(ee["n"]), pct(ee["rejected"] / ee["n"])
    agg = {}
    for g in r["by_group"]:
        a = agg.setdefault(g["source"], [0, 0]) if g["split"] == "test" else None
        if a is not None:
            a[0] += g["n"]
            a[1] += g["rejected"]
    for k, (n, rej) in agg.items():
        t["GateRejSrc_" + k] = pct(rej / n)
    try:
        r2 = json.load(open(ck / "cxr_gate_rejection_v6_build_002_heuristic0703.json", encoding="utf-8"))
        a, b = r2["total"]["test"], r2["total"]["external"]
        t["GateNewRejTestK"], t["GateNewRejTestN"], t["GateNewRejTestPct"] = "{:,}".format(a["rejected"]), "{:,}".format(a["n"]), pct(a["rejected"] / a["n"])
        t["GateNewRejExtK"], t["GateNewRejExtN"], t["GateNewRejExtPct"] = str(b["rejected"]), str(b["n"]), pct(b["rejected"] / b["n"])
    except Exception as e:  # noqa
        print("WARNING: new gate rejection artifact unreadable:", e)


load_gate_tokens()

# ----------------------------------------------------------------------------- BibTeX + IEEE
ACCENTS = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302", "~": "\u0303", "=": "\u0304", ".": "\u0307"}
MONTHS = {"jan": "Jan.", "feb": "Feb.", "mar": "Mar.", "apr": "Apr.", "may": "May", "jun": "Jun.", "jul": "Jul.",
          "aug": "Aug.", "sep": "Sep.", "oct": "Oct.", "nov": "Nov.", "dec": "Dec."}
PARTICLES = {"van", "von", "de", "der", "del", "da", "di", "le", "la", "den", "dos", "du"}


def _unlatex(s, url=False):
    if url:
        return re.sub(r"\s+", "", s.replace("\\_", "_").replace("\\%", "%").replace("\\&", "&")).strip("{}")
    s = re.sub(r"\\([\'`\"^~=.])\s*\{?\\?(\w)\}?", lambda m: unicodedata.normalize("NFC", m.group(2) + ACCENTS[m.group(1)]), s)
    s = re.sub(r"\{\\c\s*(\w)\}|\\c\{(\w)\}|\\c\s+(\w)", lambda m: unicodedata.normalize("NFC", (m.group(1) or m.group(2) or m.group(3)) + "\u0327"), s)
    s = s.replace("\\ss", "\u00df").replace("{\\o}", "\u00f8").replace("\\o ", "\u00f8").replace("\\&", "&").replace("\\%", "%")
    s = s.replace("\\_", "_").replace("\\textendash", "\u2013").replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def parse_bib(text):
    entries = OrderedDict()
    pos = 0
    n = len(text)
    while True:
        m = re.compile(r"@(\w+)\s*[\{\(]").search(text, pos)
        if not m:
            break
        typ = m.group(1).lower()
        i = m.end()
        if typ in ("comment", "string", "preamble"):
            depth = 1
            while i < n and depth:
                depth += (text[i] == "{") - (text[i] == "}")
                i += 1
            pos = i
            continue
        j = text.index(",", i)
        key = text[i:j].strip()
        i = j + 1
        fields = {}
        while i < n:
            while i < n and text[i] in " \t\r\n,":
                i += 1
            if i >= n or text[i] in "})":
                i += 1
                break
            mm = re.compile(r"([\w\-]+)\s*=\s*").match(text, i)
            if not mm:
                i += 1
                continue
            name = mm.group(1).lower()
            i = mm.end()
            if text[i] == "{":
                depth, k = 1, i + 1
                while k < n and depth:
                    depth += (text[k] == "{") - (text[k] == "}")
                    k += 1
                val = text[i + 1:k - 1]
                i = k
            elif text[i] == '"':
                k, depth = i + 1, 0
                while k < n and not (text[k] == '"' and depth == 0):
                    depth += (text[k] == "{") - (text[k] == "}")
                    k += 1
                val = text[i + 1:k]
                i = k + 1
            else:
                k = i
                while k < n and text[k] not in ",}":
                    k += 1
                val = text[i:k].strip()
                i = k
            fields[name] = val
        pos = i
        entries[key] = (typ, fields)
    return entries


def _fmt_author(a):
    a = a.strip()
    if a.startswith("{") and a.endswith("}"):
        return _unlatex(a)
    a = _unlatex(a)
    if "," in a:
        last, first = [x.strip() for x in a.split(",", 1)]
    else:
        parts = a.split()
        if len(parts) == 1:
            return parts[0]
        k = len(parts) - 1
        while k > 1 and parts[k - 1].lower() in PARTICLES:
            k -= 1
        first, last = " ".join(parts[:k]), " ".join(parts[k:])
    inits = []
    for tok in re.split(r"\s+", first):
        if not tok:
            continue
        if re.fullmatch(r"(?:[A-Z]\.)+", tok):
            inits.append(tok)
        elif "-" in tok:
            inits.append("-".join(p[0] + "." for p in tok.split("-") if p))
        else:
            inits.append(tok[0] + ".")
    return (" ".join(inits) + " " + last).strip()


def _authors(raw):
    parts = re.split(r"\s+and\s+", raw.strip())
    out = [_fmt_author(p) for p in parts if p.strip() and p.strip() != "others"]
    et_al = "others" in [p.strip() for p in parts]
    if et_al or len(out) > 6:
        return out[0] + " et al."
    if len(out) == 1:
        return out[0]
    if len(out) == 2:
        return " and ".join(out)
    return ", ".join(out[:-1]) + ", and " + out[-1]


def ieee(entry):
    """Return list of (text, italic) segments for one BibTeX entry."""
    typ, f = entry
    g = lambda k: _unlatex(f[k]) if k in f and f[k].strip() else ""
    seg = []

    def add(t, it=False):
        if t:
            seg.append((t, it))

    auth = _authors(f.get("author") or f.get("editor", "")) if (f.get("author") or f.get("editor")) else ""
    title = g("title").rstrip(".")
    year, month = g("year"), MONTHS.get(g("month").lower()[:3], g("month"))
    pages = g("pages").replace("-", "\u2013").replace("\u2013\u2013", "\u2013")
    pp = ("pp. " + pages) if "\u2013" in pages else (("p. " + pages) if pages else "")
    date = (month + " " + year).strip()
    if typ == "article":
        add(auth + ", " if auth else "")
        add('\u201c' + title + ',\u201d ')
        add(g("journal") or g("journaltitle"), True)
        for lab, k in (("vol. ", "volume"), ("no. ", "number")):
            if g(k):
                add(", " + lab + g(k))
        if g("articleno") or g("eid"):
            add(", Art. no. " + (g("articleno") or g("eid")))
        if pp:
            add(", " + pp)
        add(", " + date if date else "")
    elif typ in ("inproceedings", "conference"):
        add(auth + ", " if auth else "")
        add('\u201c' + title + ',\u201d in ')
        add(g("booktitle"), True)
        if g("address") or g("location"):
            add(", " + (g("address") or g("location")))
        add(", " + date if date else "")
        if pp:
            add(", " + pp)
    elif typ in ("book", "phdthesis", "mastersthesis"):
        add(auth + ", " if auth else "")
        add(title, True)
        pub = ", ".join(x for x in [g("publisher") or g("school"), ] if x)
        add(". " + ((g("address") + ": ") if g("address") else "") + pub if pub else "")
        add(", " + year if year else "")
    elif typ in ("incollection", "inbook"):
        add(auth + ", " if auth else "")
        add('\u201c' + title + ',\u201d in ')
        add(g("booktitle"), True)
        if g("publisher"):
            add(", " + ((g("address") + ": ") if g("address") else "") + g("publisher"))
        add(", " + year if year else "")
        if pp:
            add(", " + pp)
    else:  # misc, techreport, unpublished ...
        add(auth + ", " if auth else "")
        add('\u201c' + title + ',\u201d ')
        arx = g("eprint")
        if arx and "arxiv" in (g("archiveprefix") + g("eprinttype") + g("journal") + g("howpublished") + g("url")).lower() + "arxiv" * (not g("journal")):
            add("arXiv preprint arXiv:" + arx)
        else:
            add(g("howpublished") or g("journal") or g("institution") or g("publisher") or g("note"))
        add(", " + date if date else "")
    text = "".join(s for s, _ in seg).rstrip(", ")
    # tidy trailing comma then add final period
    if seg:
        while seg and seg[-1][0].rstrip().endswith(","):
            seg[-1] = (seg[-1][0].rstrip()[:-1], seg[-1][1])
    if g("doi"):
        add(", doi: " + _unlatex(f["doi"], url=True))
    elif g("url"):
        add(". [Online]. Available: " + _unlatex(f["url"], url=True))
        add("")
    add(".")
    note = g("note")
    if note and re.search(r"unverif|could not|verify|check|not verified|unresolved", note, re.I):
        add(" [CHECK: " + note + "]")
        LOG["refs_notes"].append(note)
    return seg



# ----------------------------------------------------------------------------- generated tables from build reports
BUILD_DIR = DOCS.parent / "data" / "derived"
CLASSES = ["COVID-19", "Normal", "Pleural Effusion", "Pneumonia", "Pulmonary Nodule / Mass", "Tuberculosis"]
BUILDS = ["v6_build_001", "v6_build_002"]
_BR = {}


def build_report(b):
    if b not in _BR:
        try:
            _BR[b] = json.load(open(BUILD_DIR / b / "v6_build_report.json", encoding="utf-8"))
        except Exception as e:  # noqa
            LOG["markers"].append(("gen", "[PENDING: build report %s unreadable]" % b))
            _BR[b] = None
    return _BR[b]


def _n(x):
    return "{:,}".format(x) if isinstance(x, int) else str(x)


def gen_rows(kind):
    r1, r2 = build_report(BUILDS[0]), build_report(BUILDS[1])
    if not r1 or not r2:
        return [["[PENDING: build reports not found]"]]
    if kind == "composition":
        rows = [["Class", "Run 1 train", "Run 1 val", "Run 1 test", "Run 2 train", "Run 2 val", "Run 2 test"]]
        tot = [0] * 6
        for c in CLASSES:
            row = [c]
            for k, r in enumerate((r1, r2)):
                for j, sp in enumerate(("train", "val", "test")):
                    v = r["counts"][sp].get(c, 0)
                    tot[k * 3 + j] += v
                    row.append(_n(v))
            rows.append(row)
        rows.append(["**Total**"] + ["**%s**" % _n(t) for t in tot])
        return rows
    if kind == "sources":
        srcs = ["nih_cxr14", "kermany", "shenzhen", "cohen_covid", "covid_radiography", "tbx11k", "montgomery"]
        rows = [["Source", "Run 1 train / val / test", "Run 2 train / val / test", "External", "Excluded (run 2)"]]
        for s_ in srcs:
            def g(r, sp):
                return r["by_source_split"].get("%s|%s" % (s_, sp), 0)
            a = " / ".join(_n(g(r1, sp)) for sp in ("train", "val", "test"))
            b = " / ".join(_n(g(r2, sp)) for sp in ("train", "val", "test"))
            rows.append([s_, a if any(g(r1, sp) for sp in ("train", "val", "test")) else "not used", b if any(g(r2, sp) for sp in ("train", "val", "test")) else "n/a",
                         _n(g(r2, "external")), _n(g(r2, "excluded"))])
        return rows
    if kind == "excl":
        keys = sorted(set(r1["exclusions"]) | set(r2["exclusions"]), key=lambda k: -r2["exclusions"].get(k, 0))
        rows = [["Reason recorded in the build report", "Run 1", "Run 2"]]
        for k in keys:
            rows.append([k.replace("_", " "), _n(r1["exclusions"].get(k, 0)), _n(r2["exclusions"].get(k, 0))])
        return rows
    if kind == "checks":
        rows = [["Check or statistic", "Run 1", "Run 2"]]
        for lab, key in (("Patient groups spanning splits", "groups_spanning_splits"), ("Identical files (MD5) spanning splits", "md5_spanning_splits"),
                         ("External-cohort files (MD5) present in development data", "external_md5_in_development")):
            rows.append([lab, _n(r1["leakage"][key]), _n(r2["leakage"][key])])
        rows.append(["Near-duplicate pairs merged into one split group", _n(r1["near_duplicate_pairs"]), _n(r2["near_duplicate_pairs"])])
        for lab, key in (("Development rows with genuine patient identity", "genuine"), ("Development rows with identity derived from file name", "derived_from_filename"),
                         ("Development rows with unknown identity (grouped per image)", "unknown")):
            rows.append([lab, _n(r1["patient_identity"].get(key, 0)), _n(r2["patient_identity"].get(key, 0))])
        rows.append(["Split fingerprint (first 12 characters of SHA-256)", r1["split_fingerprint"][:12], r2["split_fingerprint"][:12]])
        rows.append(["Manifest SHA-256 (first 12 characters)", r1["manifest_sha256"][:12], r2["manifest_sha256"][:12]])
        return rows
    return [["unknown generator " + kind]]


# ----------------------------------------------------------------------------- Markdown parsing
def parse_md(text):
    """Return list of blocks."""
    blocks = []
    lines = text.splitlines()
    i = 0
    pending_caption = None
    para = []

    def flush():
        nonlocal para
        if para:
            blocks.append(("p", " ".join(s.strip() for s in para)))
            para = []

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("<!--"):
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        if not s:
            flush()
            i += 1
            continue
        if s.startswith("```"):
            flush()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip())
                i += 1
            i += 1
            blocks.append(("code", code))
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            flush()
            blocks.append(("h%d" % len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        m = re.match(r"^@(\w+)(?:\[([^\]]*)\])?:?\s*(.*)$", s)
        if m and m.group(1) in ("center", "left", "spacer", "sig", "pagebreak", "right", "gen"):
            flush()
            if m.group(1) == "gen":
                kind_, _, cap_ = m.group(3).partition("|")
                cap_ = cap_.strip()
                lab_ = None
                mm_ = re.search(r"\s*\{#([\w:\-]+)\}\s*$", cap_)
                if mm_:
                    lab_, cap_ = mm_.group(1), cap_[:mm_.start()]
                blocks.append(("table", cap_, lab_, gen_rows(kind_.strip())))
                i += 1
                continue
            blocks.append(("dir", m.group(1), m.group(2) or "", m.group(3)))
            i += 1
            continue
        if s.startswith("Table:"):
            flush()
            pending_caption = s[6:].strip()
            i += 1
            continue
        m = re.match(r"^!\[([\w\-]+)\]\((.*)\)\s*(\{#[\w:\-]+\})?\s*$", s)
        if m:
            flush()
            lab = m.group(3)[2:-1] if m.group(3) else None
            blocks.append(("fig", m.group(1), m.group(2).strip(), lab))
            i += 1
            continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = lines[i].strip().strip("|")
                cells = [c.strip() for c in re.split(r"(?<!\\)\|", r)]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append([c.replace("\\|", "|") for c in cells])
                i += 1
            cap, lab = pending_caption or "", None
            mm = re.search(r"\s*\{#([\w:\-]+)\}\s*$", cap)
            if mm:
                lab, cap = mm.group(1), cap[:mm.start()]
            pending_caption = None
            blocks.append(("table", cap, lab, rows))
            continue
        if re.match(r"^[-*]\s+", s):
            flush()
            items = []
            while i < len(lines) and (re.match(r"^\s*[-*]\s+", lines[i]) or (lines[i].startswith("  ") and lines[i].strip() and items)):
                if re.match(r"^\s*[-*]\s+", lines[i]):
                    items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]).strip())
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            blocks.append(("ul", items))
            continue
        if re.match(r"^\d+\.\s+", s):
            flush()
            items = []
            while i < len(lines) and (re.match(r"^\s*\d+\.\s+", lines[i]) or (lines[i].startswith("  ") and lines[i].strip() and items)):
                if re.match(r"^\s*\d+\.\s+", lines[i]):
                    items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]).strip())
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            blocks.append(("ol", items))
            continue
        para.append(ln)
        i += 1
    flush()
    return blocks


def inline_texts(block):
    k = block[0]
    if k in ("p", "h1", "h2", "h3"):
        yield block[1]
    elif k in ("ul", "ol"):
        yield from block[1]
    elif k == "table":
        yield block[1]
        for r in block[3]:
            yield from r
    elif k == "fig":
        yield block[2]
    elif k == "dir":
        yield block[3]


# ----------------------------------------------------------------------------- pre-scan
class Plan:
    def __init__(self):
        self.files = []          # (kind, blocks)
        self.labels = {}         # label -> "5.2"
        self.tables = []         # (num, caption)
        self.figures = []        # (num, caption)
        self.toc = []            # (level, text)
        self.cite_order = []     # keys in first-citation order
        self.warn = []

    def scan(self):
        chap = 0
        app = 0
        for fname, kind in PLAN:
            if fname is None:
                self.files.append((kind, None, None))
                continue
            p = CONTENT / fname
            if not p.exists():
                self.warn.append("missing content file: " + fname)
                self.files.append((kind, fname, []))
                continue
            blocks = parse_md(p.read_text(encoding="utf-8"))
            self.files.append((kind, fname, blocks))
            prefix = ""
            sec = sub = 0
            tcount = fcount = 0
            for b in blocks:
                if b[0] == "h1":
                    if kind == "chapter":
                        chap += 1
                        prefix = str(chap)
                        self.toc.append((1, "CHAPTER %d %s" % (chap, b[1].upper())))
                    elif kind == "appendix":
                        app += 1
                        prefix = chr(64 + app)
                        self.toc.append((1, "APPENDIX %s %s" % (prefix, b[1].upper())))
                    else:
                        self.toc.append((1, b[1].upper()))
                    sec = sub = 0
                    tcount = fcount = 0
                elif b[0] == "h2":
                    sec += 1
                    sub = 0
                    lab = ("%s.%d " % (prefix, sec)) if prefix else ""
                    self.toc.append((2, lab + b[1]))
                elif b[0] == "h3":
                    sub += 1
                    lab = ("%s.%d.%d " % (prefix, sec, sub)) if prefix else ""
                    self.toc.append((3, lab + b[1]))
                elif b[0] == "table" and b[1]:
                    tcount += 1
                    num = "%s.%d" % (prefix, tcount)
                    if b[2]:
                        self.labels[b[2]] = "Table " + num
                    self.tables.append((num, b[1]))
                elif b[0] == "fig":
                    fcount += 1
                    num = "%s.%d" % (prefix, fcount)
                    if b[3]:
                        self.labels[b[3]] = "Figure " + num
                    self.figures.append((num, b[2]))
                for t in inline_texts(b):
                    for m in re.finditer(r"\[@([^\]]+)\]", t):
                        for k in re.split(r"[;,]\s*", m.group(1)):
                            k = k.strip().lstrip("@")
                            if k and k not in self.cite_order:
                                self.cite_order.append(k)


# ----------------------------------------------------------------------------- docx helpers
def set_style_font(style, size=None, bold=None, italic=None, color=RGBColor(0, 0, 0), name=FONT):
    rpr = style.element.get_or_add_rPr()
    for old in rpr.findall(qn("w:rFonts")):
        rpr.remove(old)
    rf = OxmlElement("w:rFonts")
    for a in ("ascii", "hAnsi", "eastAsia", "cs"):
        rf.set(qn("w:" + a), name)
    rpr.insert(0, rf)
    if size:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color is not None:
        style.font.color.rgb = color


def fix_run_font(run, name=FONT):
    rpr = run._element.get_or_add_rPr()
    for old in rpr.findall(qn("w:rFonts")):
        rpr.remove(old)
    rf = OxmlElement("w:rFonts")
    for a in ("ascii", "hAnsi", "eastAsia", "cs"):
        rf.set(qn("w:" + a), name)
    rpr.insert(0, rf)


def add_run(par, text, size=None, bold=None, italic=None, mono=False, highlight=False, color=None):
    r = par.add_run(text)
    fix_run_font(r, "Courier New" if mono else FONT)
    if size:
        r.font.size = Pt(size)
    if bold:
        r.font.bold = True
    if italic:
        r.font.italic = True
    if highlight:
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    if color:
        r.font.color.rgb = color
    return r


def add_field(par, instr, result="", size=None, bold=None):
    def mk(kind):
        r = par.add_run()
        fix_run_font(r)
        if size:
            r.font.size = Pt(size)
        if bold:
            r.font.bold = True
        fc = OxmlElement("w:fldChar")
        fc.set(qn("w:fldCharType"), kind)
        r._element.append(fc)
        return r

    mk("begin")
    r = par.add_run()
    fix_run_font(r)
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = " %s " % instr
    r._element.append(it)
    mk("separate")
    add_run(par, result, size=size, bold=bold)
    mk("end")


def format_cites(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append("[%d]\u2013[%d]" % (nums[i], nums[j]))
        else:
            out.extend("[%d]" % n for n in nums[i:j + 1])
        i = j + 1
    return ", ".join(out)


class Builder:
    def __init__(self, plan, bib):
        self.plan = plan
        self.bib = bib
        self.cite_num = {k: i + 1 for i, k in enumerate(plan.cite_order)}
        self.doc = Document()
        self.ctx = ""
        self.init_doc()

    # ---- setup
    def init_doc(self):
        d = self.doc
        sec = d.sections[0]
        sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
        sec.left_margin, sec.right_margin = Inches(1.5), Inches(1.0)
        sec.top_margin = sec.bottom_margin = Inches(1.0)
        sec.header_distance = sec.footer_distance = Inches(0.5)
        st = d.styles
        n = st["Normal"]
        set_style_font(n, 12)
        pf = n.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(6)
        pf.space_before = Pt(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for name, size, align in (("Heading 1", 16, WD_ALIGN_PARAGRAPH.CENTER), ("Heading 2", 14, WD_ALIGN_PARAGRAPH.LEFT),
                                  ("Heading 3", 12, WD_ALIGN_PARAGRAPH.LEFT)):
            h = st[name]
            set_style_font(h, size, bold=True, italic=False)
            h.paragraph_format.alignment = align
            h.paragraph_format.line_spacing = 1.15
            h.paragraph_format.keep_with_next = True
            h.paragraph_format.space_before = Pt(0 if name == "Heading 1" else 12)
            h.paragraph_format.space_after = Pt(12 if name == "Heading 1" else 6)
            h.paragraph_format.page_break_before = (name == "Heading 1")
        fh = st.add_style("FrontHeading", WD_STYLE_TYPE.PARAGRAPH)
        fh.base_style = n
        set_style_font(fh, 16, bold=True)
        fh.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fh.paragraph_format.page_break_before = True
        fh.paragraph_format.space_after = Pt(12)
        fh.paragraph_format.keep_with_next = True
        cap = st["Caption"]
        set_style_font(cap, 11, bold=True, italic=False)
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.line_spacing = 1.0
        cap.paragraph_format.space_before = Pt(6)
        cap.paragraph_format.space_after = Pt(6)
        for nm in ("List Bullet",):
            set_style_font(st[nm], 12)
            st[nm].paragraph_format.line_spacing = 1.5
            st[nm].paragraph_format.space_after = Pt(3)
            st[nm].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for nm in ("Header", "Footer"):
            set_style_font(st[nm], 10)
        # settings: ask Word to refresh fields on open
        if "--no-update-prompt" not in sys.argv:
            s = d.settings.element
            uf = OxmlElement("w:updateFields")
            uf.set(qn("w:val"), "true")
            s.append(uf)
        cp = d.core_properties
        cp.author = "Katkam Sathwik"
        cp.title = "LungAI: M.Tech Project Report"
        cp.subject = "Six-class chest X-ray classification with DenseNet-121"
        cp.keywords = "chest X-ray; DenseNet-121; Grad-CAM; patient-grouped split; M.Tech"

    # ---- inline
    def inline(self, par, text, size=None, bold=False, italic=False):
        pos = 0
        for m in INLINE_RE.finditer(text):
            if m.start() > pos:
                add_run(par, text[pos:m.start()], size, bold, italic)
            pos = m.end()
            if m.group("bold"):
                self.inline(par, m.group("b"), size, True, italic)
            elif m.group("ital"):
                self.inline(par, m.group("i"), size, bold, True)
            elif m.group("code"):
                add_run(par, m.group("c"), (size or 12) - 1.5, bold, italic, mono=True)
            elif m.group("cite"):
                keys = [k.strip().lstrip("@") for k in re.split(r"[;,]\s*", m.group("keys"))]
                nums, bad = [], []
                for k in keys:
                    if k in self.bib:
                        nums.append(self.cite_num[k])
                    else:
                        bad.append(k)
                        LOG["missing_cites"].add(k)
                if nums:
                    add_run(par, format_cites(nums), size, bold, italic)
                for k in bad:
                    add_run(par, "[?%s]" % k, size, bold, italic, highlight=True)
            elif m.group("tok"):
                k = m.group("t")
                if k in RESULTS:
                    add_run(par, str(RESULTS[k]), size, bold, italic)
                else:
                    LOG["missing_tokens"].add(k)
                    add_run(par, "[MISSING: %s]" % k, size, bold, italic, highlight=True)
            elif m.group("ref"):
                lab = m.group("r")
                if lab in self.plan.labels:
                    add_run(par, self.plan.labels[lab], size, bold, italic)
                else:
                    LOG["missing_tokens"].add("ref:" + lab)
                    add_run(par, "[?%s]" % lab, size, bold, italic, highlight=True)
            elif m.group("mark"):
                LOG["markers"].append((self.ctx, m.group("mark")))
                add_run(par, m.group("mark"), size, bold, italic, highlight=True)
        if pos < len(text):
            add_run(par, text[pos:], size, bold, italic)

    # ---- block writers
    def para(self, text, style=None, align=None, size=None, bold=False, spacing=None, after=None, keep_next=False):
        p = self.doc.add_paragraph(style=style)
        if align is not None:
            p.alignment = align
        if spacing is not None:
            p.paragraph_format.line_spacing = spacing
        if after is not None:
            p.paragraph_format.space_after = Pt(after)
        if keep_next:
            p.paragraph_format.keep_with_next = True
        if text:
            self.inline(p, text, size, bold)
        return p

    def list_block(self, kind, items):
        for n, it in enumerate(items, 1):
            if kind == "ul":
                p = self.doc.add_paragraph(style="List Bullet")
            else:
                p = self.doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.5)
                p.paragraph_format.first_line_indent = Inches(-0.3)
                p.paragraph_format.space_after = Pt(3)
                add_run(p, "%d.\u00a0\u00a0" % n)
            self.inline(p, it)

    def code_block(self, lines):
        for ln in lines:
            p = self.doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.left_indent = Inches(0.2)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            sh = OxmlElement("w:shd")
            sh.set(qn("w:val"), "clear")
            sh.set(qn("w:color"), "auto")
            sh.set(qn("w:fill"), "F2F2F2")
            p._element.get_or_add_pPr().append(sh)
            add_run(p, ln if ln else " ", 9, mono=True)
        self.para("", after=4, spacing=1.0)

    def caption(self, kind, num, text):
        """kind 'Table'|'Figure'; num like '5.2'; real SEQ field with per-chapter reset."""
        prefix, idx = num.rsplit(".", 1)
        p = self.doc.add_paragraph(style="Caption")
        add_run(p, "%s %s." % (kind, prefix), 11, bold=True)
        instr = "SEQ %s \\* ARABIC" % kind + (" \\r 1" if idx == "1" else "")
        add_field(p, instr, idx, 11, True)
        add_run(p, ". ", 11, bold=True)
        self.inline(p, text, 11, bold=False)
        return p

    def table_block(self, b):
        _, cap, lab, rows = b
        if cap:
            num = self.table_nums.pop(0)
            cp = self.caption("Table", num, cap)
            cp.paragraph_format.keep_with_next = True
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        t = self.doc.add_table(rows=len(rows), cols=ncol)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        wts = []
        for c in range(ncol):
            L = max(len(re.sub(r"[*`]", "", rows[r][c])) for r in range(len(rows)))
            wts.append(min(max(L, 6), 34))
        tot = sum(wts)
        widths = [Inches(TEXT_WIDTH_IN * w / tot) for w in wts]
        numeric = []
        for c in range(ncol):
            body = [rows[r][c] for r in range(1, len(rows)) if rows[r][c]]
            numeric.append(bool(body) and all(re.fullmatch(r"[\s\d.,%\u00b1()/\u2013\-+{}A-Za-z]*", x) and re.search(r"\d|\{\{", x) and len(x) < 22 for x in body))
        for ri, row in enumerate(rows):
            tr = t.rows[ri]
            trpr = tr._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trpr.append(cs)
            if ri == 0:
                th = OxmlElement("w:tblHeader")
                trpr.append(th)
            for ci, txt in enumerate(row):
                cell = tr.cells[ci]
                cell.width = widths[ci]
                p = cell.paragraphs[0]
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(2)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (numeric[ci] and ri > 0) or ri == 0 else WD_ALIGN_PARAGRAPH.LEFT
                parts = txt.split("<br>")
                for k, part in enumerate(parts):
                    if k:
                        p.add_run().add_break()
                    self.inline(p, part, 10, bold=(ri == 0))
                if ri == 0:
                    tcpr = cell._tc.get_or_add_tcPr()
                    sh = OxmlElement("w:shd")
                    sh.set(qn("w:val"), "clear")
                    sh.set(qn("w:color"), "auto")
                    sh.set(qn("w:fill"), "D9D9D9")
                    tcpr.append(sh)
        self.para("", after=4, spacing=1.0)

    def figure_block(self, b):
        _, stem, cap, lab = b
        num = self.fig_nums.pop(0)
        png = FIGS / (stem + ".png")
        LOG["figs"].append(stem)
        if png.exists():
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run()
            pic = run.add_picture(str(png), width=Inches(5.5))
            maxh = Inches(7.4)
            if pic.height > maxh:
                ratio = maxh / pic.height
                pic.height = int(pic.height * ratio)
                pic.width = int(pic.width * ratio)
        else:
            LOG["missing_figs"].append(stem)
            t = self.doc.add_table(rows=1, cols=1)
            t.style = "Table Grid"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            c = t.rows[0].cells[0]
            c.width = Inches(5.0)
            tr = t.rows[0]._tr.get_or_add_trPr()
            h = OxmlElement("w:trHeight")
            h.set(qn("w:val"), str(int(2.2 * 1440)))
            h.set(qn("w:hRule"), "atLeast")
            tr.append(h)
            tcpr = c._tc.get_or_add_tcPr()
            va = OxmlElement("w:vAlign")
            va.set(qn("w:val"), "center")
            tcpr.append(va)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            LOG["markers"].append((self.ctx, "[Figure pending: %s]" % stem))
            add_run(p, "[Figure pending: %s]" % stem, 11, highlight=True)
        self.caption("Figure", num, cap)

    def directive(self, b):
        _, name, args, text = b
        opts = {}
        for a in [x.strip() for x in args.split(",") if x.strip()]:
            if "=" in a:
                k, v = a.split("=", 1)
                opts[k] = v
            else:
                opts[a] = True
        if name == "pagebreak":
            self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        elif name == "spacer":
            p = self.doc.add_paragraph()
            p.paragraph_format.space_after = Pt(float(text or 12))
            p.paragraph_format.line_spacing = 1.0
        elif name == "sig":
            p = self.doc.add_paragraph()
            p.paragraph_format.space_before = Pt(26)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_run(p, "_" * 34)
            q = self.doc.add_paragraph()
            q.paragraph_format.line_spacing = 1.0
            q.alignment = WD_ALIGN_PARAGRAPH.LEFT
            self.inline(q, text)
        else:
            al = {"center": WD_ALIGN_PARAGRAPH.CENTER, "left": WD_ALIGN_PARAGRAPH.LEFT, "right": WD_ALIGN_PARAGRAPH.RIGHT}[name]
            p = self.doc.add_paragraph()
            p.alignment = al
            p.paragraph_format.line_spacing = float(opts.get("spacing", 1.15))
            p.paragraph_format.space_after = Pt(float(opts.get("after", 6)))
            self.inline(p, text, float(opts.get("size", 12)), bool(opts.get("bold")), bool(opts.get("italic")))

    # ---- special pages
    def field_list(self, title, instr, entries, note):
        p = self.doc.add_paragraph(style="FrontHeading")
        add_run(p, title)
        n = self.doc.add_paragraph()
        n.alignment = WD_ALIGN_PARAGRAPH.LEFT
        add_run(n, note, 10, italic=True)
        if not entries:
            entries = ["(no entries)"]
        for k, e in enumerate(entries):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(3)
            if isinstance(e, tuple):
                p.paragraph_format.left_indent = Inches(0.3 * (e[0] - 1))
                e = e[1]
            if k == 0:
                self._fld(p, "begin")
                r = p.add_run()
                it = OxmlElement("w:instrText")
                it.set(qn("xml:space"), "preserve")
                it.text = " %s " % instr
                r._element.append(it)
                self._fld(p, "separate")
            add_run(p, e, 11)
            if k == len(entries) - 1:
                self._fld(p, "end")

    def _fld(self, p, kind):
        r = p.add_run()
        fc = OxmlElement("w:fldChar")
        fc.set(qn("w:fldCharType"), kind)
        r._element.append(fc)

    def references(self):
        p = self.doc.add_paragraph(style="Heading 1")
        add_run(p, "REFERENCES")
        self.toc_ref = True
        for k in self.plan.cite_order:
            num = self.cite_num[k]
            q = self.doc.add_paragraph()
            q.alignment = WD_ALIGN_PARAGRAPH.LEFT
            q.paragraph_format.left_indent = Inches(0.45)
            q.paragraph_format.first_line_indent = Inches(-0.45)
            q.paragraph_format.line_spacing = 1.15
            q.paragraph_format.space_after = Pt(6)
            add_run(q, "[%d]\t" % num, 11)
            q.paragraph_format.tab_stops.add_tab_stop(Inches(0.45))
            if k in self.bib:
                for text, it in ieee(self.bib[k]):
                    if text.startswith(" [CHECK"):
                        add_run(q, text, 11, highlight=True)
                    else:
                        add_run(q, text, 11, italic=it)
            else:
                add_run(q, "[Reference missing for key %s]" % k, 11, highlight=True)
        LOG["uncited_bib"] = [k for k in self.bib if k not in self.cite_num]

    # ---- headers / footers
    def header_footer(self, sec):
        sec.header.is_linked_to_previous = False
        sec.footer.is_linked_to_previous = False
        hp = sec.header.paragraphs[0]
        hp.text = ""
        hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run(hp, SHORT_TITLE, 10, italic=True)
        ppr = hp._element.get_or_add_pPr()
        bd = OxmlElement("w:pBdr")
        b = OxmlElement("w:bottom")
        for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", "000000")):
            b.set(qn("w:" + k), v)
        bd.append(b)
        ppr.append(bd)
        fp = sec.footer.paragraphs[0]
        fp.text = ""
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_field(fp, "PAGE", "1", 11)

    @staticmethod
    def page_numbering(sec, fmt, start=1):
        sp = sec._sectPr
        for old in sp.findall(qn("w:pgNumType")):
            sp.remove(old)
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:fmt"), fmt)
        pg.set(qn("w:start"), str(start))
        sp.append(pg)

    # ---- main
    def build(self):
        plan = self.plan
        self.table_nums = [n for n, _ in plan.tables]
        self.fig_nums = [n for n, _ in plan.figures]
        started_prelim = started_main = False
        chap = app = 0
        for kind, fname, blocks in plan.files:
            self.ctx = fname or kind
            if kind == "cover" and fname == "01_title.md":
                self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            if kind in ("front", "toc", "lof", "lot") and not started_prelim:
                sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
                self.page_numbering(sec, "lowerRoman", 1)
                self.header_footer(sec)
                started_prelim = True
            if kind == "chapter" and not started_main:
                sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
                self.page_numbering(sec, "decimal", 1)
                started_main = True
            if kind == "toc":
                entries = [(lv, t) for lv, t in plan.toc]
                self.field_list("TABLE OF CONTENTS", 'TOC \\o "1-3" \\h \\z \\u', entries,
                                "Note: right-click in this list and choose Update field (or press F9) to refresh entries and page numbers.")
                continue
            if kind == "lof":
                self.field_list("LIST OF FIGURES", 'TOC \\h \\z \\c "Figure"',
                                ["Figure %s. %s" % (n, re.sub(r"[*`]", "", c)) for n, c in plan.figures],
                                "Note: right-click in this list and choose Update field to refresh page numbers.")
                continue
            if kind == "lot":
                self.field_list("LIST OF TABLES", 'TOC \\h \\z \\c "Table"',
                                ["Table %s. %s" % (n, re.sub(r"[*`]", "", c)) for n, c in plan.tables],
                                "Note: right-click in this list and choose Update field to refresh page numbers.")
                continue
            if kind == "refs":
                self.references()
                continue
            for b in blocks:
                k = b[0]
                if k == "h1":
                    if kind in ("chapter", "appendix"):
                        if kind == "chapter":
                            chap += 1
                            label = "CHAPTER %d" % chap
                        else:
                            app += 1
                            label = "APPENDIX %s" % chr(64 + app)
                        p = self.doc.add_paragraph(style="Heading 1")
                        add_run(p, label)
                        p.add_run().add_break()
                        add_run(p, b[1].upper())
                        self.sec = self.sub = 0
                        self.prefix = str(chap) if kind == "chapter" else chr(64 + app)
                    else:
                        p = self.doc.add_paragraph(style="Heading 1")
                        add_run(p, b[1].upper())
                        self.prefix = ""
                        self.sec = self.sub = 0
                elif k == "h2":
                    self.sec += 1
                    self.sub = 0
                    lab = ("%s.%d  " % (self.prefix, self.sec)) if self.prefix else ""
                    p = self.doc.add_paragraph(style="Heading 2")
                    self.inline(p, lab + b[1], 14, True)
                elif k == "h3":
                    self.sub += 1
                    lab = ("%s.%d.%d  " % (self.prefix, self.sec, self.sub)) if self.prefix else ""
                    p = self.doc.add_paragraph(style="Heading 3")
                    self.inline(p, lab + b[1], 12, True)
                elif k == "p":
                    self.para(b[1])
                elif k in ("ul", "ol"):
                    self.list_block(k, b[1])
                elif k == "code":
                    self.code_block(b[1])
                elif k == "table":
                    self.table_block(b)
                elif k == "fig":
                    self.figure_block(b)
                elif k == "dir":
                    self.directive(b)
        self.doc.save(str(OUT))


# ----------------------------------------------------------------------------- open-check
def check():
    d = Document(str(OUT))
    hs = {"Heading 1": 0, "Heading 2": 0, "Heading 3": 0}
    words = 0
    marker_re = re.compile(MARKER_RE)
    markers = []
    for p in d.paragraphs:
        if p.style.name in hs:
            hs[p.style.name] += 1
        words += len(p.text.split())
        markers += marker_re.findall(p.text)
    tcells = 0
    for t in d.tables:
        for r in t.rows:
            for c in r.cells:
                words += len(c.text.split())
                markers += marker_re.findall(c.text)
                tcells += 1
    xml = d.element.xml
    seq_t = len(re.findall(r"SEQ Table", xml))
    seq_f = len(re.findall(r"SEQ Figure", xml))
    rows = sum(len(t.rows) for t in d.tables)
    pics = len(d.inline_shapes)
    est = words / 300.0 + pics * 0.55 + rows * 0.035 + 6
    print("\n=== open-check of %s ===" % OUT.name)
    print("sections: %d | Heading1/2/3: %d/%d/%d | tables: %d (%d rows) | pictures: %d" % (len(d.sections), hs["Heading 1"], hs["Heading 2"], hs["Heading 3"], len(d.tables), rows, pics))
    print("SEQ Table fields: %d | SEQ Figure fields: %d | words: %d | highlighted markers found: %d" % (seq_t, seq_f, words, len(markers)))
    print("estimated pages (rough, words/300 + figures + table rows): ~%d" % round(est))
    return markers, est


def main():
    plan = Plan()
    plan.scan()
    bibfile = SHARED / "references.bib"
    bib = parse_bib(bibfile.read_text(encoding="utf-8")) if bibfile.exists() else OrderedDict()
    if not bib:
        print("WARNING: references.bib missing or empty - citations will show as [?key]")
    b = Builder(plan, bib)
    b.build()
    markers, est = check()
    print("\n--- build log ---")
    for w in plan.warn:
        print("WARN", w)
    print("figures referenced: %d | missing figure files: %s" % (len(LOG["figs"]), LOG["missing_figs"]))
    print("citation keys used: %d | missing from bib: %s" % (len(plan.cite_order), sorted(LOG["missing_cites"])))
    print("bib entries uncited: %s" % LOG["uncited_bib"])
    print("missing result tokens/refs:", sorted(LOG["missing_tokens"]))
    from collections import Counter
    c = Counter(m[1][:70] for m in LOG["markers"])
    print("markers (%d):" % len(LOG["markers"]))
    for t, n in sorted(c.items()):
        print("  %3d x %s" % (n, t))
    print("output:", OUT)


if __name__ == "__main__":
    main()
