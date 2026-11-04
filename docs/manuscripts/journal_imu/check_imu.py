"""Consistency checks for the IMU manuscript folder (run with the audit venv)."""
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
shared = here.parent / "shared"
macros_txt = (shared / "results_macros.tex").read_text(encoding="utf8")
macros = dict(re.findall(r"\\newcommand\{\\(R[A-Za-z]+)\}\{(.*)\}", macros_txt))

tex_files = [here / "main.tex", here / "declarations.tex", here / "highlights.tex",
             here / "supplementary.tex"] + sorted((here / "sections").glob("*.tex"))
bibs = [shared / "references.bib"]

problems = []

# control characters / BOM
for f in tex_files + bibs + [here / "README.md"]:
    raw = f.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append(f"BOM in {f.name}")
    txt = raw.decode("utf8")
    bad = [c for c in txt if ord(c) < 32 and c not in "\n\r\t"]
    if bad:
        problems.append(f"control chars in {f.name}: {bad[:5]}")

# macros
used = set()
for f in tex_files:
    t = f.read_text(encoding="utf8")
    t = "\n".join(re.sub(r"(?<![\\])%.*", "", l) if not l.lstrip().startswith("%") else "" for l in t.splitlines())
    for m in re.findall(r"\\(R[A-Z][A-Za-z]*)", t):
        used.add((m, f.name))
missing = sorted({(m, n) for m, n in used if m not in macros and m not in {"Rightarrow"}})
print("macros used:", len({m for m, _ in used}), "missing:", missing)
problems += [f"missing macro {m} in {n}" for m, n in missing]

# citations
keys = set()
for b in bibs:
    keys |= set(re.findall(r"^@\w+\{([^,]+),", b.read_text(encoding="utf8"), re.M))
cited = set()
for f in tex_files:
    t = f.read_text(encoding="utf8")
    t = "\n".join(re.sub(r"(?<![\\])%.*", "", l) for l in t.splitlines())
    for grp in re.findall(r"\\cite\{([^}]*)\}", t):
        cited |= {k.strip() for k in grp.split(",")}
miss_cite = sorted(cited - keys)
print("cited:", len(cited), "missing keys:", miss_cite)
problems += [f"missing cite key {k}" for k in miss_cite]

# labels / refs (main + sections; supplementary separately)
def labrefs(files):
    labels, refs = set(), set()
    for f in files:
        t = f.read_text(encoding="utf8")
        t = "\n".join(re.sub(r"(?<![\\])%.*", "", l) for l in t.splitlines())
        labels |= set(re.findall(r"\\label\{([^}]*)\}", t))
        refs |= set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", t))
    return labels, refs
main_files = [here / "main.tex", here / "declarations.tex"] + sorted((here / "sections").glob("*.tex"))
lab, ref = labrefs(main_files)
print("main refs without label:", sorted(ref - lab))
problems += [f"ref without label {r}" for r in sorted(ref - lab)]
print("labels never referenced (info):", sorted(lab - ref))
lab_s, ref_s = labrefs([here / "supplementary.tex"])
problems += [f"supp ref without label {r}" for r in sorted(ref_s - lab_s)]


def expand(text):
    text = "\n".join(re.sub(r"(?<![\\])%.*", "", l) for l in text.splitlines())
    text = re.sub(r"\\(R[A-Za-z]+)", lambda m: macros.get(m.group(1), "XX"), text)
    return text


def words(text):
    t = expand(text)
    t = re.sub(r"\\textbf\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\$[^$]*\$", " m ", t)
    t = re.sub(r"\\(?:cite|ref|eqref|label)\{[^}]*\}", " ", t)
    t = re.sub(r"\\[A-Za-z]+\*?", " ", t)
    t = re.sub(r"[{}~\\]", " ", t)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.,%/']*", t))


abstract = (here / "sections" / "00_abstract.tex").read_text(encoding="utf8")
aw = words(abstract)
print("abstract words:", aw)
if aw > 250:
    problems.append(f"abstract {aw} > 250")

# keywords
main = (here / "main.tex").read_text(encoding="utf8")
kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", main, re.S).group(1)
nkw = len([k for k in kw.split(r"\sep") if k.strip()])
print("keywords:", nkw)
if not 1 <= nkw <= 7:
    problems.append("keywords out of range")

# highlights
hl = (here / "highlights.tex").read_text(encoding="utf8")
items = re.findall(r"\\item\s+(.*)", hl)
lens = [len(re.sub(r"\s*%.*$", "", i).strip()) for i in items]
print("highlights:", len(items), lens)
if not 3 <= len(items) <= 5 or max(lens) > 85:
    problems.append("highlights out of limits")

# approximate main-text words (sections 01, 03-08, excluding tables, equations, captions)
total = 0
for f in sorted((here / "sections").glob("*.tex")):
    if f.name.startswith("00"):
        continue
    t = f.read_text(encoding="utf8")
    t = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{(equation|gather)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", t, flags=re.S)
    n = words(t)
    total += n
    print(f"  {f.name}: {n}")
print("approx main-text words (no tables/captions/equations):", total)
bspc = here.parent / "journal_bspc" / "sections"
tb = 0
for f in sorted(bspc.glob("0[1-8]*.tex")):
    t = f.read_text(encoding="utf8")
    t = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{(equation|gather)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", t, flags=re.S)
    tb += words(t)
print("BSPC same measure:", tb)

# banned wording
banned = [r"significan", r"p\s*=", r"pre-?registered", r"\bfirst\b(?! author)", r"state[- ]of[- ]the[- ]art",
          r"p-value", r"scanner fingerprint", r"novel method", r"\bnovel\b"]
for f in tex_files:
    t = "\n".join(re.sub(r"(?<![\\])%.*", "", l) for l in f.read_text(encoding="utf8").splitlines())
    for b in banned:
        for m in re.finditer(b, t, re.I):
            ctx = t[max(0, m.start() - 50): m.end() + 40].replace("\n", " ")
            print(f"  wording [{b}] {f.name}: ...{ctx}...")

print("PROBLEMS:", problems if problems else "none")
sys.exit(1 if problems else 0)
