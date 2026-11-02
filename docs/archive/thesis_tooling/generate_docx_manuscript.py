"""
Converts the markdown manuscript into an Elsevier publication-ready Word DOCX document,
adhering to academic formatting standards (Times New Roman / Calibri, 1.5 line spacing,
1-inch margins, formatted tables with borders, clear section hierarchies).
"""
import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = r"d:\Sathwik\lung_disease_detector\Lung_Disease_Detector-main"
MD_PATH = os.path.join(ROOT, "docs", "MANUSCRIPT_BSPC_LUNGAI.md")
SUB_DIR = os.path.join(ROOT, "submission_package")
DOCX_OUT = os.path.join(SUB_DIR, "Manuscript_BSPC_Elsevier.docx")
COVER_OUT = os.path.join(SUB_DIR, "Cover_Letter_Elsevier.docx")

def set_cell_border(cell, **kwargs):
    """Set cell borders in Word table."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge in ('top', 'left', 'bottom', 'right'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = 'w:{}'.format(edge)
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, val in edge_data.items():
                element.set(qn('w:{}'.format(key)), str(val))

def build_docx():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = Document()

    # Page Setup: Letter / A4, 1-inch margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.25
    normal_style.paragraph_format.space_after = Pt(4)

    # Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(12)
    title_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_t = title_p.add_run("Forensic Dataset Reconstruction and Cross-Source Domain Shift Analysis in Multi-Class Thoracic Radiography: An Empirical Study Using DenseNet-121")
    run_t.font.name = 'Times New Roman'
    run_t.font.size = Pt(16)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(16, 44, 87)

    # Authors
    auth_p = doc.add_paragraph()
    auth_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    auth_p.paragraph_format.space_after = Pt(2)
    run_a = auth_p.add_run("Sathwik Katkam*")
    run_a.font.name = 'Times New Roman'
    run_a.font.size = Pt(12)
    run_a.font.bold = True

    # Affiliation
    aff_p = doc.add_paragraph()
    aff_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    aff_p.paragraph_format.space_after = Pt(14)
    run_aff = aff_p.add_run("Department of Computer Science and Engineering,\nJawaharlal Nehru Technological University Hyderabad (JNTUH), Kukatpally, Hyderabad, Telangana 500085, India\n*Corresponding Author Email: skatkam@jntuh.ac.in")
    run_aff.font.name = 'Times New Roman'
    run_aff.font.size = Pt(10)
    run_aff.font.italic = True

    # Parse and add sections from Markdown
    in_table = False
    table_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip top header metadata already processed
        if i < 8 and (line.startswith("# ") or line.startswith("**Sathwik") or line.startswith("Department") or line.startswith("*Target Journal")):
            i += 1
            continue

        if line == "---":
            i += 1
            continue

        # Check Table start
        if line.startswith("|") and line.endswith("|"):
            in_table = True
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            # End of table, process it
            in_table = False
            # Render table
            rows_data = []
            for t_line in table_lines:
                if re.match(r'^\|[\s\-:]+\|$', t_line):
                    continue  # Separator line
                cells = [c.strip() for c in t_line.strip('|').split('|')]
                rows_data.append(cells)
            
            if rows_data:
                num_cols = max(len(r) for r in rows_data)
                tbl = doc.add_table(rows=len(rows_data), cols=num_cols)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                tbl.autofit = True
                
                for r_idx, row in enumerate(rows_data):
                    for c_idx, cell_val in enumerate(row):
                        if c_idx < num_cols:
                            cell = tbl.cell(r_idx, c_idx)
                            cell.text = cell_val.replace("**", "").replace("*", "")
                            p = cell.paragraphs[0]
                            p.paragraph_format.space_after = Pt(2)
                            p.paragraph_format.space_before = Pt(2)
                            if r_idx == 0:
                                for r in p.runs:
                                    r.font.bold = True
                                    r.font.size = Pt(9.5)
                                    r.font.name = 'Times New Roman'
                                set_cell_border(cell, bottom={"sz": 12, "val": "single", "color": "000000"},
                                                      top={"sz": 12, "val": "single", "color": "000000"})
                            else:
                                for r in p.runs:
                                    r.font.size = Pt(9)
                                    r.font.name = 'Times New Roman'
                                if r_idx == len(rows_data) - 1:
                                    set_cell_border(cell, bottom={"sz": 12, "val": "single", "color": "000000"})
                                else:
                                    set_cell_border(cell, bottom={"sz": 4, "val": "single", "color": "E0E0E0"})

                doc.add_paragraph()  # spacer
            table_lines = []

        # Headings
        if line.startswith("### "):
            h = doc.add_heading(level=3)
            h.paragraph_format.space_before = Pt(10)
            h.paragraph_format.space_after = Pt(3)
            run = h.add_run(line.replace("### ", ""))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(16, 44, 87)
        elif line.startswith("## "):
            h = doc.add_heading(level=2)
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(4)
            run = h.add_run(line.replace("## ", ""))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = RGBColor(16, 44, 87)
        elif line.startswith("# "):
            h = doc.add_heading(level=1)
            h.paragraph_format.space_before = Pt(16)
            h.paragraph_format.space_after = Pt(6)
            run = h.add_run(line.replace("# ", ""))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(16, 44, 87)
        elif line.startswith("* ") or line.startswith("- "):
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_after = Pt(2)
            text = line[2:]
            parts = re.split(r'(\*\*.*?\*\*)', text)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    r = p.add_run(part[2:-2])
                    r.font.bold = True
                else:
                    p.add_run(part)
        elif line:
            p = doc.add_paragraph()
            parts = re.split(r'(\*\*.*?\*\*)', line)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    r = p.add_run(part[2:-2])
                    r.font.bold = True
                else:
                    p.add_run(part)

        i += 1

    doc.save(DOCX_OUT)
    print(f"Generated Elsevier formatted DOCX: {DOCX_OUT}")

    # Build Cover Letter DOCX
    cov_doc = Document()
    for s in cov_doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    p_head = cov_doc.add_paragraph()
    p_head.paragraph_format.space_after = Pt(14)
    r = p_head.add_run("To:\nThe Editor-in-Chief,\nBiomedical Signal Processing and Control (BSPC),\nElsevier B.V.")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)

    p_sub = cov_doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(10)
    r_sub = p_sub.add_run("Subject: Submission of Original Research Article — 'Forensic Dataset Reconstruction and Cross-Source Domain Shift Analysis in Multi-Class Thoracic Radiography: An Empirical Study Using DenseNet-121'")
    r_sub.font.bold = True
    r_sub.font.name = 'Times New Roman'

    p_body = cov_doc.add_paragraph()
    p_body.paragraph_format.space_after = Pt(8)
    p_body.paragraph_format.line_spacing = 1.25
    p_body.add_run("Dear Editor-in-Chief,\n\nWe are pleased to submit our original research manuscript entitled \"Forensic Dataset Reconstruction and Cross-Source Domain Shift Analysis in Multi-Class Thoracic Radiography: An Empirical Study Using DenseNet-121\" for consideration for publication in Biomedical Signal Processing and Control.\n\n"
                   "While convolutional neural networks routinely demonstrate impressive performance on single-cohort benchmarks, translating these models into multi-center clinical practice remains hampered by shortcut learning, patient-identity leakage, and severe domain shift. In this empirical study, we present a forensically audited dataset reconstruction comprising 10,547 verified planar chest radiographs across 10,270 unique patients from eight independent clinical repositories, enforcing zero patient overlap across training, validation, and test partitions.\n\n"
                   "Key highlights and contributions of this study include:\n"
                   "1. Forensic Dataset Decontamination: Elimination of cross-modality contamination (purging axial CT slices and non-radiographic artifacts) and 83 near-duplicate pairs, establishing a rigorous leak-free multi-class benchmark.\n"
                   "2. Transparent Reporting of Domain Shift Penalty: We candidly report that enforcing multi-source patient isolation resulted in nominal test accuracy adjusting from 78.53% to 76.18%, reflecting the dismantling of institutional shortcut cues rather than algorithmic regression.\n"
                   "3. Quarantined External Generalization & Sensor Shift: Zero-shot evaluation on the Montgomery County cohort revealed that while 100% binary abnormality sensitivity was maintained (58/58 TB cases recognized as pathological), exact disease classification suffered severe breakdown due to digitizer noise (film grain edge density, median Laplacian variance >1,580 vs 373 on direct digital CXRs).\n\n"
                   "This manuscript has not been published previously, is not under consideration for publication elsewhere, and its publication has been approved by all authors.\n\n"
                   "We have selected the standard subscription publishing model (No Article Processing Charges / $0 APC).\n\n"
                   "Thank you very much for your time and consideration of our work.\n\n"
                   "Sincerely,\n\n"
                   "Sathwik Katkam\n"
                   "Department of Computer Science and Engineering,\n"
                   "Jawaharlal Nehru Technological University Hyderabad (JNTUH),\n"
                   "Kukatpally, Hyderabad, Telangana 500085, India\n"
                   "Email: skatkam@jntuh.ac.in")

    cov_doc.save(COVER_OUT)
    print(f"Generated Cover Letter DOCX: {COVER_OUT}")

if __name__ == "__main__":
    build_docx()
