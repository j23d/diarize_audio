#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description="Transkript TXT → LaTeX/PDF")
parser.add_argument("--input", required=True, help="Eingabe-Transkript (.txt)")
parser.add_argument("--output", default=None, help="Ausgabe-PDF (Default: <input>.pdf)")
parser.add_argument("--mapping", nargs="*", default=[],
                    help="Sprecher-Mapping z.B. SPEAKER_00=Coach SPEAKER_01=Klient")
parser.add_argument("--intro", default=None, help="Optionale Intro-Textdatei")
parser.add_argument("--outro", default=None, help="Optionale Outro-Textdatei")
args = parser.parse_args()

if not os.path.exists(args.input):
    print(f"Datei nicht gefunden: {args.input}")
    sys.exit(1)

speaker_map = {}
for entry in args.mapping:
    if "=" not in entry:
        print(f"Ungültiges Mapping (erwartet KEY=Wert): {entry}")
        sys.exit(1)
    k, v = entry.split("=", 1)
    speaker_map[k.strip()] = v.strip()

def read_file(path):
    if path is None:
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()

def escape_latex(text):
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]
    for char, rep in replacements:
        text = text.replace(char, rep)
    return text

line_pattern = re.compile(r"^\[([^\]]+)\]\s+(\S+):\s*(.*)$")

with open(args.input, encoding="utf-8") as f:
    transcript_lines = f.readlines()

entries = []
for raw in transcript_lines:
    raw = raw.rstrip()
    m = line_pattern.match(raw)
    if m:
        timestamp, speaker, text = m.group(1), m.group(2), m.group(3)
        display_name = speaker_map.get(speaker, speaker)
        entries.append((timestamp, display_name, text))
    elif raw:
        entries.append((None, None, raw))

intro_text = escape_latex(read_file(args.intro))
outro_text = escape_latex(read_file(args.outro))

lines_tex = []
for timestamp, speaker, text in entries:
    if speaker is None:
        lines_tex.append(escape_latex(text) + r"\\")
    else:
        ts_safe = escape_latex(timestamp)
        sp_safe = escape_latex(speaker)
        tx_safe = escape_latex(text)
        lines_tex.append(
            r"\textbf{" + sp_safe + r"}" +
            r" \textit{\small[" + ts_safe + r"]}" +
            r": " + tx_safe + r"\\"
        )

base = os.path.splitext(args.input)[0]
tex_file = base + ".tex"
pdf_output = args.output if args.output else base + ".pdf"
pdf_base = os.path.splitext(pdf_output)[0]

tex_content = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[ngerman]{babel}
\usepackage[margin=2.5cm]{geometry}
\usepackage{parskip}
\setlength{\parskip}{4pt}
\begin{document}
"""

if intro_text:
    tex_content += intro_text + "\n\n\\bigskip\n\n"

tex_content += "\n".join(lines_tex) + "\n"

if outro_text:
    tex_content += "\n\\bigskip\n\n" + outro_text + "\n"

tex_content += r"\end{document}" + "\n"

with open(tex_file, "w", encoding="utf-8") as f:
    f.write(tex_content)

print(f"LaTeX geschrieben: {tex_file}")

result = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", f"-jobname={os.path.basename(pdf_base)}", tex_file],
    cwd=os.path.dirname(os.path.abspath(tex_file)) or ".",
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print("pdflatex fehlgeschlagen (oder nicht installiert). Nur .tex erstellt.")
    print(result.stdout[-2000:] if result.stdout else "")
    sys.exit(1)

generated_pdf = os.path.join(
    os.path.dirname(os.path.abspath(tex_file)),
    os.path.basename(pdf_base) + ".pdf"
)
if generated_pdf != os.path.abspath(pdf_output):
    os.replace(generated_pdf, pdf_output)

print(f"PDF erstellt: {pdf_output}")
