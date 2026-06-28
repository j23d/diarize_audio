#!/usr/bin/env python3

import sys
import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description="Batch-Verarbeitung via whisperx_speaker_diarization_de.py")
parser.add_argument("audio_files", nargs="+", help="Pfade zu Audiodateien")
parser.add_argument("--model", default="large-v3", help="Whisper-Modell (Default: large-v3)")
parser.add_argument("--language", default="de", help="Sprache (Default: de)")
parser.add_argument("--format", choices=["txt", "json", "srt"], default="txt", help="Ausgabeformat (Default: txt)")
args = parser.parse_args()

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisperx_speaker_diarization_de.py")

errors = []
for audio_file in args.audio_files:
    print(f"\n--- Verarbeite: {audio_file} ---")
    result = subprocess.run(
        [sys.executable, script, audio_file,
         "--model", args.model,
         "--language", args.language,
         "--format", args.format],
    )
    if result.returncode != 0:
        print(f"❌ Fehler bei: {audio_file}")
        errors.append(audio_file)

print(f"\n=== Batch abgeschlossen: {len(args.audio_files) - len(errors)}/{len(args.audio_files)} erfolgreich ===")
if errors:
    print("Fehler bei:")
    for f in errors:
        print(f"  - {f}")
    sys.exit(1)
