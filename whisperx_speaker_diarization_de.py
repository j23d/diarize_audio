#!/usr/bin/env python3

import whisperx
import os
import sys
import argparse
import torch
import json
import subprocess
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("Fehler: Kein HF_TOKEN in .env gefunden.")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Transkription mit Sprechererkennung via WhisperX + pyannote")
parser.add_argument("audio_file", help="Pfad zur Audiodatei (z.B. dialog.wav)")
parser.add_argument("--model", default="large-v3", help="Whisper-Modell (Default: large-v3)")
parser.add_argument("--language", default="de", help="Sprache (z.B. de, en – Default: de)")
parser.add_argument("--format", choices=["txt", "json", "srt"], default="txt", help="Ausgabeformat (Default: txt)")
args = parser.parse_args()

if not os.path.exists(args.audio_file):
    print(f"Datei nicht gefunden: {args.audio_file}")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Lade Whisper-Modell: {args.model}")
if args.model == "turbo":
    print("❌ 'turbo' wird von whisperx nicht unterstützt. Bitte verwende 'large-v3' oder 'medium'.")
    sys.exit(1)

model = whisperx.load_model(args.model, device=device, compute_type=compute_type)
transcription = model.transcribe(args.audio_file, language=args.language)

print("Führe Alignment durch...")
align_model, metadata = whisperx.load_align_model(language_code=args.language, device=device)
aligned = whisperx.align(transcription["segments"], align_model, metadata, args.audio_file, device)

# === Audio vorbereiten (Mono, 16kHz WAV) ===
base, _ = os.path.splitext(args.audio_file)
prepared_wav = base + "_prepared.wav"

print("Bereite Audio mit ffmpeg vor...")
subprocess.run([
    "ffmpeg", "-y", "-i", args.audio_file,
    "-ac", "1", "-ar", "16000", "-f", "wav", prepared_wav
], check=True)

print("Starte Sprechererkennung mit pyannote.audio...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
diarization = pipeline(prepared_wav)

# === Sprecherzuordnung manuell (nach Zeitüberlappung) ===
def find_speaker(start, end, diarization):
    max_overlap = 0
    chosen_speaker = "Unknown"
    for turn in diarization.itertracks(yield_label=True):
        seg_start = turn[0].start
        seg_end = turn[0].end
        speaker = turn[2]
        overlap = max(0, min(end, seg_end) - max(start, seg_start))
        if overlap > max_overlap:
            max_overlap = overlap
            chosen_speaker = speaker
    return chosen_speaker

segments_with_speakers = []
for seg in aligned["segments"]:
    speaker = find_speaker(seg["start"], seg["end"], diarization)
    seg["speaker"] = speaker
    segments_with_speakers.append(seg)

output_file = f"{base}_diarized.{args.format}"
print(f"Schreibe Ausgabe: {output_file}")

if args.format == "txt":
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments_with_speakers:
            line = f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['speaker']}: {segment['text']}"
            f.write(line + "\n")

elif args.format == "json":
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(segments_with_speakers, f, ensure_ascii=False, indent=2)

elif args.format == "srt":
    def fmt(ts):
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts % 1) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, segment in enumerate(segments_with_speakers, 1):
            f.write(f"{idx}\n")
            f.write(f"{fmt(segment['start'])} --> {fmt(segment['end'])}\n")
            f.write(f"{segment['speaker']}: {segment['text']}\n\n")

print("Fertig.")
