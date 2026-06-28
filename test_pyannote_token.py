#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("FAIL: Kein HF_TOKEN in .env gefunden.")
    sys.exit(1)

print(f"OK: HF_TOKEN gefunden ({hf_token[:8]}...)")

models = [
    "pyannote/speaker-diarization",
    "pyannote/segmentation",
    "pyannote/embedding",
    "pyannote/feature-extraction",
]

try:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    print("OK: huggingface_hub verbunden.")
except ImportError:
    print("WARN: huggingface_hub nicht installiert, überspringe Modellzugriffstest.")
    sys.exit(0)

errors = []
for model_id in models:
    try:
        api.model_info(model_id)
        print(f"OK: Zugriff auf {model_id}")
    except Exception as e:
        print(f"FAIL: {model_id} — {e}")
        errors.append(model_id)

if errors:
    print(f"\n{len(errors)} Modell(e) nicht zugänglich. Token-Rechte prüfen oder Nutzungsbedingungen akzeptieren.")
    sys.exit(1)
else:
    print("\nAlle Modelle erreichbar. Token OK.")
