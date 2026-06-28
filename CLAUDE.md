# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**WhisperX Speaker Diarization Batch Tool** - A Python command-line solution for automatic speech transcription and speaker diarization using WhisperX (speech-to-text) and pyannote.audio (speaker identification).

### Key Technologies
- **WhisperX**: Speech transcription with word-level timestamps
- **pyannote.audio**: Speaker diarization (identifying who speaks when)
- **ffmpeg**: Audio preprocessing to mono 16kHz WAV format
- Python 3.12+

## Development Environment Setup

### Activate Virtual Environment
```bash
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Required Environment Variables
Before running scripts, set:
```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

This environment variable is documented in the README and is required for PyTorch model loading compatibility.

### Hugging Face Token
A `.env` file is required in the project root:
```
HF_TOKEN=hf_your_actual_huggingface_token_here
```

Required Hugging Face model access:
- `pyannote/speaker-diarization`
- `pyannote/segmentation`
- `pyannote/embedding`
- `pyannote/feature-extraction`

## Architecture & File Structure

### Core Scripts

1. **whisperx_speaker_diarization_de.py** (main single-file processor)
   - Handles transcription and diarization for a single audio file
   - Outputs: `.txt`, `.json`, or `.srt` format
   - German-language focused (but supports other languages via `--language` flag)
   - Process:
     1. Load WhisperX model (default: large-v3)
     2. Transcribe audio file
     3. Align transcriptions to word boundaries
     4. Prepare audio (mono, 16kHz WAV via ffmpeg)
     5. Run pyannote speaker diarization
     6. Match speaker labels to transcript segments by time overlap
     7. Write output in requested format

2. **diarize_audio** (bash wrapper)
   - Activates venv and calls `diarize_audio.py` with arguments
   - Intended for batch processing multiple audio files

3. **diarize_audio.py** (batch processor)
   - Accepts multiple audio file paths + `--model`, `--language`, `--format` flags
   - Iterates files sequentially, calls `whisperx_speaker_diarization_de.py` via subprocess for each
   - Tracks failures; prints summary; exits with code 1 if any file failed
   - Same defaults as main script: `large-v3`, `de`, `txt`

### Output Formats

- **`.txt`**: `[start - end] SPEAKER_XX: Text` format with timestamps
- **`.json`**: Structured segments with precise timestamps and speaker labels
- **`.srt`**: SubRip subtitle format for video captions

## Supported Audio Formats

Tested and working:
- wav
- mp3
- ogg
- m4a

## Important Notes

### Model Constraints
- WhisperX does **not** support `turbo` models
- Recommended models: `large-v3`, `medium`, `small`
- Script explicitly rejects `turbo` with error message

### Speaker Matching Algorithm
Uses time-overlap-based matching (in `find_speaker` function):
- For each transcript segment, finds the diarization track with maximum time overlap
- Assigns that speaker label to the segment
- Handles overlapping speakers by selecting the one with most overlap

### Output Cleanup
Generated files are added to .gitignore:
- `*_prepared.wav`: Preprocessed audio files (mono, 16kHz)
- `*_diarized.*`: Output transcript files
- `.srt`, `.json`: Additional output formats

### Additional Scripts
- `test_pyannote_token.py`: Token testing utility for verifying HF token + pyannote model access
- `transkript2pdf.py`: Convert `.txt` transcripts to PDF

## Common Commands

### Single File Transcription
```bash
python whisperx_speaker_diarization_de.py audio_file.m4a --model large-v3 --language de --format srt
```

### Batch Processing
```bash
./diarize_audio audio1.wav /path/to/audio2.mp3 audio3.ogg --model large-v3 --language de --format srt
```

### Check WhisperX Models
Recommended models: `large-v3` (best quality), `medium` (balanced), `small` (faster)

## Development Considerations

### GPU vs CPU
- Script auto-detects CUDA: uses `float16` for GPU, `int8` for CPU
- Large models (large-v3) require significant VRAM on GPU

### File Paths
- Scripts assume UTF-8 encoding for output files
- Audio preprocessing creates temporary `_prepared.wav` files in same directory as input
- Output files follow naming: `{basename}_diarized.{format}`

### Error Handling
- Script exits if:
  - HF_TOKEN not in .env
  - Audio file not found
  - ffmpeg not available in PATH
  - Turbo model requested
  - Audio preprocessing fails
