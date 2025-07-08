
# WhisperX Speaker Diarization Batch Tool

This project provides a command-line solution for **automatic speech transcription** and **speaker diarization** using [WhisperX](https://github.com/m-bain/whisperX) and [pyannote.audio](https://github.com/pyannote/pyannote-audio).

It supports processing **individual audio files** and **batch-processing multiple files** in a folder.

---

## Features

- WhisperX-based transcription with accurate word-level timestamps
- Speaker diarization (who speaks when) using pyannote.audio
- Supports `.txt`, `.json`, and `.srt` output formats
- Automatic audio preprocessing (mono, 16kHz) via `ffmpeg`
- Batch processing with resume support via log file
- Token management with `.env` files (for Hugging Face API access)


## Requirements

- Python 3.12+
- WhisperX
- pyannote.audio
- ffmpeg (installed and accessible in system PATH)
- A valid Hugging Face token with access to pyannote models (`.env` required)


## Installation

Create a venv and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Hugging Face token

Create a `.env` file in the project root directory to store your Hugging Face token for accessing pyannote models:

HF_TOKEN=hf_your_actual_huggingface_token_here

If you don't have a token, you need to:
1. Create an account at [Hugging Face](https://huggingface.co).
2. Generate a personal access token: https://huggingface.co/settings/tokens.
3. Accept the terms for each required model before usage:
   - [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
   - [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
   - [pyannote/embedding](https://huggingface.co/pyannote/embedding)
   - [pyannote/feature-extraction](https://huggingface.co/pyannote/feature-extraction)


## Usage

### Activate venv

```bash
source /path/to/venv/bin/activate
```

### Single File Transcription & Diarization
```bash
/path/to/whisperx_speaker_diarization_de.py audio_file.m4a --model large-v3 --language de --format srt
```

### Batch Processing
```bash
/path/to/diarize_audio audio1.wav /path/to/audio2.mp3 audio3.ogg --model large-v3 --language de --format srt
```

The batch script automatically logs processed files to `whisperx_batch.log` and skips them in future runs.


## Output Examples

- `.txt`: Plain text with timestamps and speaker tags
- `.json`: Structured JSON with precise timestamps and speaker labels
- `.srt`: SubRip subtitle format for video captions

## Tested formats

So far the following inputs gave reasonable good quality output:
- wav
- mp3
- ogg
- m4a

## Notes

- The scripts assume they are run inside the Python virtual environment (venv/pyenv) where dependencies are installed.
- WhisperX does not support `turbo` models; recommended models: `large-v3`, `medium`, `small`.
- Diarization requires acceptance of the terms for `pyannote` models on Hugging Face.


## License

MIT License
See the [LICENSE](./LICENSE) file for details.
