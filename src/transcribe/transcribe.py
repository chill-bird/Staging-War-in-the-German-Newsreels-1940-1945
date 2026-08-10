"""
transcribe.py
---

Transcribes audio files via whisper model
"""

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pathlib import Path
import os
import time
import torch
from whisper.utils import get_writer

load_dotenv(dotenv_path=".env")  # Relative to root directory

AUDIO_SOURCE_DIR = Path(os.getenv("AUDIO_SOURCE_DIR")).resolve()
TRANSCRIPTIONS_DIR = Path(os.getenv("TRANSCRIPTIONS_DIR")).resolve()
assert AUDIO_SOURCE_DIR.is_dir(), "Could not find audio source directory."

SLURM_TASK_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))

assert torch.cuda.is_available(), "CUDA not available. Aborting..."
model = WhisperModel("turbo", device="cuda", compute_type="float16")


def transcription_exists(wav_file: Path, target_dir: Path) -> bool:
    """Check if a corresponding transcription file already exists in target_dir."""

    expected_filename = wav_file.stem + ".tsv"
    expected_path = target_dir / expected_filename
    return expected_path.is_file()


def transcribe_all(source_dir: Path, target_dir: Path):
    """Process all wav files in the source directory and save transcriptions."""

    target_dir.mkdir(parents=False, exist_ok=True)
    wav_files = sorted(source_dir.glob("*.wav"))

    if SLURM_TASK_ID >= len(wav_files):
        print("Task ID exceeds number of files — exiting.")
        return

    wav_file = wav_files[SLURM_TASK_ID]
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Transcribing: {wav_file.name}")
    start = time.time()

    segments, info = model.transcribe(str(wav_file), language="de", beam_size=2, vad_filter=True)

    result = {"segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments]}

    writer = get_writer("tsv", target_dir)
    writer(result, wav_file)
    elapsed = time.time() - start
    print(f"runtime_sec={elapsed:.1f}")


if __name__ == "__main__":
    transcribe_all(AUDIO_SOURCE_DIR, TRANSCRIPTIONS_DIR)
