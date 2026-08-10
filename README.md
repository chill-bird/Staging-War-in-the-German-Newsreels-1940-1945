# README

## Requirements

- Python 3.11 to 3.13
- Git LFS

## Getting Started

- Create a virtual environment `python -m venv .venv`
- Activate virtual environment `source .venv/bin/activate`
- Install requirements `pip install -r requirements.txt`
- Copy `.env.example` and save as `.env`. Add environment values to environment variables.

## Analysis

### Q1: Analyze formal features in combat scenes

```sh
python -m src.analysis.q1_formal_features
```

### Q2: Analyze formal features and narratives in scenes with soldiers

```sh
python -m src.analysis.q2_formal_features
python -m src.analysis.q2_narratives
```

## Data preprocessing

### Transcribe audio data via whisper

Transcribes .wav audio files in cluster job

```sh
python -m src.transcribe.transcribe
```

### Get timestamps for chapters with OpenAI

Prompts GPT to align chapters with audio transcription elements. Before starting script, an API key at OpenAI is required: <https://platform.openai.com/api-keys> and to be placed inside `.env` file.

```sh
python -m src.chapters.match_chapters_gpt
```

### Classify chapters with OpenAI

Prompts GPT to align chapters with audio transcription elements. Before starting script, an API key at OpenAI is required: <https://platform.openai.com/api-keys> and to be placed inside `.env` file.

```sh
python -m src.chapters.flag_chapters
```

### Merge chapter info

Merges information from timestamps and chapter descriptions to one TSV file `CHAPTERS_DATA_TSV`.

```sh
python -m src.chapters.merge_chapter_info
```

### Select keyframes with LMSKE

Extracts keyframe from video as described in <https://github.com/ttharden/Keyframe-Extraction-for-video-summarization> and saves them.

1. Run TransNetV2 <https://github.com/soCzech/TransNetV2>

- Create a virtual environment `python -m venv .venv`
- Activate virtual environment `source .venv/bin/activate`
- Install requirements `pip install -r requirements_lmske.txt`

```sh
git clone https://github.com/soCzech/TransNetV2.git
```

Run `run_transnetv2.sh`

1. Run OpenCLIP
In the second step ChineseCLIP was swapped for on OpenCLIP model for better compatibility with german context.

```sh
git clone https://github.com/ttharden/Keyframe-Extraction-for-video-summarization.git
```

Run `run_clip.sh`

1. Extract Keyframes

- Reformat CLIP Feature Outputs with `reformat_features.py`
- Replace `init_center.py`in Keyframe-Extraction-for-video-summarization project with `/src/keyframe_selection_LMSKE/init_center.py` (contains bug fix)
- Replace `KMeans_improvment.py`in Keyframe-Extraction-for-video-summarization project with `/src/keyframe_selection_LMSKE/iKMeans_improvment.py` (contains bug fix)
- Run `run_keyframe_selection.sh`

### Classify shot scale

Classifies Shot scales in keyframes of chapters using CLIP.

```sh
python -m src.keyframes.classify_shot_scale
```

## Acknowledgements

Computations for this work were done (in part) using resources of the Leipzig University Computing Center.
