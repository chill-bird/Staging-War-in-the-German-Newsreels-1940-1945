"""
classify_shot_scale.py
---

Classify shot scale to long shot, medium shot, close shot (s. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8451474)
"""

import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.util import get_keyframe_paths

# Paths
load_dotenv()

CHAPTER_DATA = Path(os.getenv("CHAPTERS_DATA_TSV")).resolve()
KEYFRAMES_DIR = Path(os.getenv("KEYFRAMES_DIR")).resolve()
TRANSCRIPTIONS_DIR = Path(os.getenv("TRANSCRIPTIONS_DIR")).resolve()
SHOT_SCALE_CLASSIFICATIONS_DIR = Path(os.getenv("SHOT_SCALE_CLASSIFICATIONS_DIR")).resolve()
assert CHAPTER_DATA.is_file(), "Could not find TSV."
assert KEYFRAMES_DIR.is_dir(), "Could not find directory for keyframes."

SHOT_SCALE_CLASSIFICATIONS_DIR.mkdir(parents=False, exist_ok=True)
assert (
    SHOT_SCALE_CLASSIFICATIONS_DIR.is_dir()
), "Could not find directory for shot scale classification results."


SHOT_PROMPTS = {
    "Long Shot": [
        "a wide shot from a black and white documentary",
        "a long shot showing the full scene",
        "a wide shot with small human figures",
        "a wide shot showing people and surroundings",
    ],
    "Medium Shot": [
        "a black and white medium shot of people",
        "a medium shot showing a person from the waist up",
        "a medium shot focusing on people in a scene",
        "a medium shot focusing on machinery in a scene",
    ],
    "Close-up": [
        "a black and white close-up shot of a face",
        "a close-up of a person's face in a black and white documentary",
        "a close shot focusing on part of a person",
        "a close shot focusing on part of a machine in a black and white documentary",
    ],
    # "Divider": ["white letters on black screen"],
}

MODEL_NAME = "openai/clip-vit-large-patch14"
MODEL = CLIPModel.from_pretrained(MODEL_NAME).to("cuda")
PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)
MODEL.eval()


def file_not_existent(row: pd.Series) -> bool:
    """Returns True if file missing."""

    result_filename = row["filestem"] + "_" + row["chapter"] + ".csv"
    result_filepath = Path(SHOT_SCALE_CLASSIFICATIONS_DIR / result_filename).resolve()
    return not result_filepath.is_file()


def encode_prompts(prompts_dict: dict) -> dict[str, torch.Tensor]:
    text_embeddings = {}
    for label, prompts in prompts_dict.items():
        inputs = PROCESSOR(text=prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = MODEL.get_text_features(**inputs)
        # Extract the tensor either from out.text_embeds or out.pooler_output
        if hasattr(out, "text_embeds"):
            emb = out.text_embeds
        else:
            emb = out.pooler_output

        emb = F.normalize(emb, dim=-1)
        text_embeddings[label] = emb.mean(dim=0)
    return text_embeddings


def classify_image_batch(img_paths: list[Path], text_embeddings: dict, batch_size: int = 32):
    """
    Classify a batch of images (chapter-level batching with sub-batches of batch_size).
    Returns a list of dicts with predicted label and score for each image.
    """
    results = []

    # Split images into sub-batches of batch_size
    for i in range(0, len(img_paths), batch_size):
        sub_batch_paths = img_paths[i : i + batch_size]
        batch_of_images = [Image.open(p).convert("RGB") for p in sub_batch_paths]

        # Convert images to model inputs
        inputs = PROCESSOR(images=batch_of_images, return_tensors="pt", padding=True).to("cuda")

        # Forward pass
        with torch.no_grad():
            out = MODEL.get_image_features(**inputs)

        if hasattr(out, "image_embeds"):
            image_embs = out.image_embeds
        else:
            image_embs = out.pooler_output

        image_embs = F.normalize(image_embs, dim=-1)
        assert torch.isfinite(image_embs).all(), "NaNs or Infs in image embeddings!"

        # Compute predictions for each image in sub-batch
        for j, img_path in enumerate(sub_batch_paths):
            img_emb = image_embs[j]
            scores = {label: (img_emb @ emb).item() for label, emb in text_embeddings.items()}

            pred_label = max(scores, key=scores.get)
            pred_score = round(scores[pred_label], 3)

            results.append({"frame": img_path.stem, "prediction": pred_label, "score": pred_score})

    return results


def main() -> None:

    text_embeddings = encode_prompts(SHOT_PROMPTS)

    df = pd.read_csv(CHAPTER_DATA, sep="\t")
    # Only data with timestamps
    df = df[(df["start"].notna()) & (df["end"].notna())]
    # Only chapters classified as combat scene or showing soldiers
    df = df[
        (df["german_soldiers_depicted"].notna()) & (df["german_soldiers_depicted"])
        | (df["is_combat_scene"].notna()) & (df["is_combat_scene"])
    ]

    # Only for chapters where classification not already exists
    df["classification_missing"] = df.apply(lambda row: file_not_existent(row), axis=1)
    df = df[df["classification_missing"]]
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    # task_id = 5

    row = df.iloc[task_id]
    print(f"[{task_id+1}/{len(df)}] Processing {row['title']} Chapter {row['chapter']}...")

    result_filename = row["filestem"] + "_" + row["chapter"] + ".csv"
    result_filepath = Path(SHOT_SCALE_CLASSIFICATIONS_DIR / result_filename).resolve()

    imgs_paths = get_keyframe_paths(row["filestem"], row["start_frame"], row["end_frame"])

    # Batch classification
    batch_results = classify_image_batch(imgs_paths, text_embeddings, batch_size=32)

    # Add chapter and filestem info
    for r in batch_results:
        r["filestem"] = row["filestem"]
        r["chapter"] = row["chapter"]
    if batch_results:
        # Save results
        df = pd.DataFrame(batch_results)
        df.to_csv(result_filepath, index=False)
        print(f"Results saved to {result_filepath}")
    else:
        print("No images found.")


if __name__ == "__main__":
    main()
