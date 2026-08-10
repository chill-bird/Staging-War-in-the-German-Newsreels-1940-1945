import os
import sys
import glob
import argparse

sys.path.append("/work/xb27qenu-ca_lmske/Keyframe-Extraction-for-video-summarization/src/extraction")
sys.path.append("/work/xb27qenu-ca_lmske/Keyframe-Extraction-for-video-summarization/src/scripts")
from Keyframe_extraction import scen_keyframe_extraction

def main():
    parser = argparse.ArgumentParser(description="Run keyframe extraction on videos")
    
    parser.add_argument(
        "--video_file",
        type=str,
        required=True,
        help="Directory containing video files (.mp4)"
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        required=True,
        help="Directory containing video feature files (.pkl)"
    )
    parser.add_argument(
        "--scene_file",
        type=str,
        required=True,
        help="Directory containing scene files (.scenes.txt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted keyframes"
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="Optional: specify a single video basename to process (without extension)"
    )
    
    args = parser.parse_args()

     # === Print parameters for debugging ===
    print("\n===== RUN PARAMETERS =====")
    print(f"Video file        : {args.video_file}")
    print(f"Feature file      : {args.feature_file}")
    print(f"Scene file        : {args.scene_file}")
    print(f"Output directory  : {args.output_dir}")
    print(f"Video name        : {args.video_name}")
    print("============================\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Call keyframe extraction
    scen_keyframe_extraction(
        scenes_path=args.scene_file,
        features_path=args.feature_file,
        video_path=args.video_file,
        save_path=args.output_dir,
        folder_path=args.video_name
    )

if __name__ == "__main__":
    main()
