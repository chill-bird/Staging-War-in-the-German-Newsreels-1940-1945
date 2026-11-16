"""
Runs all analyses.
"""

from src.analysis.analyze_asl import main as analyze_asl
from src.analysis.analyze_shot_scales import main as analyze_shot_scales
from src.analysis.analyze_soldiers import main as analyze_soldiers
from src.analysis.analyze_wpm import main as analyze_wpm
from src.analysis.fit_models import CHAPTER_GROUPING, NESTED_GROUPING


def main(grouping_level: str = CHAPTER_GROUPING) -> None:
    analyze_wpm()
    analyze_asl(grouping_level)
    analyze_shot_scales(grouping_level)
    analyze_soldiers(grouping_level)


if __name__ == "__main__":
    grouping_level = CHAPTER_GROUPING
    # grouping_level = NESTED_GROUPING
    main(grouping_level)
