"""
q2_narratives.py
---

Analyze per-chapter narative feature averages 
"""

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import ruptures as rpt
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np

warnings.filterwarnings("ignore", message="findfont:.*")

from src.util import (
    NARRATIVES_DATA_TSV,
    DOC_DIR,
    get_cleaned_narratives_df,
)
from src.analysis.q2_plots import plot_narrative_trends_per_episode, plot_category_trends_per_episode, plot_heatmap, plot_keyword_trends, plot_topic_profile


# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

FIG_NARRATIVES_DEPICTIONS_PATTERN1_FILE = Path(DOC_DIR / "imgs" / "naratives_depictions_pattern1.pdf").resolve()
FIG_NARRATIVES_DEPICTIONS_PATTERN2_FILE = Path(DOC_DIR / "imgs" / "naratives_depictions_pattern2.pdf").resolve()
FIG_DIR = Path(DOC_DIR / "imgs").resolve()
FIG_DIR.mkdir(parents=True, exist_ok=True)

DF = get_cleaned_narratives_df()

# -----------------------------------------------------------------
# Column schemas
# -----------------------------------------------------------------
categories = {
    "actor_configuration": [
        "individual_hero",
        "collective_military",
        "civilian_collective",
        "enemy_focus",
        "mixed_interaction"
    ],
    "narrative_framing": [
        "heroic_combat",
        "defensive_war",
        "liberation_mission",
        "civilizational_struggle",
        "comradely_front",
        "technological_modernity",
        "sacrificial_martyrdom",
        "soldier_as_victim",
        "inevitable_victory",
        "administrative_normality",
        "occupation_normalization",
        "humanitarian_relief",
        "cultural_guardianship",
        "none"
    ],
    "legitimation_strategy": [
        "defensive",
        "retaliatory",
        "preventive",
        "civilizing",
        "ideological",
        "existential_threat",
        "humanitarian",
        "none"
    ],
    "enemy_moral_status": [
        "legitimate_opponent",
        "weaker_enemy",
        "criminalized_enemy",
        "dehumanized_enemy",
        "existential_threat",
        "none"
    ],
    "embodiment_mode": [
        "disciplined_masculinity",
        "wounded_body",
        "invulnerable_body",
        "eroticized_body",
        "erotic_renunciation",
        "emotionally_restrained",
        "emotional_expressivity",
        "neutral_military_presence",
        "none"
    ],
    "violence_visibility": [
        "shown",
        "implied",
        "absent"
    ],
    "agency_level": [
        "low",
        "medium",
        "high"
    ],
}

prefixes = {
    "actor_configuration": "actor_configuration_",
    "narrative_framing": "narrative_framing_",
    "legitimation_strategy": "legitimation_strategy_",
    "enemy_moral_status": "enemy_moral_status_",
    "embodiment_mode": "embodiment_mode_",
    "violence_visibility": "violence_visibility_",
    "agency_level": "agency_level_"
}

# German labels for selected categories
labels_map = {
    "actor_configuration": {
        "individual_hero": "Individueller Held",
        "collective_military": "Kollektive Militär",
        "civilian_collective": "Zivilkollektiv",
        "enemy_focus": "Feindfokus",
        "mixed_interaction": "Gemischte Interaktion",
    },
    "narrative_framing": {
        "heroic_combat": "Heroischer Kampf",
        "defensive_war": "Verteidigungskrieg",
        "liberation_mission": "Befreiungsmission",
        "civilizational_struggle": "Zivilisatorischer Kampf",
        "comradely_front": "Kameradschaftlicher Front",
        "technological_modernity": "Technologische Modernität",
        "sacrificial_martyrdom": "Opferbereiter Märtyrer",
        "soldier_as_victim": "Soldat als Opfer",
        "inevitable_victory": "Unvermeidbarer Sieg",
        "administrative_normality": "Administrative Normalität",
        "occupation_normalization": "Besetzungsnormalisierung",
        "humanitarian_relief": "Humanitäre Hilfe",
        "cultural_guardianship": "Kulturelle Bewahrung",
        "none": "Keine",
    },
    "legitimation_strategy": {
        "defensive": "Defensiv",
        "retaliatory": "Vergeltend",
        "preventive": "Präventiv",
        "civilizing": "Zivilisierend",
        "ideological": "Ideologisch",
        "existential_threat": "Existenzielle Bedrohung",
        "humanitarian": "Humanitär",
        "none": "Keine",
    },
    "enemy_moral_status": {
        "legitimate_opponent": "Legitimer Gegner",
        "weaker_enemy": "Schwächerer Gegner",
        "criminalized_enemy": "Kriminalisierter Gegner",
        "dehumanized_enemy": "Entmenschlichter Gegner",
        "existential_threat": "Existenzielle Bedrohung",
        "none": "Keine",
    },
    "embodiment_mode": {
        "disciplined_masculinity": "Disziplinierte Männlichkeit",
        "wounded_body": "Verwundeter Körper",
        "invulnerable_body": "Unverwundbarer Körper",
        "eroticized_body": "Erotisierter Körper",
        "erotic_renunciation": "Erotische Enthaltung",
        "emotionally_restrained": "Emotional Zurückhaltend",
        "emotional_expressivity": "Emotionale Ausdruckskraft",
        "neutral_military_presence": "Neutrale Militärpräsenz",
        "none": "Keine",
    },
    "violence_visibility": {
        "shown": "Gezeigt",
        "implied": "Angedeutet",
        "absent": "Abwesend",
    },
    "agency_level": {
        "low": "Niedrig",
        "medium": "Mittel",
        "high": "Hoch",
    },
}

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def compute_yearly_ratios(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by year with mean proportion of labels.
    prefix: e.g., "narrative_framing_"
    """
    label_cols = [c for c in df.columns if c.startswith(prefix)]
    yearly = df.groupby("year")[label_cols].mean()
    return yearly

def get_max_label(df: pd.DataFrame, col: str):
    """Returns the episode/year with the max value for a label."""
    max_idx = df[col].idxmax()
    max_val = df[col].max()
    return max_idx, max_val

def compute_spearman_trends(df: pd.DataFrame, cols: list):
    """
    Computes Spearman correlation of each label with time.
    Returns a sorted DataFrame with rho and p-value.
    """
    results = []
    time = df.index.values

    for col in cols:
        rho, p = spearmanr(time, df[col])
        results.append({
            "narrative": col.replace("narrative_framing_", ""),
            "spearman_rho": rho,
            "p_value": p
        })

    trend_df = pd.DataFrame(results).sort_values("spearman_rho", ascending=False)
    return trend_df

def compute_dimension_correlations(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    prefix: str,
):
    """
    Computes Pearson correlations between dummy variables of a given dimension.

    """
    results = []

    for a, b in pairs:
        col1 = f"{prefix}{a}"
        col2 = f"{prefix}{b}"

        if col1 not in df.columns or col2 not in df.columns:
            print(f"Skipping pair ({a}, {b}) – column missing")
            continue

        r, p = pearsonr(df[col1], df[col2])

        results.append(
            {
                "category_a": a,
                "category_b": b,
                "pearson_r": r,
                "p_value": p,
            }
        )

    return pd.DataFrame(results)

def detect_narrative_shifts(
    df: pd.DataFrame,
    column: str,
    episode_col: str = "episode",
    model: str = "rbf",
    pen: float = 3,
):
    """
    Detect structural breakpoints in a narrative time series using ruptures.

    Returns:
        list of episode numbers where shifts occur
    """

    # aggregate per episode
    episode_means = (
        df.groupby(episode_col)[column]
        .mean()
        .sort_index()
    )

    series = episode_means.values

    algo = rpt.Pelt(model=model).fit(series)
    breaks = algo.predict(pen=pen)

    # remove final index
    breaks = [b for b in breaks if b < len(series)]

    episode_numbers = episode_means.index[breaks].tolist()

    return episode_numbers

def compute_yearly_keyword_trends(
    df: pd.DataFrame,
    text_column: str = "narrative_summary",
    year_column: str = "year",
    max_features: int = 500,
    top_k: int = 20,
):
    """
    Computes top TF-IDF keywords per year.

    Returns:
        DataFrame with top keywords per year
    """

    texts = df[text_column].fillna("")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
    )

    X = vectorizer.fit_transform(texts)

    terms = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(X.toarray(), columns=terms)
    tfidf_df[year_column] = df[year_column].values

    # average tf-idf per year
    yearly_keywords = tfidf_df.groupby(year_column).mean()

    # extract top words
    top_words = yearly_keywords.apply(
        lambda row: row.nlargest(top_k).index.tolist(), axis=1
    )

    top_words_table = top_words.apply(pd.Series)
    top_words_table.columns = [f"top_{i+1}" for i in range(top_words_table.shape[1])]

    return top_words_table, yearly_keywords

def compute_nmf_topics(
    df: pd.DataFrame,
    text_column: str = "narrative_summary",
    n_topics: int = 5,
    max_features: int = 1000,
    ngram_range: tuple[int,int] = (1,2),
    min_df: int = 5,
    n_top_words: int = 10
):
    """
    Runs NMF topic modeling on narrative summaries.

    Returns:
        df: original DataFrame with dominant topic column added
        W: document-topic matrix
        H: topic-term matrix
        terms: feature names from vectorizer
    """
    texts = df[text_column].fillna("")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df
    )
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_

    # dominant topic for each document
    dominant_topic = np.argmax(W, axis=1)
    df = df.copy()
    df["text_topic"] = dominant_topic

    # Print top words per topic
    for topic_idx, topic in enumerate(H):
        top_terms = [terms[i] for i in topic.argsort()[-n_top_words:][::-1]]
        print(f"\nTopic {topic_idx}: {', '.join(top_terms)}")

    return df, W, H, terms

def compute_topic_profile(df: pd.DataFrame, dummy_cols: list[str]):
    """
    Computes average dummy variables per topic.

    Returns:
        topic_profile: DataFrame, rows = topics, columns = dummy variables
    """
    topic_profile = df.groupby("text_topic")[dummy_cols].mean()
    return topic_profile

# -----------------------------------------------------------------
# Main Analysis
# -----------------------------------------------------------------

def main():
    """Runs analysis on narrative features and saves plots and prints statistics."""

    df = DF.copy()

    # narrative trend analysis
    yearly = compute_yearly_ratios(df, "narrative_framing_")
    columns = [
        "narrative_framing_defensive_war",
        "narrative_framing_inevitable_victory",
    ]

    labels = {
        "narrative_framing_defensive_war": "Verteidigungskrieg",
        "narrative_framing_inevitable_victory": "Unvermeidbarer Sieg",
    }

    plt = plot_narrative_trends_per_episode(df, columns, labels)
    plt.savefig(FIG_NARRATIVES_DEPICTIONS_PATTERN1_FILE)

    columns = [
        "narrative_framing_administrative_normality",
        "narrative_framing_occupation_normalization",
    ]

    labels = {
        "narrative_framing_administrative_normality": "Administrative Normalität",
        "narrative_framing_occupation_normalization": "Besetzungsnormalisierung",
    }

    plt = plot_narrative_trends_per_episode(df, columns, labels)
    plt.savefig(FIG_NARRATIVES_DEPICTIONS_PATTERN2_FILE)

    # output peaks for specific narrative
    col = "narrative_framing_inevitable_victory"
    max_year, max_val = get_max_label(yearly, col)
    print(f"Maximum {col}: {max_val}")
    print(f"Year: {max_year}")

    #compute Spearman correlations
    top_cols = [c for c in yearly.columns if "narrative_framing_" in c]
    trend_df = compute_spearman_trends(yearly, top_cols)
    print(trend_df)

    # plot trends for alle categories with all columns
    for category, prefix in prefixes.items():
        # Dynamically select all dummy columns with the prefix
        cols_in_df = [c for c in df.columns if c.startswith(prefix)]
        if not cols_in_df:
            continue  # skip if none exist

        # Map the original labels (without prefix) to German labels if available
        label_dict = {}
        for c in cols_in_df:
            base_name = c.replace(prefix, "")
            if category in labels_map and base_name in labels_map[category]:
                label_dict[c] = labels_map[category][base_name]
            else:
                label_dict[c] = base_name  # fallback to raw name

        # Plot and save category trends
        plt = plot_category_trends_per_episode(
            df,
            columns=cols_in_df,
            labels=label_dict,
            title=f"{category.replace('_', ' ').title()} Trends"
        )
        fig_file = FIG_DIR / f"{category}_trends.pdf"
        print(f"Saving plot to {fig_file}")
        plt.savefig(fig_file)

        # Plot and save correlations
        if len(cols_in_df) >= 2:
            corr = df[cols_in_df].corr()
            plt = plot_heatmap(
                corr,
                title=f"{category.replace('_', ' ').title()} Korrelationen innerhalb der Dimension",
                row_prefix=prefix,
                col_prefix=prefix
            )
            fig_file = FIG_DIR / f"{category}_within_corr.pdf"
            print(f"Saving correlation heatmap to {fig_file}")
            plt.savefig(fig_file)

    # Plot and save correlations between dimensions
    cross_pairs = [
        ("actor_configuration", "narrative_framing"),
        ("embodiment_mode", "narrative_framing"),
        ("legitimation_strategy", "narrative_framing"),
        ("legitimation_strategy", "enemy_moral_status"),
        ("legitimation_strategy", "agency_level")
    ]
    for dim1, dim2 in cross_pairs:
        cols1 = [c for c in df.columns if c.startswith(prefixes[dim1])]
        cols2 = [c for c in df.columns if c.startswith(prefixes[dim2])]
        if not cols1 or not cols2:
            continue

        corr = DF[cols1 + cols2].corr().loc[cols1, cols2]
        plt = plot_heatmap(
            corr,
            title=f"{dim1.replace('_', ' ').title()} vs {dim2.replace('_', ' ').title()} Korrelationen",
            row_prefix=prefixes[dim1],
            col_prefix=prefixes[dim2]
        )
        fig_file = FIG_DIR / f"{dim1}_vs_{dim2}_corr.pdf"
        print(f"Saving cross-dimension correlation heatmap to {fig_file}")
        plt.savefig(fig_file)

    # print correlations for specific pairs
    narrative_pairs = [
        ("inevitable_victory", "heroic_combat"),
        ("defensive_war", "heroic_combat"),
    ]

    corr_df = compute_dimension_correlations(
        DF,
        pairs=narrative_pairs,
        prefix=prefixes["narrative_framing"],
    )

    print("\nNarrative correlations")
    print(corr_df.to_string(index=False))

    # detect narrative shifts
    narratives_to_test = [
        "narrative_framing_inevitable_victory",
        "narrative_framing_heroic_combat",
        "narrative_framing_defensive_war",
    ]

    for col in narratives_to_test:
        shifts = detect_narrative_shifts(DF, col)
        label = col.replace("narrative_framing_", "")
        print(f"Detected narrative shifts for '{label}' occur at episodes: {shifts}")

    # detect keyword trend in summaries
    keywords_table, yearly_keywords = compute_yearly_keyword_trends(DF)

    print("\nTop keywords per year:")
    print(keywords_table)

    out_file = FIG_DIR / "keyword_trends_by_year.csv"
    keywords_table.to_csv(out_file)

    print(f"Saved keyword trends to {out_file}")

    # plot the keyword trends
    plt = plot_keyword_trends(yearly_keywords)
    fig_file = FIG_DIR / "keyword_trends.pdf"
    plt.savefig(fig_file)
    print(f"Saved keyword trend plot to {fig_file}")

    # topic modeling 
    nmf_df, W, H, terms = compute_nmf_topics(DF, n_topics=5)

    all_dummy_cols = [c for c in DF.columns if any(c.startswith(p) for p in prefixes.values())]

    topic_profile = compute_topic_profile(nmf_df, all_dummy_cols)
    plt = plot_topic_profile(topic_profile)

    fig_file = FIG_DIR / "topic_profile_heatmap.pdf"
    plt.savefig(fig_file)
    print(f"Saved topic profile heatmap to {fig_file}")

    # print some example summaries for each topic
    for topic in range(W.shape[1]):
        print(f"\n{'='*40}\nTopic {topic} examples:")
        examples = nmf_df[nmf_df["text_topic"] == topic]["narrative_summary"].head(5)
        for e in examples:
            print("-", e)
        

if __name__ == "__main__":
    main()