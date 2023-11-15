import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import umap
import matplotlib.patches as mpatches

sns.set(style="white")

COLOR_BY_SPEAKER = {
    "female_hebrews_main": "fuchsia",
    "Nydia Ecury": "deeppink",
    "KompasKorsou": "dodgerblue",
    "Pito Salas": "orange",
    "male_john_main": "green",
    "male_luke_main": "gold",
    "male_matthew_main": "orangered",
    "male_acts_main": "chartreuse",
    "male_jude_main": "peru",
    "male_paul_main": "mediumblue",
    "male_peter_main": "darkred",
    "male_james_main": "springgreen",
    "male_mark_main": "indigo",
}

SPLIT_BY_SPEAKER = {
    "train": [
        "Nydia Ecury",
        "KompasKorsou",
        "male_jude_main",
        "male_james_main",
        "male_peter_main",
        "male_john_main",
        "male_matthew_main",
        "male_acts_main",
        "male_luke_main",
    ],
    "dev": [
        "male_paul_main",
    ],
    "test": [
        "female_hebrews_main",
        "Pito Salas",
    ],
}


def get_umap_projections(
    df: pd.DataFrame, n_components: int, n_neighbors: int, min_dist: float
) -> np.ndarray:
    embeds = np.stack(df["speaker_embedding"].values)
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist
    )
    projs = reducer.fit_transform(embeds)
    return projs


def draw_umap_separated_by_speaker(u: np.ndarray, speaker_labels: np.ndarray):
    max_alpha = 1
    min_alpha = 0.8
    max_size = 55
    min_size = 40
    n_components = u.shape[1]
    unique_speakers, speaker_counts = np.unique(speaker_labels, return_counts=True)
    min_count = min(speaker_counts)
    max_count = max(speaker_counts)
    # Define height bounds
    min_height_multiplier = 0.45
    max_height_multiplier = 3
    # Calculate height multipliers
    height_multipliers = {
        speaker: min_height_multiplier
        + (max_height_multiplier - min_height_multiplier)
        * (count - min_count)
        / (max_count - min_count)
        for speaker, count in zip(unique_speakers, speaker_counts)
    }
    # Calculate height ratios for each subplot
    height_ratios = [
        height_multipliers[speaker]
        for group in SPLIT_BY_SPEAKER
        for speaker in SPLIT_BY_SPEAKER[group]
    ]
    # Create figure and GridSpec with custom heights
    fig = plt.figure(figsize=(12 + 1, 0.7 * sum(height_ratios) + 1.5))
    gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios)
    current_ax = 0
    level_colors = {
        "train": "lemonchiffon",
        "dev": "lightcyan",
        "test": "lavenderblush",
    }
    for group in ["train", "dev", "test"]:
        speakers_in_group = SPLIT_BY_SPEAKER[group]
        for idx, speaker in enumerate(speakers_in_group):
            speaker_u = u[speaker_labels == speaker]
            n_obs = speaker_u.shape[0]
            alpha = max_alpha - (max_alpha - min_alpha) * (n_obs - min_count) / (
                max_count - min_count
            )
            size = max_size - (max_size - min_size) * (n_obs - min_count) / (
                max_count - min_count
            )
            ax = fig.add_subplot(gs[current_ax])
            ax.set_facecolor(level_colors[group])
            scatter = ax.scatter(
                speaker_u[:, 0],
                speaker_u[:, 1]
                if n_components == 2
                else np.random.uniform(0, 1, n_obs),
                marker=".",
                s=size,
                alpha=alpha,
                c=[COLOR_BY_SPEAKER[speaker]],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                1.04,
                0.5,
                f"{speaker}\n({n_obs} utterances)",
                transform=ax.transAxes,
                fontdict={"fontsize": 10},
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(level_colors[group])
                spine.set_linewidth(2)
            current_ax += 1
    train_patch = mpatches.Patch(color="khaki", label="Train")
    dev_patch = mpatches.Patch(color="lightskyblue", label="Dev")
    test_patch = mpatches.Patch(color="lightpink", label="Test")
    patches = [train_patch, dev_patch, test_patch]
    plt.legend(handles=patches, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -1.7))
    plt.subplots_adjust(right=0.8)
    fig.suptitle(
        "Speaker Bias: 1D UMAP Projections of Resemblyzer Embeddings by Speaker",
        fontsize=16,
        fontweight="bold",
    )
    return fig


# Read in the data
df = pd.read_csv("corpus.csv")
# Convert array strings into array objects
df["speaker_embedding"] = df["speaker_embedding"].apply(
    lambda x: np.fromstring(x[1:-1], sep=" ")
)
# Hyperparameter combinations
hyperparameters = [
    {"n_neighbors": 30, "min_dist": 0.1},
]

# print the mean, median, mode, min, and max of the duration_ms column
print("Duration Statistics:")
print(df["duration_ms"].describe())
print("sum: ", df["duration_ms"].sum())


for params in hyperparameters:
    print(
        f"Computing UMAP projections with n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}..."
    )
    projections = get_umap_projections(
        df,
        n_components=1,
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
    )
    fig = draw_umap_separated_by_speaker(projections, df["speaker"].values)
    filename = f"speaker_bias.png"
    fig.savefig(filename)
    print(f"Saved figure to {filename}")
