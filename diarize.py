from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import os

import numpy as np
import pandas as pd
from functools import cache
from umap import UMAP
import sounddevice as sd
from sklearn.metrics import f1_score
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from rich import print
from sklearn.linear_model import LogisticRegression
import rich
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from spectralcluster import (
    SpectralClusterer,
    RefinementOptions,
    ThresholdType,
    ICASSP2018_REFINEMENT_SEQUENCE,
)

from lists import UTTERANCES_TO_IGNORE_IN_CLUSTERING, BAD_UTTERANCES
from plot_diarization import plot_diarization

RANDOM_SEED = 50  # 42
EXCLUDE_PITO = True
np.random.seed(RANDOM_SEED)
N_BIBLE_SAMPLES = None
KNOWN_SPEAKERS = {  # Some quick hand-labeling
    104010101: "male_john_main",
    104010201: "male_john_main",
    104010301: "male_john_main",
    104010401: "male_john_main",
    104010501: "male_john_main",
    104010601: "male_john_main",
    104010701: "male_john_main",
    104010801: "male_john_main",
    104010901: "male_john_main",
    104011001: "male_john_main",
    104011101: "male_john_main",
    104011201: "male_john_main",
    104011301: "male_john_main",
    104011401: "male_john_main",
    104011501: "male_john_main",
    104011601: "male_john_main",
    104011701: "male_john_main",
    104011801: "male_john_main",
    104011901: "male_john_main",
    104012202: "nbg",
    104012203: "nbg",
    104012602: "ndst",
    104012702: "ndst",
    104013001: "ndst",
    104013101: "ndst",
    101050301: "male_jesus_main",
    101050401: "male_jesus_main",
    101050501: "male_jesus_main",
    101050601: "male_jesus_main",
    101050701: "male_jesus_main",
    101050801: "male_jesus_main",
    101050901: "male_jesus_main",
    101051001: "male_jesus_main",
    101051101: "male_jesus_main",
    101051201: "male_jesus_main",
    101051301: "male_jesus_main",
    101051401: "male_jesus_main",
    101051501: "male_jesus_main",
    101051601: "male_jesus_main",
    101051701: "male_jesus_main",
    101051801: "male_jesus_main",
    101051901: "male_jesus_main",
    101052001: "male_jesus_main",
    101052101: "male_jesus_main",
    101052201: "male_jesus_main",
    101052301: "male_jesus_main",
    101052401: "male_jesus_main",
    101052501: "male_jesus_main",
    101052601: "male_jesus_main",
    101052701: "male_jesus_main",
    101052801: "male_jesus_main",
    101052901: "male_jesus_main",
    101053001: "male_jesus_main",
    101053101: "male_jesus_main",
    101053201: "male_jesus_main",
    101053301: "male_jesus_main",
    101053401: "male_jesus_main",
    101053501: "male_jesus_main",
    101053601: "male_jesus_main",
    101053701: "male_jesus_main",
    101061302: "male_jesus_main",
    101041601: "ang",
    101041602: "ang",
    101012101: "god",
    101012301: "gabr",
    101012302: "gabr",
    119111801: "ang2",
    101010101: "male_matthew_main",
    101010201: "male_matthew_main",
    101010301: "male_matthew_main",
    101010401: "male_matthew_main",
    101010501: "male_matthew_main",
    101010601: "male_matthew_main",
    101010701: "male_matthew_main",
    101010801: "male_matthew_main",
    101010901: "male_matthew_main",
    101011001: "male_matthew_main",
    101011101: "male_matthew_main",
    101011201: "male_matthew_main",
    101011301: "male_matthew_main",
    101011401: "male_matthew_main",
    101011501: "male_matthew_main",
    101011601: "male_matthew_main",
    101011701: "male_matthew_main",
    101011801: "male_matthew_main",
    101011901: "male_matthew_main",
    119120101: "female_hebrews_main",
    119120201: "female_hebrews_main",
    119120301: "female_hebrews_main",
    119120401: "female_hebrews_main",
    119120501: "female_hebrews_main",
    119111001: "female_hebrews_main",
    119111101: "female_hebrews_main",
    119111201: "female_hebrews_main",
    119111301: "female_hebrews_main",
    119111401: "female_hebrews_main",
    119111501: "female_hebrews_main",
    119111601: "female_hebrews_main",
    119111701: "female_hebrews_main",
    119132101: "female_hebrews_main",
    103030001: "female_hebrews_main",
    101030001: "female_hebrews_main",
    119132103: "female_hebrews_main",
    103030001: "female_hebrews_main",
    102060101: "male_mark_main",
    102060102: "male_mark_main",
    102060303: "male_mark_main",
    102060601: "male_mark_main",
    102060701: "male_mark_main",
    102061701: "male_mark_main",
    102061901: "male_mark_main",
    102062101: "male_mark_main",
    102062201: "male_mark_main",
    102062401: "male_mark_main",
    102062701: "male_mark_main",
    102063001: "male_mark_main",
    103010101: "male_luke_main",
    103010201: "male_luke_main",
    103010301: "male_luke_main",
    103010401: "male_luke_main",
    103010501: "male_luke_main",
    103010601: "male_luke_main",
    103010701: "male_luke_main",
    103010801: "male_luke_main",
    103010901: "male_luke_main",
    103011001: "male_luke_main",
    103011101: "male_luke_main",
    103011201: "male_luke_main",
    103010101: "male_acts_main",
    103010201: "male_acts_main",
    103010301: "male_acts_main",
    103010401: "male_acts_main",
    105011001: "male_acts_main",
    105011302: "male_acts_main",
    105012301: "male_acts_main",
    105020301: "male_acts_main",
    105020601: "male_acts_main",
    105030202: "male_acts_main",
    106010101: "male_paul_main",
    106010201: "male_paul_main",
    106010301: "male_paul_main",
    106010401: "male_paul_main",
    106010501: "male_paul_main",
    106010601: "male_paul_main",
    106010701: "male_paul_main",
    106010801: "male_paul_main",
    106010901: "male_paul_main",
    106011001: "male_paul_main",
    106011101: "male_paul_main",
    106011201: "male_paul_main",
    106011301: "male_paul_main",
    106011401: "male_paul_main",
    106011501: "male_paul_main",
    106011601: "male_paul_main",
    106011701: "male_paul_main",
    106101202: "male_paul_main",
    108080302: "male_paul_main",
    120010101: "male_james_main",
    120010201: "male_james_main",
    120010301: "male_james_main",
    120010401: "male_james_main",
    120010501: "male_james_main",
    120010601: "male_james_main",
    120010701: "male_james_main",
    120010801: "male_james_main",
    120010901: "male_james_main",
    120011001: "male_james_main",
    120011101: "male_james_main",
    120011201: "male_james_main",
    120011301: "male_james_main",
    120011401: "male_james_main",
    120011501: "male_james_main",
    120011601: "male_james_main",
    120011701: "male_james_main",
    120011801: "male_james_main",
    120011901: "male_james_main",
    120012001: "male_james_main",
    120012101: "male_james_main",
    120012201: "male_james_main",
    120012301: "male_james_main",
    120012401: "male_james_main",
    120012501: "male_james_main",
    120012601: "male_james_main",
    120012701: "male_james_main",
    126010101: "male_jude_main",
    126010201: "male_jude_main",
    126010301: "male_jude_main",
    126010401: "male_jude_main",
    126010501: "male_jude_main",
    126010601: "male_jude_main",
    126010701: "male_jude_main",
    126010801: "male_jude_main",
    126010901: "male_jude_main",
    126011101: "male_jude_main",
    126011201: "male_jude_main",
    126011301: "male_jude_main",
    126011601: "male_jude_main",
    126011701: "male_jude_main",
    126011301: "male_jude_main",
    121010101: "male_peter_main",
    121010201: "male_peter_main",
    121010301: "male_peter_main",
    121010401: "male_peter_main",
    121010501: "male_peter_main",
    121010601: "male_peter_main",
    121010701: "male_peter_main",
    121010801: "male_peter_main",
    121011001: "male_peter_main",
    121012001: "male_peter_main",
    121013001: "male_peter_main",
    121014001: "male_peter_main",
    104183902: "crazy_fx",
}

CLASS_WEIGHTS = {
    "male_matthew_main": 1.09649,
    "_other_": 0.15,
    "male_jesus_main": 0.5952,
    "male_mark_main": 1.98,
    "male_acts_main": 3.4,
    "male_luke_main": 2.60416,
    "male_john_main": 1.35,
    "male_paul_main": 1.30208,
    "female_hebrews_main": 1.48809,
    "male_james_main": 0.7716049,
    "male_jude_main": 1.488095,
    "pito": 0.32051,
    "male_peter_main": 1.98095,
}

print("👷 Beginning script...")
print(f"🔢 {len(set(KNOWN_SPEAKERS.values()))} unique labels in labeled data")
_DF = pd.read_csv("corpus.csv")
_DF["speaker_embedding"] = _DF["speaker_embedding"].apply(
    lambda x: np.fromstring(x[1:-1], sep=" ")
)
_X = _DF["speaker_embedding"].tolist()
STD_SCALER = StandardScaler()
STD_SCALER.fit(_X)


@cache
def load_data(use_known_speakers: bool = False) -> pd.DataFrame:
    bible_df = _DF[_DF["utterance_id"] < 200000000]
    bible_df = bible_df[~bible_df["utterance_id"].isin(BAD_UTTERANCES)]
    bible_df = bible_df[
        ~bible_df["utterance_id"].isin(UTTERANCES_TO_IGNORE_IN_CLUSTERING)
    ]
    known_speakers_df = bible_df[bible_df["utterance_id"].isin(KNOWN_SPEAKERS.keys())]
    if use_known_speakers:
        bible_df = known_speakers_df
    else:
        if N_BIBLE_SAMPLES is not None:
            bible_df = bible_df.sample(n=N_BIBLE_SAMPLES, random_state=RANDOM_SEED)
        bible_df = pd.concat([bible_df, known_speakers_df])
        bible_df = bible_df.drop_duplicates(subset="utterance_id")
    if not EXCLUDE_PITO:
        pito_df = _DF[_DF["speaker"] == "Pito Salas"]
        return pd.concat([bible_df, pito_df])
    else:
        return bible_df


@cache
def get_speaker(utterance_id: int) -> str:
    if utterance_id in KNOWN_SPEAKERS:
        if (
            sum([KNOWN_SPEAKERS[utterance_id] == v for v in KNOWN_SPEAKERS.values()])
            >= 5
        ):
            return KNOWN_SPEAKERS[utterance_id]
        else:
            return "_other_"
    else:
        return "pito"


@cache
def get_importances() -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    df = load_data(use_known_speakers=True)
    X = df["speaker_embedding"].tolist()
    X = STD_SCALER.transform(X)
    y = df["utterance_id"].apply(get_speaker)
    y.index = df["utterance_id"]
    # # Calculate class weights manually
    # class_frequencies = Counter(y)
    # total_samples = len(y)
    # class_weights = {
    #     class_label: total_samples / (len(class_frequencies) * freq)
    #     for class_label, freq in class_frequencies.items()
    # }
    class_weights = CLASS_WEIGHTS
    clf = LogisticRegression(
        class_weight=class_weights, random_state=RANDOM_SEED, max_iter=3000
    )
    clf.fit(X, y.tolist())
    abs_value_coefs = np.abs(clf.coef_)
    mean_coefs = np.mean(abs_value_coefs, axis=0)
    importances = MinMaxScaler().fit_transform(mean_coefs.reshape(-1, 1)).flatten()
    return importances, X, y


def play_fast_snippet_of_wav(
    fpath: str, begin_ms: int = 1000, end_ms: int = 3000, speed: float = 1.9
) -> None:
    try:
        # Load the file with pydub
        audio = AudioSegment.from_file(fpath)
        if len(audio) > 2500:
            audio = audio[begin_ms:end_ms]
        if speed is not None and speed != 1:
            audio = audio.speedup(playback_speed=speed)
        audio = audio.fade_in(200).fade_out(200)
        # Convert the PyDub AudioSegment to a NumPy array for playback
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = np.reshape(samples, (-1, 2))
        # Play the audio
        sd.play(samples, audio.frame_rate)
    except:
        return
    sd.wait()


def spectral_cluster(
    X: List[np.ndarray],
    proportion_weighted: float,
    n_umap_components: Optional[int],
    n_lda_components: Optional[int],
    **kwargs,
) -> List[int]:
    X = STD_SCALER.transform(X)
    importances, X_bible, y_bible = get_importances()
    assert X.shape[1] == importances.shape[0]
    weighted_X = X * importances  # Weighted by logistic regression importance
    X = X * (1 - proportion_weighted) + weighted_X * proportion_weighted
    if n_umap_components is not None or n_lda_components is not None:
        weighted_X_bible = X_bible * importances
        X_bible = (
            X_bible * (1 - proportion_weighted) + weighted_X_bible * proportion_weighted
        )
    # Apply UMAP
    if n_umap_components is not None:
        reducer = UMAP(
            n_components=n_umap_components,
            random_state=RANDOM_SEED,
            n_jobs=1,
            target_metric="categorical",
        )
        encoded_y_bible = LabelEncoder().fit_transform(y_bible)
        reducer.fit(X_bible, encoded_y_bible)
        X = reducer.transform(X)
    # Apply LDA
    if n_lda_components is not None:
        y_bible = y_bible.index.map(get_speaker)
        weighted_X_bible = X_bible * importances
        X_bible = (
            X_bible * (1 - proportion_weighted) + weighted_X_bible * proportion_weighted
        )
        lda = LDA(n_components=n_lda_components)
        lda.fit(X_bible, y_bible)
        X = lda.transform(X)
    # Fit Spectral Clustering
    refinement_options = RefinementOptions(
        p_percentile=kwargs["p_percentile"],
        gaussian_blur_sigma=kwargs["gaussian_blur_sigma"],
        thresholding_type=ThresholdType.RowMax,
        thresholding_soft_multiplier=0.12,
        refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE,
    )
    clusterer = SpectralClusterer(
        min_clusters=kwargs["min_clusters"],
        max_clusters=kwargs["max_clusters"],
        refinement_options=refinement_options,
        max_spectral_size=kwargs["max_spectral_size"],
    )
    labels = clusterer.predict(np.array(X))
    return labels, clusterer


def get_bible_f1s_and_clusters(df: pd.DataFrame) -> Tuple[dict, dict]:
    assert "cluster" in df.columns
    known_ids = set(KNOWN_SPEAKERS.keys())  # Filter to known bible speakers
    df = df[df["utterance_id"].isin(known_ids)].copy()
    df["speaker_class"] = df["utterance_id"].apply(get_speaker)
    bible_f1s = {}
    bible_clusters = {}
    for speaker, group in df.groupby("speaker_class"):
        if speaker == "_other_":
            continue
        speaker_cluster = group["cluster"].value_counts().index[0]
        if speaker_cluster == -1:
            speaker_cluster = group["cluster"].value_counts().index[1]
        is_spkr_most_common_cluster = group["cluster"] == speaker_cluster
        tp = len(group[is_spkr_most_common_cluster])
        fp = len(group[~is_spkr_most_common_cluster])
        fn = len(df[df["cluster"] == speaker_cluster]) - tp
        f1 = tp / (tp + 0.5 * (fp + fn))
        bible_f1s[speaker] = f1
        bible_clusters[speaker] = speaker_cluster
    return bible_f1s, bible_clusters


def get_results_info(predicted_clusters: List[int], df: pd.DataFrame) -> tuple:
    df["cluster"] = predicted_clusters
    bible_f1s, bible_clusters = get_bible_f1s_and_clusters(df)
    df["is_pito"] = df["speaker"].apply(lambda x: "Pito" in x)
    if EXCLUDE_PITO:
        pito_cluster = None
        confusion_dict = None
        all_f1s = list(bible_f1s.values())
    else:
        pito_cluster = df[df["is_pito"]]["cluster"].value_counts().index[0]
        if pito_cluster == -1:
            pito_cluster = df[df["is_pito"]]["cluster"].value_counts().index[1]
        df["is_pito_most_common_cluster"] = df["cluster"] == pito_cluster
        pito_f1 = f1_score(df["is_pito"], df["is_pito_most_common_cluster"])
        confusion_dict = {
            "TP": df[
                (df["is_pito"] == True) & (df["is_pito_most_common_cluster"] == True)
            ].shape[0],
            "TN": df[
                (df["is_pito"] == False) & (df["is_pito_most_common_cluster"] == False)
            ].shape[0],
            "FP": df[
                (df["is_pito"] == False) & (df["is_pito_most_common_cluster"] == True)
            ].shape[0],
            "FN": df[
                (df["is_pito"] == True) & (df["is_pito_most_common_cluster"] == False)
            ].shape[0],
        }
        all_f1s = list(bible_f1s.values()) + [pito_f1]
    macro_f1 = np.mean(all_f1s)
    return macro_f1, pito_cluster, confusion_dict, bible_f1s, bible_clusters


def run_experiments(param_combos: List[dict], df: pd.DataFrame) -> tuple:
    best_f1 = 0
    best_pito_cluster = None
    best_conf_dict = None
    best_prms = None
    best_lbls = None
    best_bible_f1s = None
    best_clusterer = None
    for params in param_combos:
        pred_labels, clusterer = spectral_cluster(
            X=X,
            **params,
        )
        f1, pito_cluster, conf_dict, bible_f1s, bible_clusters = get_results_info(
            pred_labels, df
        )
        print(f"🧪 - F1: {f1:.2f} - NCls: {len(set(pred_labels))} - Prms: {params}")
        if f1 > best_f1:
            best_f1 = f1
            best_prms = params
            best_lbls = pred_labels
            best_pito_cluster = pito_cluster
            best_conf_dict = conf_dict
            best_bible_f1s = bible_f1s
            best_clusterer = clusterer
    df["cluster"] = best_lbls
    print("...")
    print(f"🏆 - F1: {best_f1:.2f} - NCls: {len(set(best_lbls))} - Prms: {best_prms}")
    return (
        best_f1,
        best_pito_cluster,
        best_conf_dict,
        best_bible_f1s,
        best_clusterer,
        bible_clusters,
    )


def get_centroids(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    centroids = {}
    for cluster in df["cluster"].unique():
        cluster_df = df[df["cluster"] == cluster]
        centroid = np.mean(cluster_df["speaker_embedding"].tolist(), axis=0)
        centroids[cluster] = centroid
    return centroids


def get_closest_centroid_distance(x: np.ndarray, centroids: Dict[int, np.ndarray]):
    distances = []
    for centroid in centroids.values():
        distances.append(np.linalg.norm(x - centroid))
    return min(distances)


PARAM_COMBOS = [
    {
        "proportion_weighted": 0.5,  # 0.8
        "n_umap_components": 75,
        "n_lda_components": None,
        "min_clusters": 14,
        "max_clusters": 50,
        "p_percentile": 0.89,
        "gaussian_blur_sigma": 0.218,
        "max_spectral_size": 8_000,
    },
]


###################
# RUN EXPERIMENTS #
###################

df = load_data()
X = df["speaker_embedding"].tolist()
f1, pito_cluster, conf_dict, bible_f1s, clusterer, bible_clusters = run_experiments(
    PARAM_COMBOS, df
)

if not EXCLUDE_PITO:
    print("📋 Pito Confusion Matrix:")
    print("\t❌ Predicted Pito but was something else (FP): " + str(conf_dict["FP"]))
    print("\t❌ Predicted something else but was Pito (FN): " + str(conf_dict["FN"]))
    print("\t✅ Predicted Pito and was Pito (TP): " + str(conf_dict["TP"]))
    print("👨 Pito Cluster:", pito_cluster)
print("📖 Bible Clusters:")
for speaker, cluster in bible_clusters.items():
    print(f"\t{speaker}: {cluster}")
print("📖 Bible F1s:")
for speaker, f1 in bible_f1s.items():
    print(f"\t{speaker}: {f1:.2f}")
print("📊 Cluster Frequencies:")
print("\t" + "\n\t".join(df["cluster"].value_counts().__str__().split("\n")[1:-1]))

centroids = get_centroids(df)
df["degree_of_uncertainty"] = df["speaker_embedding"].apply(
    lambda x: get_closest_centroid_distance(x, centroids)
)
df["degree_of_uncertainty"] = (
    MinMaxScaler()
    .fit_transform(df["degree_of_uncertainty"].values.reshape(-1, 1))
    .flatten()
)
df.drop(columns=["speaker_embedding"]).to_csv("clustered_corpus.csv", index=False)

##########################
# PLOT FINAL DIARIZATION #
##########################

cluster_labels = df["cluster"].unique()
cluster_w_speaker_classes = {}
for cluster_label in cluster_labels:
    speaker_classes = []
    if cluster_label == pito_cluster:
        speaker_classes.append("Pito Salas")
    for speaker, cluster in bible_clusters.items():
        if cluster == cluster_label:
            speaker_classes.append(speaker.split("_")[1].title())
    speaker_classes = "\n".join(speaker_classes)
    val = f"{cluster_label}\n{speaker_classes}"
    cluster_w_speaker_classes[cluster_label] = val
df["speaker_classes_of_cluster"] = df["cluster"].map(cluster_w_speaker_classes)

df["author"] = df["speaker"].apply(lambda x: x.split("_")[1].title() if "_" in x else x)
df["author"] = df["author"].apply(lambda x: f"author: {x}" if x != "Pito Salas" else x)
plot_diarization(df)

##########################################
# LABEL UNCERTAIN AS BAD, IGNORE, & KEEP #
##########################################

user_input = input("Would you like to label uncertain utterances? (y/n):")
if user_input == "y":
    uncertain_df = df[df["degree_of_uncertainty"] > 0.5]
    uncertain_df = uncertain_df.sort_values(by="degree_of_uncertainty", ascending=False)
    ignore, keep, bad = [], [], []
    for i, row in uncertain_df.iterrows():
        fpath = os.path.join("corpus_audio", row["file_name"])
        print(f"📝 {row['utterance_id']}--{row['duration_ms'] / 1000:.2f}s")
        play_fast_snippet_of_wav(fpath, begin_ms=1, end_ms=-1, speed=1)
        given_input = input(
            "Select one: finish labeling (f), ignore (i), keep (k), bad (b):"
        )
        if given_input == "f":
            break
        elif given_input == "i":
            ignore.append(row["utterance_id"])
        elif given_input == "k":
            keep.append(row["utterance_id"])
        elif given_input == "b":
            bad.append(row["utterance_id"])
    print("📋 Bad:")
    print(bad)
    print("📋 Ignore:")
    print(ignore)

########################################################
# LISTENING TO/ASSESING QUALITY OF CLUSTERS ON THE FLY #
########################################################

user_input = input("Would you like to listen to a speaker cluster? (y/n):")
if user_input == "y":
    user_input = input("Pick a cluster to listen to:")
    n_seconds = input("For how many seconds would like to listen?")
    n_seconds = int(n_seconds)
    selected_cluster = int(user_input)
    while selected_cluster in cluster_labels:
        fnames = df[df["cluster"] == selected_cluster]["file_name"]
        fnames = fnames.sample(len(fnames)).to_list()
        fpaths = [os.path.join("corpus_audio", fname) for fname in fnames]
        n_files_to_listen_to = int(n_seconds / 2)
        for fpath in fpaths[:n_files_to_listen_to]:
            play_fast_snippet_of_wav(fpath, begin_ms=1000, end_ms=3000)
        user_input = input("Pick a cluster to listen to (f=finish):")
        if user_input == "f":
            break
        selected_cluster = int(user_input)
chosen_dev_cluster = input("Pick a cluster to use for dev set:")

#####################
# SAVING NEW CORPUS #
#####################
try:
    BAD_UTTERANCES.update(bad)
except NameError:
    pass
new_df = _DF.copy()
new_df = new_df[~new_df["utterance_id"].isin(BAD_UTTERANCES)]
new_df.set_index("utterance_id", inplace=True)
df.set_index("utterance_id", inplace=True)
new_df["diarized_speaker"] = df["cluster"]
new_df.reset_index(inplace=True)
new_df['train_dev_test_split'] = 'train'  # default to 'train'
new_df.loc[new_df['diarized_speaker'] == int(chosen_dev_cluster), 'train_dev_test_split'] = 'dev'
new_df.loc[new_df['speaker'] == 'Pito Salas', 'train_dev_test_split'] = 'test'
new_df.to_csv("diarized_corpus.csv", index=False)
print(new_df['train_dev_test_split'].value_counts())
print("👷 Done!")
