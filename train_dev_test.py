import pandas as pd

SPEAKERS_BY_SPLIT = {
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


split_by_speaker = {
    speaker: split
    for split, speakers in SPEAKERS_BY_SPLIT.items()
    for speaker in speakers
}
print(split_by_speaker)
df = pd.read_csv("corpus.csv")
print(df.columns)
df["train_dev_test_split"] = df["speaker"].map(split_by_speaker)
print(df)
df.to_csv("corpus.csv", index=False)
