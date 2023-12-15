"""Creates and pushes a DatasetDict to HuggingFace Datasets Hub."""

import pandas as pd
from datasets import DatasetDict, Dataset, Audio
from huggingface_hub import HfFolder


HF_KEY = ""
HfFolder.save_token(HF_KEY)

# Read in csv
df = pd.read_csv(
    "diarized_corpus.csv",
    usecols=["file_name", "train_dev_test_split", "unpunctuated_text"],
)
df = df.rename(columns={"unpunctuated_text": "transcript", "file_name": "audio"})
df["audio"] = df["audio"].apply(lambda x: f"corpus_audio/{x}")

# Create DatasetDict where each split is a Dataset
dataset_dict = DatasetDict()
for split in ["train", "dev", "test"]:
    dataset_dict[split] = Dataset.from_pandas(
        df[df["train_dev_test_split"] == split][["audio", "transcript"]]
    ).cast_column("audio", Audio())

# Push to HuggingFace Datasets Hub
dataset_dict.push_to_hub("papi_asr_corpus")
