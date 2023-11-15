"""
The following script "builds" the final corpus by:

    1. Concatenating all the audio files into a single 16kHz, 16bit, mono wav file.
    2. Creating the .csv file that is indexed by utterance and contains all pertinent metadata.

The resulting corpus is therefore a single wav file and a single csv file.

..

The final .csv has the following columns:

    - utterance_id (int):
        A sortable, unique identifier for each utterance. A breakdown of the significance
        of each can be found *later in this docstring.

    - file_name (str):
        The file name of the utterance's audio file.

    - train_dev_test_split (str):
        The train/dev/test split to which the utterance is assigned.

    - duration_ms (int):
        The duration of the utterance in milliseconds.

    - unpunctuated_text (str):
        The unpunctuated text of the utterance.

    - punctuated_text (str):
        The punctuated text of the utterance.

    - principle_speaker_partition (str | NaN):
        If from the bible.is New Testament audio, the "principle speaker" associated with
        the bible recording's "principle speaker partition" from which the utterance
        derives. More explanation can be found **later in this docstring.

    - speaker_embedding (np.ndarray):
        The (256,) speaker embedding vector for the utterance.

    - speaker (str):
        The speaker slug name for the utterance.

    - speaker_was_inferred (bool):
        Whether the speaker was inferred from its embedding.

    - speaker_is_male (bool):
        Whether the given speaker's voice is judged to be like that of a biological male.

    - speaker_comment (str | NaN):
        A brief comment about the speaker, if any. E.g. accent, is a native speaker, age,
        etc.

    - recording_environment_comment (str | NaN):
        A brief comment about the recording environment or nature/quality of the
        recording, if any.

    - license (str):
        The license under which the utterance's audio is released.

...

*The utterance_id is a sortable, 9-digit unique identifier for each utterance.

    For the bible.is New Testament audio, the number has 4 groups (E.g:
    {1}{01}{01}{01}{01}) where:

        - The 1st group/digit denotes the source process of the utterance.
            - 1 = bible.is New Testament
            - 2 = Miscellaneous internet audio
            - 3 = Recorded first-hand

        - The 2nd group (2 digits) is the book number, where 01=Matthew, 02=Mark, etc.

        - The 3rd group (2 digits) is the chapter number.

        - The 4th group (2 digits) is the verse number.

        - The 5th group (2 digits) denotes the utterance number within the verse.
          NOTE: if the verse is 00, then the utterance is an introductory utterance
          (chapter audios are preceded by short statements of the book/chapter).

    For miscellaneous internet audio and recorded first-hand audio, the number has 2
    (Egroups (E.g: {2}{00000001}) where:

        - The 1st group/digit denotes the source process of the utterance.
            - 1 = bible.is New Testament
            - 2 = Miscellaneous internet audio
            - 3 = Recorded first-hand

        - The 2nd group (8 digits) denotes the utterance number.

...

**Regarding the notion of a "principle speaker partition", the idea is as follows...

Different partitions of the bible data have different principle speakers.

The bible.is New Testament audio a dramatic reading of the New Testament where any given
book is *mostly* read by a voice intended to represent the book's author, with direct
speech (quoted from other biblical characters) read by other voices. Since there is no
documentation (to our knowledge) of when these character voices insert, we resort to
inferring the speaker from the speaker embeddings (calculated using `resemblyzer`).

However, one basic heuristic that informed our inference was this notion of therre being
a "principle speaker" for any given book / section--the main narration voice that
represents the book's author. We also extend this heuristic to include the notions that
1. the same voice introduces each chapter, and 2. the same voice narrates all of Jesus'
utterances.

Thus, the "principle speaker partition" derives from these heuristic patterns that we
have observed.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict
from copy import deepcopy
import os
import time

import numpy as np
import pandas as pd
import spacy
import soundfile as sf
from resemblyzer import preprocess_wav, VoiceEncoder

from schema import (
    Span,
    load_mp3,
    SILENCE_TOKEN,
    join_list_of_tokens,
    is_punctuation,
    unpunctuate_papiamentu,
    SAMPLE_RATE,
    play_audio,
    LEFT_COLLAPSING,
    RIGHT_COLLAPSING,
)
from mappings import (
    SPEAKER_COMMENTS,
    RECORDING_ENV_COMMENTS,
    SPEAKER_BY_OTHER_MP3S,
    LICENSE_BY_SPEAKER,
    IS_MALE_BY_SPEAKER,
)
from utils import (
    get_mp3_file_path_from_chapter_slug,
    get_alignment_file_path_from_chapter_slug,
    ScrapedBible,
    OTHER_MP3S_DIR,
    ALIGNMENTS_DIR,
    OTHER_MP3S_TRANSCRIPTIONS_DIR,
)
from plot_speakers import get_speaker_plot


FINAL_AUDIO_DIR = "corpus_audio"
FINAL_METADATA_PATH = "corpus.csv"

VOICE_ENCODER = VoiceEncoder()


def tprint(*args) -> None:
    """Prints with a timestamp."""
    t = f"[{time.strftime('%H:%M:%S')}]"
    print(t, *args)


@dataclass
class Utterance:
    speaker_embedding: Optional[np.ndarray]
    span: Optional[Span]
    audio: Optional[np.ndarray]
    punctuated: Optional[str]
    unpunctuated: Optional[str]

    @classmethod
    def from_span(cls, span: Span) -> "Utterance":
        """Creates an Utterance from a Span."""
        return cls(
            speaker_embedding=None,
            span=span,
            audio=span.get_audio(),
            punctuated=span.get_punctuated_text(),
            unpunctuated=" ".join(span.get_unpunctuated_tokens_as_strings()),
        )

    @classmethod
    def from_files(cls, mp3_path: str, transcription_path: str) -> "Utterance":
        """Creates an Utterance from file paths."""
        audio = load_mp3(mp3_path)
        with open(transcription_path, "r") as f:
            punctuated = f.read()
        unpunctuated = unpunctuate_papiamentu(punctuated)
        return cls(
            speaker_embedding=None,
            span=None,
            audio=audio,
            punctuated=punctuated,
            unpunctuated=unpunctuated,
        )

    @property
    def utterance_num(self) -> int:
        if self.span is not None:
            return self.span.utterance_num
        else:
            return self.utterance_num

    def embed_speaker(self) -> None:
        wav = preprocess_wav(self.audio)
        embedding = VOICE_ENCODER.embed_utterance(wav)
        self.speaker_embedding = embedding


def recursively_collapse_sentence_strings(sentences: List[str]) -> List[str]:
    sentences_copy = sentences[:]
    for i, sent in enumerate(sentences):
        if is_punctuation(sent):
            n_leftward = sum([c in LEFT_COLLAPSING for c in sent])
            n_rightward = sum([c in RIGHT_COLLAPSING for c in sent])
            if n_leftward >= n_rightward:
                items_before = sentences_copy[: i - 1]
                items_after = sentences_copy[i + 1 :]
                merged = sentences_copy[i - 1] + sent
                sentences_copy = items_before + [merged] + items_after
            else:
                items_before = sentences_copy[:i]
                items_after = sentences_copy[i + 2 :]
                merged = sent + sentences_copy[i + 1]
                sentences_copy = items_before + [merged] + items_after
            sentences_copy = recursively_collapse_sentence_strings(sentences_copy)
            break
    return sentences_copy


def tokenize_into_utterances(spans: List[Span]) -> List[Utterance]:
    """Uses an out-of-the-box sentence tokenizer then conservative rule-based
    heuristics are then used to split the sentences further (if needed) into utterances.
    """
    nlp = spacy.load("es_core_news_sm")

    new_spans = []
    for span in spans:
        # Extract word-level tokens (w/ punctuation, bypassing audio silences)
        word_level_tokens = []
        word_level_tokens_idxs = []
        for i, audio_token in enumerate(span.tokens):
            if isinstance(audio_token, str):  # Is punctuation
                word_level_tokens.append(audio_token)
                word_level_tokens_idxs.append(i)
            else:  # Else is an audio token
                if audio_token.uncased != SILENCE_TOKEN:
                    word_level_tokens.append(audio_token.cased)
                    word_level_tokens_idxs.append(i)

        text = join_list_of_tokens(word_level_tokens)
        # Tokenize with spacy
        doc = nlp(text)
        sents = [sent.text for sent in doc.sents]
        # Collapse beginning/end sentences that are just punctuation
        if is_punctuation(sents[0]):
            sents = [sents[0] + sents[1]] + sents[2:]
        if is_punctuation(sents[-1]):
            sents = sents[:-2] + [sents[-2] + sents[-1]]
        if is_punctuation(sents[-1]):  # In cases of 2 punctuation-only sentences
            sents = sents[:-2] + [sents[-2] + sents[-1]]
        sents = recursively_collapse_sentence_strings(sents)

        cur_word_level_token_idx = 0
        new_span_tokens_idxs: List[Tuple[int, int]] = []
        for sent in sents:
            start_span_tokens_idx = word_level_tokens_idxs[cur_word_level_token_idx]
            while (
                len(word_level_tokens) > cur_word_level_token_idx
                and word_level_tokens[cur_word_level_token_idx] in sent
            ):
                cur_token = word_level_tokens[cur_word_level_token_idx]
                sent = sent.replace(cur_token, "", 1)  # Remove token from sent
                cur_word_level_token_idx += 1
            end_span_tokens_idx = word_level_tokens_idxs[cur_word_level_token_idx - 1]
            new_span_tokens_idxs.append((start_span_tokens_idx, end_span_tokens_idx))

        assert new_span_tokens_idxs[0][0] in [0, 1]

        if len(new_span_tokens_idxs) > 1:
            # Divide into fewer spans
            split_spans = []

            if span.tokens[new_span_tokens_idxs[0][0]] == SILENCE_TOKEN:
                previous_half_silence_token = span.tokens[new_span_tokens_idxs[0][0]]
            else:
                previous_half_silence_token = None

            for i, (start_idx, end_idx) in enumerate(new_span_tokens_idxs):
                # Get tokens for the new span
                if previous_half_silence_token is not None:
                    tokens_for_this_new_span = [previous_half_silence_token]
                    tokens_for_this_new_span.extend(
                        span.tokens[start_idx : end_idx + 1]
                    )
                else:
                    tokens_for_this_new_span = span.tokens[start_idx : end_idx + 1]
                # Split & add if next token is silence
                if (
                    len(span.tokens) > end_idx + 1
                    and not isinstance(
                        span.tokens[end_idx + 1], str
                    )  # FIXME? handle when it is a str
                    and span.tokens[end_idx + 1].uncased == SILENCE_TOKEN
                ):
                    first_half, second_half = span.tokens[end_idx + 1].bisect()
                    tokens_for_this_new_span.append(first_half)
                    previous_half_silence_token = second_half
                else:
                    previous_half_silence_token = None

                new_span = Span(
                    tokens=tokens_for_this_new_span,
                    bible_metadata=deepcopy(span.bible_metadata),
                    utterance_num=i + 1,
                    known_speaker=span.known_speaker,
                )

                split_spans.append(new_span)
            new_spans.extend(split_spans)
        else:
            span.utterance_num = 1
            new_spans.append(span)
    utterances = [Utterance.from_span(span) for span in new_spans]
    return utterances


def load_ready_made_utterances() -> List[Utterance]:
    """Loads the utterances that don't need any alignment/tokenization"""
    ready_utterances_dirs = ["Pito_Salas_utterances"]
    utterances = []
    for dir in ready_utterances_dirs:
        slug = "_".join(dir.split("_")[:-1])
        with open(os.path.join(dir, f"{slug}_utterances.txt"), "r") as f:
            all_punctuated_utterance_text = f.read()
        punctuated_utterance_texts = all_punctuated_utterance_text.split("\n")
        unpunctuated_utterance_texts = [
            unpunctuate_papiamentu(text) for text in punctuated_utterance_texts
        ]
        for i, (punc, unpunc) in enumerate(
            zip(punctuated_utterance_texts, unpunctuated_utterance_texts)
        ):
            if punc:
                audio = load_mp3(os.path.join(dir, f"{slug}_{i+1:02d}.mp3"))
                # Monkey-patch half-baked Span object for getting speaker & utterance num
                span = Span(
                    known_speaker=slug.replace("_", " "),
                    utterance_num=i + 1,
                )
                utterrance = Utterance(
                    speaker_embedding=None,
                    span=span,
                    audio=audio,
                    punctuated=punc,
                    unpunctuated=unpunc,
                )
                utterances.append(utterrance)
    return utterances


def generate_figures(df: pd.DataFrame) -> None:
    """Generates figures for the final report."""
    speakers = df["speaker"].values
    embeds = np.stack(df["speaker_embedding"].values)
    ax = get_speaker_plot(embeds, speakers, title="Speaker Embeddings")
    ax.figure.savefig("speaker_embeddings.png", dpi=300)


class UtteranceMetadata(TypedDict):
    """An instance of utterance metadata."""

    utterance_id: int
    file_name: str
    train_dev_test_split: Optional[str]
    duration_ms: Optional[int]
    unpunctuated_text: str
    punctuated_text: str
    principle_speaker_partition: Optional[str]
    speaker_embedding: Optional[np.ndarray]
    speaker: Optional[str]
    speaker_was_inferred: Optional[bool]
    speaker_is_male: Optional[bool]
    speaker_comment: Optional[str]
    recording_environment_comment: Optional[str]
    audio_license: Optional[str]


cat_2_cur_num = 1


def get_utterance_id(utterance: Utterance) -> str:
    """The utterance_id is a sortable, 9-digit unique identifier for each utterance.

    For the bible.is New Testament audio, the number has 4 groups (E.g:
    {1}{01}{01}{01}{01}) where:

        - The 1st group/digit denotes the source process of the utterance.
            - 1 = bible.is New Testament
            - 2 = Miscellaneous internet audio
            - 3 = Recorded first-hand

        - The 2nd group (2 digits) is the book number, where 01=Matthew, 02=Mark, etc.

        - The 3rd group (2 digits) is the chapter number.

        - The 4th group (2 digits) is the verse number.

        - The 5th group (2 digits) denotes the utterance number within the verse.
          NOTE: if the verse is 00, then the utterance is an introductory utterance
          (chapter audios are preceded by short statements of the book/chapter).

    For miscellaneous internet audio and recorded first-hand audio, the number has 2
    (Egroups (E.g: {2}{00000001}) where:

        - The 1st group/digit denotes the source process of the utterance.
            - 1 = bible.is New Testament
            - 2 = Miscellaneous internet audio
            - 3 = Recorded first-hand

        - The 2nd group (8 digits) denotes the utterance number."""

    utterance_num = utterance.span.utterance_num
    if utterance.span.bible_metadata is not None:
        book_num = utterance.span.bible_metadata.book
        chapter_num = utterance.span.bible_metadata.chapter
        verse_num = utterance.span.bible_metadata.verse
        uid = f"1{book_num:02d}{chapter_num:02d}{verse_num:02d}{utterance_num:02d}"
    else:
        global cat_2_cur_num
        uid = f"2{cat_2_cur_num:08d}"
        cat_2_cur_num += 1
    return int(uid)


def build_corpus(utterances: List[Utterance]) -> None:
    """Builds the final corpus by creating the .csv file that is indexed by utterance and
    contains all pertinent metadata and saving the corresponding audio files to the
    corpus audio directory.
    """
    utterance_metadata_rows = []
    for utterance in utterances:
        speaker = utterance.span.known_speaker or utterance.span.principle_speaker
        utterance_id = get_utterance_id(utterance)
        file_name = f"{utterance_id}.wav"
        utterance_metadata_rows.append(
            UtteranceMetadata(
                utterance_id=utterance_id,
                file_name=file_name,
                train_dev_test_split=None,
                duration_ms=len(utterance.audio) / SAMPLE_RATE * 1000,
                unpunctuated_text=utterance.unpunctuated,
                punctuated_text=utterance.punctuated,
                principle_speaker_partition=utterance.span.principle_speaker,
                speaker_embedding=utterance.speaker_embedding,
                speaker=speaker,
                speaker_was_inferred=not bool(speaker),
                speaker_is_male=IS_MALE_BY_SPEAKER[speaker],
                speaker_comment=SPEAKER_COMMENTS[speaker],
                recording_environment_comment=RECORDING_ENV_COMMENTS[speaker],
                audio_license=LICENSE_BY_SPEAKER[speaker],
            )
        )
        file_path = os.path.join(FINAL_AUDIO_DIR, file_name)
        sf.write(file_path, utterance.audio, 16000)

    df = pd.DataFrame(utterance_metadata_rows)
    df.to_csv(FINAL_METADATA_PATH, index=False)
    return df


def get_spans() -> Tuple[List[Span], List[Span]]:
    """Loops through alignment files and instantiates Span objects"""
    bible_spans = []
    scraped_bible = ScrapedBible()
    for chapter_slug in scraped_bible.get_chapter_slugs():
        tprint("Getting span for", chapter_slug)
        alignment_path = get_alignment_file_path_from_chapter_slug(chapter_slug)
        mp3_path = get_mp3_file_path_from_chapter_slug(chapter_slug)
        spans = Span.from_file_paths(mp3_path, alignment_path, is_bible_chapter=True)
        bible_spans.extend(spans)

    other_spans = []
    for slug, speaker in SPEAKER_BY_OTHER_MP3S.items():
        tprint("Getting span for", slug)
        mp3_path = os.path.join(OTHER_MP3S_DIR, f"{slug}.mp3")
        alignment_path = os.path.join(ALIGNMENTS_DIR, f"{slug}.csv")
        transcription_path = os.path.join(OTHER_MP3S_TRANSCRIPTIONS_DIR, f"{slug}.txt")
        spans = Span.from_file_paths(
            mp3_path,
            alignment_path,
            transcription_path,
            is_bible_chapter=False,
            known_speaker=speaker,
        )
        other_spans.extend(spans)

    return bible_spans, other_spans


def sanity_check_bible_span(span: Span) -> None:
    tprint(
        f"#### Sanity checking Span({span.book_name}-{span.chapter}:{span.verse}-{span.utterance_num}) ####"
    )
    if span.verse in [18, 23]:
        for token in span.tokens:
            if isinstance(token, str):
                tprint(token)
            else:
                tprint(f"playing {token.cased}")
                token.play_audio()


def sanity_check_utterance(utterance: Utterance) -> None:
    tprint(f"#### Sanity checking utterance ####")
    if utterance.audio is not None:
        if utterance.span is not None:
            for token in utterance.span.tokens:
                if isinstance(token, str):
                    tprint(token)
                else:
                    tprint(f"playing {token.cased}")
                    token.play_audio()
        else:
            tprint(utterance.punctuated)
            play_audio(utterance.audio)


def main():
    tprint("Getting spans...")
    bible_verse_spans, other_audio_spans = get_spans()
    tprint("Tokenizing into utterances...")
    utterances = tokenize_into_utterances(bible_verse_spans + other_audio_spans)
    tprint("Incrementing ready-made utterances...")
    utterances.extend(load_ready_made_utterances())
    tprint("Embedding speakers...")
    for utterance in utterances:
        try:
            utterance.embed_speaker()
        except Exception as e:
            tprint(f"Failed to embed speaker for utterance {utterance}: {e}")
    tprint("Building corpus...")
    build_corpus(utterances)
    tprint("Done!")


if __name__ == "__main__":
    main()
