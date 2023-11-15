import re
import os
from typing import Optional, Union, List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import regex
import pyaudio
import unicodedata
import pandas as pd
from pydub import AudioSegment

from levenshtein import normalized_lev_sim
from number_conversion import (
    int_to_papiamentu,
    int_to_spanish,
)
from mappings import (
    BOOKS,
    PRINCIPLE_SPEAKERS_BY_BOOK,
    BOOK_NAMES,
)

BOOK_ANNOUNCENMENT_UTTERANCE = "{long}"
CHAPTER_ANNOUNCENMENT_UTTERANCE = "{short} {chapter_num}"

SAMPLE_RATE = 16000
N_CHANNELS = 1

SILENCE_TOKEN = "<p:>"
MP3_DIR = "mp3s"
BIBLE_DATA_CSV_PATH = "papiamentu_bible.csv"


LEFT_COLLAPSING = {".", ",", ";", ":", "!", "?", ")", "]", "}", "’", "”"}
RIGHT_COLLAPSING = {"(", "[", "{", "‘", "“"}
NON_DIRECTIONAL = {"'", '"'}


def is_punctuation(string: str):
    if len(string) > 4:
        return False
    checks = []
    for char in string:
        check = not (char.isalnum() or unicodedata.category(char).startswith("M"))
        checks.append(check)
    return all(checks)


def collapse_whitespace(text):
    """Collapses whitespace in a string according to basic heuristics."""
    # State to alternate the collapsing direction for non-directional quotes
    alternate_state = False  # Start with rightward collapsing
    # Iterate over the string and collapse spaces accordingly
    collapsed_text = ""
    i = 0
    while i < len(text):
        char = text[i]
        if char in NON_DIRECTIONAL:
            if alternate_state:  # Rightward collapsing
                if i > 0 and text[i - 1] == " ":
                    collapsed_text = collapsed_text[:-1]
            else:  # Leftward collapsing
                if i < len(text) - 1 and text[i + 1] == " ":
                    i += 1  # Skip the next space
            alternate_state = not alternate_state
        elif char in LEFT_COLLAPSING and i > 0 and collapsed_text[-1] == " ":
            collapsed_text = collapsed_text[:-1]
        elif char in RIGHT_COLLAPSING and i < len(text) - 1 and text[i + 1] == " ":
            i += 1  # Skip the next space
        collapsed_text += char
        i += 1
    return collapsed_text


def join_list_of_tokens(tokens: List[str]) -> str:
    """Joins a list of tokens into a string."""
    text = " ".join(tokens).strip()
    return collapse_whitespace(text)


def load_mp3(file_path: str) -> np.ndarray:
    """Loads an mp3 and converts it to a mono, 16 bit, 16kHz sample-rate wav in the
    format of an np.ndarray."""
    audio = AudioSegment.from_mp3(file_path)
    if audio.channels == 2:
        audio = audio.set_channels(N_CHANNELS)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_sample_width(2)
    audio_array = np.array(audio.get_array_of_samples())
    return audio_array


def get_introductory_utterances(
    book_name: str, long_book_name: str, chapter_num: int, unpunctuate: bool = True
) -> List[str]:
    """Returns the introductory utterance read at the beginning of a chapter audio."""
    if int(chapter_num) == 1:
        intros = [BOOK_ANNOUNCENMENT_UTTERANCE.format(long=long_book_name)]
    else:
        intros = []
    ch_announcement = CHAPTER_ANNOUNCENMENT_UTTERANCE.format(
        short=book_name, chapter_num=chapter_num
    )
    intros.append(ch_announcement)
    if unpunctuate:
        intros = [unpunctuate_papiamentu(i) for i in intros]
    return intros


def play_audio(audio: np.ndarray) -> None:
    """Plays audio."""
    if audio.size > 0:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(audio.itemsize),
            channels=N_CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
        )
        stream.write(audio.tobytes())
        stream.stop_stream()
        stream.close()


def find_next_change_index(
    df: pd.DataFrame, start_index: int, column: str
) -> Optional[int]:
    """Finds the next index where the value in the given column changes."""
    start_value = df.loc[start_index, column]
    mask = (df[column] != start_value) & (df.index > start_index)
    change_index = mask.idxmax()
    if not mask.any():
        return None
    return change_index


def unpunctuate_papiamentu(text: str) -> str:
    """Papiamentu-specific function reduce text to bare tokens separated by spaces."""
    # Add spaces to ends
    text = f" {text} "
    # Remove any punctuation with whitespace on either side
    cleaned_text = re.sub(r"(?<![a-zA-Z])[^\w\s']|[^\w\s'](?![a-zA-Z])", "", text)
    # Replace any whitespace with a single space
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    # Remove any leading or trailing whitespace
    cleaned_text = cleaned_text.strip()
    # Lowercase
    cleaned_text = cleaned_text.lower()
    return cleaned_text


@dataclass
class AudioToken:
    """A token with associated audio."""

    uncased: str
    cased: str
    audio: np.ndarray

    def bisect(self) -> Tuple["AudioToken", "AudioToken"]:
        """Splits the audio in half and returns two new AudioTokens."""
        assert self.uncased == SILENCE_TOKEN, "Can only bisect silence tokens."
        half = self.audio.size // 2
        return (
            AudioToken(self.uncased, self.cased, self.audio[:half]),
            AudioToken(self.uncased, self.cased, self.audio[half:]),
        )

    def play_audio(self) -> None:
        play_audio(self.audio)


@dataclass
class BibleSpanMetadata:
    """Metadata about a span of tokens/audio from the Bible."""

    book: Optional[int]
    chapter: Optional[int]
    verse: Optional[int] = None  # 0 if introductory utterance
    verse_start_token_idxs: Optional[Dict[int, int]] = None


class Span:
    """A span of tokens/audio from the data."""

    def __init__(
        self,
        tokens: Optional[List[Union[AudioToken, str]]] = None,
        bible_metadata: Optional[BibleSpanMetadata] = None,
        utterance_num: Optional[int] = None,
        known_speaker: Optional[str] = None,
    ):
        # self.tokens are AudioTokens interleaved with punctuation chars
        self.tokens = tokens
        self.bible_metadata = bible_metadata
        self.utterance_num = utterance_num
        self.known_speaker = known_speaker

    @classmethod
    def from_file_paths(
        cls,
        mp3_path: str,
        alignment_path: str,
        punctuated_text_file: Optional[str] = None,
        is_bible_chapter: bool = False,
        known_speaker: Optional[str] = None,
    ) -> List["Span"]:
        return _create_spans(
            mp3_path=mp3_path,
            alignment_path=alignment_path,
            is_bible_chapter=is_bible_chapter,
            punctuated_text_file=punctuated_text_file,
            known_speaker=known_speaker,
        )

    @property
    def book(self) -> Optional[int]:
        if self.bible_metadata is None:
            return None
        return self.bible_metadata.book

    @property
    def chapter(self) -> Optional[int]:
        if self.bible_metadata is None:
            return None
        return self.bible_metadata.chapter

    @property
    def verse(self) -> Optional[int]:
        if self.bible_metadata is None:
            return None
        return self.bible_metadata.verse

    @property
    def book_name(self) -> str:
        if self.bible_metadata is None:
            return None
        return BOOKS[self.bible_metadata.book]

    @property
    def principle_speaker(self) -> str:
        if self.bible_metadata is None:
            return None
        if self.bible_metadata.verse == 0:
            # This speaker always does the intro utterances
            return PRINCIPLE_SPEAKERS_BY_BOOK["Hebreonan"]
        # TODO: Jesus?
        return PRINCIPLE_SPEAKERS_BY_BOOK[self.book_name]

    def get_punctuated_tokens_as_strings(self) -> List[str]:
        tokens = []
        for t in self.tokens:
            if isinstance(t, str):
                tokens.append(t)
            elif t.uncased != SILENCE_TOKEN:
                tokens.append(t.cased)
        return tokens

    def get_unpunctuated_tokens_as_strings(self) -> List[str]:
        tokens = []
        for t in self.tokens:
            if isinstance(t, AudioToken) and t.uncased != SILENCE_TOKEN:
                tokens.append(t.uncased)
        return tokens

    def get_punctuated_text(self) -> str:
        standalone_punctuation_regex = r"^[^\w\s]|[^\w\s]$"
        to_join = []
        for token_str in self.get_punctuated_tokens_as_strings():
            if re.match(standalone_punctuation_regex, token_str):
                to_join.append(token_str)
            else:
                to_join.append(f" {token_str}")
        return join_list_of_tokens(to_join)

    def get_audio(self) -> np.ndarray:
        audios = [t.audio for t in self.tokens if isinstance(t, AudioToken)]
        if len(audios) == 0:
            raise ValueError("No audio in this span.")
        return np.concatenate(audios)

    def play_audio(self) -> None:
        play_audio(self.get_audio())


############################################################################################
# Ad-hoc, verboose function for creating span objects given our raw data artifacts (files) #
############################################################################################


def separate_leading_punctuation(s):
    match = regex.match(r"[^\p{Alphabetic}\p{N}]+", s)
    if match:
        return (match.group(), s[match.end() :])
    else:
        return (None, s)


def separate_trailing_punctuation(s):
    match = regex.search(r"[^\p{Alphabetic}\p{N}]+$", s)
    if match:
        return (s[: match.start()], match.group())
    else:
        return (s, None)


def _create_spans(
    mp3_path: str,
    alignment_path: str,
    punctuated_text_file: Optional[str] = None,
    is_bible_chapter: bool = False,
    known_speaker: Optional[str] = None,
) -> List[Union[AudioToken, str]]:
    """Creates a list of Span objects from given file paths.

    Notes:
        - This algorithm is thoroughly sanity-checked. I.E. We arer confident that it
          creates Spans correctly for all New Testament verse and other non-bible files
          that we are feeding it.
        - That being said, the "coding practices" are especially messy, verbose, and hard
          to follow. This function is HIGHLY coupled to our data artifacts (files) and
          highly subject to breaking given minor upstream tweaks.
        - This was a compromise we made for a project deadline. If you are reading
          this and are confused, please reach out to me (Sonny) and we can discuss
          the logic/steps or discuss a refactoring this to be atomized / readable...

          (In general this repo, although thoroughly sanity-checked, does not contain
          utterly "production-worthy" code)
    """

    # Load audio
    audio = load_mp3(mp3_path)

    # Default na values are a no-go since "nan" is a Papiamentu word
    alignment_df = pd.read_csv(
        alignment_path, sep=";", keep_default_na=False, na_values=[""]
    )
    alignment_df = alignment_df.reset_index(drop=True)

    # Get metadata and punctuated text
    if is_bible_chapter:
        f_name = os.path.basename(alignment_path).split(".")[0]
        book_name, chapter = f_name.split("_")
        bible_metadata = BibleSpanMetadata(
            book=BOOK_NAMES[book_name], chapter=int(chapter)
        )

        ch_df = pd.read_csv(BIBLE_DATA_CSV_PATH)
        ch_df = ch_df[
            (ch_df["book_name_short"] == book_name) & (ch_df["chapter"] == int(chapter))
        ]

        verse_texts = list(ch_df["verse_text"])
        verse_nums = list(ch_df["verse"])
        intros = get_introductory_utterances(
            book_name, ch_df["book_name"].iloc[0], chapter, unpunctuate=False
        )
        if len(intros) == 2:
            intro = ", ".join(intros)
        elif len(intros) == 1:
            intro = intros[0]
        excerpts = [intro] + verse_texts
        verse_nums = [0] + verse_nums
    elif punctuated_text_file is not None:
        bible_metadata = None
        with open(punctuated_text_file, "r") as f:
            punctuated_text = f.read()
        excerpts = [punctuated_text]
        verse_nums = None
    else:
        raise ValueError(
            "Must provide either a punctuated_text_file or is_bible_chapter=True."
        )

    unpunctuated_words_by_excerpt = []
    punctuated_words_by_excerpt = []
    for excerpt in excerpts:
        punctuated_words = excerpt.split()
        unpunctuated_words = unpunctuate_papiamentu(excerpt).split()
        unpunctuated_words_by_excerpt.append(unpunctuated_words)
        punctuated_words_by_excerpt.append(punctuated_words)

    punctuated_words = sum(punctuated_words_by_excerpt, [])

    unpunctuated_words: List[str] = []
    verse_start_idxs = []

    for words in unpunctuated_words_by_excerpt:
        verse_start_idxs.append(len(unpunctuated_words))
        unpunctuated_words.extend(words)

    spans: List[Span] = []
    cur_align_idx = 0
    cur_punc_idx = 0
    tokens = []
    for i, unpunc in enumerate(unpunctuated_words):
        if verse_nums is not None and i in verse_start_idxs and i != 0:
            is_pending_new_verse_span = True
        else:
            is_pending_new_verse_span = False

        # Handle standalone punctuation
        punc = punctuated_words[cur_punc_idx]
        if is_punctuation(punc):
            tokens.append(punc)
            cur_punc_idx += 1
            punc = punctuated_words[cur_punc_idx]

        assert unpunctuate_papiamentu(punc) == unpunc

        # Handle silence tokens
        cur_alignment_tok = alignment_df["ORT"][cur_align_idx]
        is_silence = pd.isna(cur_alignment_tok)
        if is_silence:
            start = alignment_df["BEGIN"][cur_align_idx]
            duration = alignment_df["DURATION"][cur_align_idx]
            token = AudioToken(SILENCE_TOKEN, None, audio[start : start + duration])
            if is_pending_new_verse_span:
                token, second_half = token.bisect()
            tokens.append(token)
            cur_align_idx += 1
            cur_alignment_tok = alignment_df["ORT"][cur_align_idx]

        if is_pending_new_verse_span:
            verse_bible_metadata = BibleSpanMetadata(
                chapter=bible_metadata.chapter,
                book=bible_metadata.book,
                verse=verse_nums[verse_start_idxs.index(i) - 1],
            )  # Add span for the just-processed verse
            if all(isinstance(t, str) for t in tokens):
                # This is just punctuation (no audio), so we add the tokens to last span
                spans[-1].tokens.extend(tokens)
            else:
                spans.append(Span(tokens=tokens, bible_metadata=verse_bible_metadata))
            tokens = []
            if is_silence:
                tokens.append(second_half)  # Add second half of prev silence

        assert not pd.isna(cur_alignment_tok)  # No longer on a silence token

        # Handle numbers that were converted to spanish by MAUS alignment
        if unpunc.isdigit():
            digit = int(unpunc)
            papi_n = int_to_papiamentu(digit)
            spa_n = int_to_spanish(digit)
            close_to_papi = normalized_lev_sim(papi_n, cur_alignment_tok) > 0.55
            close_to_spa = normalized_lev_sim(spa_n, cur_alignment_tok) > 0.55
            assert close_to_papi or close_to_spa
            unpunc = int_to_papiamentu(int(unpunc))
        else:
            assert (
                cur_alignment_tok == regex.sub(r"[^\p{L}\p{M}0-9]", "", unpunc)
                or cur_alignment_tok == unpunc
                or normalized_lev_sim(punc, cur_alignment_tok) > 0.4
            )

        leading_punctuation, punc = separate_leading_punctuation(punc)
        punc, trailing_punctuation = separate_trailing_punctuation(punc)

        # Add leading punctuation to tokens
        if leading_punctuation is not None:
            for char in leading_punctuation:
                tokens.append(char)

        # Get audio snippet for token
        tok_rows = alignment_df[
            cur_align_idx : find_next_change_index(alignment_df, cur_align_idx, "TOKEN")
        ]
        tok_start = tok_rows["BEGIN"].iloc[0]
        tok_end = tok_rows["BEGIN"].iloc[-1] + tok_rows["DURATION"].iloc[-1]
        snippet = audio[tok_start:tok_end]

        # Add token
        tok = AudioToken(unpunc, punc, snippet)
        tokens.append(tok)

        # Add trailing punctuation to tokens
        if trailing_punctuation is not None:
            for char in trailing_punctuation:
                tokens.append(char)

        cur_align_idx += len(tok_rows)
        cur_punc_idx += 1

    if verse_nums is not None and bible_metadata is not None:
        verse_bible_metadata = BibleSpanMetadata(
            chapter=bible_metadata.chapter,
            book=bible_metadata.book,
            verse=verse_nums[-1],
        )
        spans.append(Span(tokens=tokens, bible_metadata=verse_bible_metadata))
    else:
        spans.append(
            Span(
                tokens=tokens,
                bible_metadata=bible_metadata,
                known_speaker=known_speaker,
            )
        )

    return spans
