import os
from typing import List

import pandas as pd

from schema import unpunctuate_papiamentu, MP3_DIR, get_introductory_utterances
from mappings import BOOK_SPECIFIC_FILE_NAME_PATTERNS

ALIGNMENTS_DIR = "alignments"
SCRAPED_BIBLE_CSV_PATH = "papiamentu_bible.csv"

OTHER_MP3S_DIR = "other_mp3s"
OTHER_MP3S_TRANSCRIPTIONS_DIR = "other_mp3s_transcriptions"


class ScrapedBible:
    def __init__(self):
        self.df = pd.read_csv(SCRAPED_BIBLE_CSV_PATH)

    def get_chapter_slugs(self) -> List[str]:
        for short_book_name, group in self.df.groupby("book_name_short", sort=False):
            for chapter_num in group["chapter"].unique():
                yield f"{short_book_name}_{chapter_num}"

    def get_chapter_text(self, slug: str) -> str:
        # Get clean verses
        book, chap = slug.split("_")
        qry = (self.df["book_name_short"] == book) & (self.df["chapter"] == int(chap))
        book_df = self.df[qry]
        unformatted = book_df["verse_text"]
        verses = unformatted.apply(lambda x: unpunctuate_papiamentu(x))
        # Get clean chapter introduction utterances
        long_book_name = book_df["book_name"].iloc[0]
        intros = get_introductory_utterances(book, long_book_name, chap)
        # Return as single string where verse is separated by newlines
        return "\n".join(intros) + "\n" + "\n".join(verses) + "\n"


def get_mp3_file_path_from_chapter_slug(slug: str) -> str:
    """Returns the path to the mp3 file for the given chapter slug."""
    book, chapter_num = slug.split("_")
    file_name = BOOK_SPECIFIC_FILE_NAME_PATTERNS[book].format(ch=int(chapter_num))
    return os.path.join(MP3_DIR, file_name)


def get_alignment_file_path_from_chapter_slug(slug: str) -> str:
    """Returns the path to the alignment file for the given chapter slug."""
    return os.path.join(ALIGNMENTS_DIR, f"{slug}.csv")
