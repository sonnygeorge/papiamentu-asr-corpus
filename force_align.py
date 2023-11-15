"""
This script is responsible for getting the force alignments for each chapter audio of the
Papiamentu New Testament. It uses the MAUS server to perform the alignment, which is
called via a REST API. The alignment files are saved for later use.
"""
import os
import requests
from lxml import etree

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Memory

from utils import (
    get_mp3_file_path_from_chapter_slug,
    get_alignment_file_path_from_chapter_slug,
    ScrapedBible,
    OTHER_MP3S_DIR,
    OTHER_MP3S_TRANSCRIPTIONS_DIR,
    ALIGNMENTS_DIR,
)
from schema import load_mp3
from mappings import SPEAKER_BY_OTHER_MP3S


memory = Memory("cachedir", verbose=0)

ALIGNMENT_LANGUAGE = "spa"
SKIP_EXISTING_FILES = True


def run_maus_alignment(
    slug: str,
    text_file_path: str,
    audio_file_path: np.ndarray,
    language: str = ALIGNMENT_LANGUAGE,
) -> str:
    """Sends necessary data to MAUS server for alignment, saves the alignment file, and
    returns its path.
    """
    url = "https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSBasic"
    data = {r"LANGUAGE": language, r"OUTFORMAT": r"csv"}
    files = {r"TEXT": open(text_file_path, "r"), r"SIGNAL": open(audio_file_path, "rb")}
    print("Sending request to MAUS server...")
    r = requests.post(url, files=files, data=data)
    r.raise_for_status()
    response_xml = etree.fromstring(r.text)
    success = response_xml.find("success").text
    warnings = response_xml.find("warnings").text
    print(f"Outcome: {success} - Warnings: {warnings}")
    if success != "false":
        print("Attempting download of result...")
        download_url = response_xml.find("downloadLink").text
        request_download = requests.get(download_url, stream=True)
        request_download.raise_for_status()
        download_path = get_alignment_file_path_from_chapter_slug(slug)
        with open(download_path, "wb") as f:
            f.write(request_download.content)
        return download_path
    else:
        output = response_xml.find("output").text
        raise ValueError(f"MAUS alignment failed on server: {output}")


def run_force_alignment_for_chapter(
    slug: str, scraped_bible: ScrapedBible
) -> pd.DataFrame:
    """Assembles the corpus for a single chapter."""
    # Load chapter audio
    mp3_file_path = get_mp3_file_path_from_chapter_slug(slug)
    chapter_audio = load_mp3(mp3_file_path)
    # Get raw chapter text
    unpunctuated_text = scraped_bible.get_chapter_text(slug)
    # Temporarily save to files for MAUS alignment
    temp_txt_path = f"{slug}_temp.txt".replace(" ", "_")
    with open(temp_txt_path, "w") as f:
        f.write(unpunctuated_text)
    temp_wav_path = f"{slug}_temp.wav".replace(" ", "_")
    sf.write(temp_wav_path, chapter_audio, 16000)
    # Run MAUS alignment
    run_maus_alignment(slug, temp_txt_path, temp_wav_path)
    # Remove temp files
    os.remove(temp_txt_path)
    os.remove(temp_wav_path)


def main():
    # Bible chapters
    scraped_bible = ScrapedBible()
    for chapter_slug in scraped_bible.get_chapter_slugs():
        alignement_file_path = get_alignment_file_path_from_chapter_slug(chapter_slug)
        if SKIP_EXISTING_FILES and os.path.exists(alignement_file_path):
            print(f"Skipping {chapter_slug} (existing alignment file)")
            continue
        try:
            print("Running force alignment for chapter:", chapter_slug)
            run_force_alignment_for_chapter(chapter_slug, scraped_bible)
        except Exception as e:
            print(f"Failed to force align {chapter_slug}: {e}")

    # Other audios
    for slug in SPEAKER_BY_OTHER_MP3S.keys():
        transcription_f_name = f"{slug}.txt"
        mp3_file_name = f"{slug}.mp3"
        alignment_file_name = f"{slug}.csv"
        text_f_path = os.path.join(OTHER_MP3S_TRANSCRIPTIONS_DIR, transcription_f_name)
        mp3_file_path = os.path.join(OTHER_MP3S_DIR, mp3_file_name)
        alignment_file_path = os.path.join(ALIGNMENTS_DIR, alignment_file_name)
        if SKIP_EXISTING_FILES and os.path.exists(alignment_file_path):
            print(f"Skipping {slug} (existing transcription file)")
            continue
        try:
            audio = load_mp3(mp3_file_path)
            temp_wav_path = f"{slug}_temp.wav".replace(" ", "_")
            sf.write(temp_wav_path, audio, 16000)
            print("Running force alignment for other audio:", slug)
            run_maus_alignment(slug, text_f_path, temp_wav_path)
            os.remove(temp_wav_path)
        except Exception as e:
            print(f"Failed to force align {slug}: {e}")


if __name__ == "__main__":
    main()
