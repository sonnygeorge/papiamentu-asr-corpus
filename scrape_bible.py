"""
This script retrieves the verses from the "Beibel na Papiamentu Koriente  2013"
version of the Bible on bible.is.
"""
from typing import Dict
import json
import re
import csv
import random
import time

import requests
from bs4 import BeautifulSoup
from joblib import Memory

memory = Memory("cachedir", verbose=0)

BASE_URL = "https://live.bible.is/bible"
LANGUAGE_SLUG = "PAPPAP"  # Papiamentu
TESTAMENTS_TO_RETRIEVE = ["NT"]  # Papiamentu only has audio for New Testament

N_RETRIES = 5
NON_HYDRATED_HTML_SLEEP_TIMES = [0.2, 0.5, 1.1, 0.1, 0.7, 3.2]  # Arbitrary
SERVER_ERROR_SLEEP_TIMES = [5, 2, 13, 1, 3, 4]  # Arbitrary
FINAL_SLEEP_TIME = 15  # Arbitrary


with open("bible.is_metadata.json", "r") as f:
    BIBLE_METADATA = json.load(f)


def request_with_retry_on_server_errors(url: str) -> requests.Response:
    """Retrieves a url, retrying on server errors"""
    for i in range(N_RETRIES):
        r = requests.get(url)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Retry on server errors
            if r.status_code >= 500:
                print(f"Server error for {url} - {e}")
                if i == N_RETRIES - 1:
                    print(f"Final attempt for {url}")
                    time.sleep(FINAL_SLEEP_TIME)
                else:
                    time.sleep(random.choice(SERVER_ERROR_SLEEP_TIMES))
                continue
            else:
                raise e
        else:
            return r
    raise ValueError(f"Failed to retrieve {url} after {N_RETRIES} attempts")


@memory.cache
def get_page_html(book_id: str, chapter: int) -> str:
    """Retrieves page html html for a given bible chapter"""
    url = f"{BASE_URL}/{LANGUAGE_SLUG}/{book_id}/{chapter}"
    r = request_with_retry_on_server_errors(url)
    if len(r.content) < 10_000:
        for i in range(N_RETRIES):
            print(
                f"Suspiciously short response for {book_id}:{chapter} of length "
                f"{len(r.content)}"
            )
            if i == N_RETRIES - 1:
                print(f"Final attempt for {book_id}:{chapter}")
                time.sleep(FINAL_SLEEP_TIME)
            else:
                time.sleep(random.choice(NON_HYDRATED_HTML_SLEEP_TIMES))
            r = request_with_retry_on_server_errors(url)
            if len(r.content) > 10_000:
                break
        if len(r.content) < 10_000:
            raise ValueError("Failed to retrieve hydrated html after multiple attempts")
    return r.content


def is_verse_or_chap_num_element(tag: BeautifulSoup) -> bool:
    """Assuming the element is a child of the center-justified text div...
    The heuristic for knowing that an element contains only a verse number is:
        1. The element is a <span> or <sup>
        2. The element has no children (is a terminal node of the xml tree)
        3. The element's text, with HTML entities decoded and whitespace stripped,
           is numeric (or a range of numbers, e.g. "1-3"), or an integer followed by a single lower-case letter (e.g. "6a")
    """
    is_span = tag.name in ["span", "sup"]
    is_terminal = not tag.find()
    if is_span and is_terminal:
        # Get the text from the tag, with HTML entities like &nbsp; decoded
        text = tag.get_text(strip=True)
        # Remove any non-breaking spaces and strip whitespace again
        text = text.replace("\xa0", " ").strip()
        # Check if the text is numeric, a range of numbers, or a number followed by a lower-case letter
        split_text = text.split("-")
        is_numeric = all([re.fullmatch(r"\d+[a-z]?", s) for s in split_text])
        return is_numeric
    return False


def is_verse_text_element(tag: BeautifulSoup) -> bool:
    """Assuming the element is a child of the center-justified text div...
    The heuristic for knowing that an element contains an overtly uttered verse is:
        1. The element is a <span> or <sup>
        2. The element has no children (is a terminal node of the xml tree)
        3. The element is not a verse number element
    """
    is_span = tag.name == "span"
    is_terminal = not tag.find()
    return is_span and is_terminal and not is_verse_or_chap_num_element(tag)


def get_verse_number_from_verse_num_element(tag: BeautifulSoup) -> int:
    """Given a verse number element, returns the verse number as an int"""
    text = tag.get_text(strip=True)
    split_text = text.split("-")
    number = split_text[0]
    # Check if number contains a lower-case letter
    if re.fullmatch(r"\d+[a-z]", number):
        # Remove the letter
        number = number[:-1]
    return int(number)


def get_main_text_block_from_page_html(html: str) -> BeautifulSoup:
    """Given the html for a bible chapter page, returns the main text block"""
    soup = BeautifulSoup(html, "html.parser")
    return soup.find("div", class_="justify")


def is_valid_ordered_list_of_verses(int_list: list) -> bool:
    # Check if there's at least one item and the first item is 1
    if not int_list or int_list[0] != 1:
        return False
    # Check if the list is increasing
    return all([int_list[i] < int_list[i + 1] for i in range(len(int_list) - 1)])


def get_verses_from_page_html(html: str) -> Dict[int, str]:
    main_text_block = get_main_text_block_from_page_html(html)

    verse_elements = main_text_block.find_all(is_verse_text_element)
    verse_or_chap_num_elements = main_text_block.find_all(is_verse_or_chap_num_element)

    verse_num_elements = verse_or_chap_num_elements[1:]  # 1st should be the chapter num
    verse_nums = [
        get_verse_number_from_verse_num_element(e) for e in verse_num_elements
    ]

    # Some sanity checks
    if not len(verse_elements) == len(verse_num_elements):
        raise ValueError(
            "Number of verse elements and verse number elements do not match"
        )
    if not is_valid_ordered_list_of_verses(verse_nums):
        raise ValueError("Verses not a valid ordered list of integers")

    verses = [e.get_text(strip=True) for e in verse_elements]
    return dict(zip(verse_nums, verses))


def main():
    rows = []  # To be saved as a csv

    for book in BIBLE_METADATA["data"]["books"]:
        book_id = book["book_id"]
        book_name = book["name"]
        book_name_short = book["name_short"]
        chapters = book["chapters"]
        testament = book["testament"]

        # Skip book that are not in TESTAMENTS_TO_RETRIEVE
        if testament not in TESTAMENTS_TO_RETRIEVE:
            continue

        for chapter in chapters:
            # Attempt to get html for chapter
            try:
                html = get_page_html(book_id, chapter)
                print(f"Retrieved html for {book_id}:{chapter}")
            except Exception as e:
                print(f"Failed to retrieve html for {book_id}:{chapter} - {e}")
                continue

            # Attempt to get verses from html
            try:
                verses = get_verses_from_page_html(html)
                print(f"Parsed verses for {book_id}:{chapter}")
            except Exception as e:
                print(f"Failed to parse verses for {book_id}:{chapter} - {e}")
                verses = get_verses_from_page_html(html)
                continue

            for verse_num, verse_text in verses.items():
                verse_row = {
                    "book_id": book_id,
                    "book_name": book_name,
                    "book_name_short": book_name_short,
                    "testament": testament,
                    "chapter": chapter,
                    "verse": verse_num,
                    "verse_text": verse_text,
                }
                rows.append(verse_row)

    # Save rows as csv
    with open("papiamentu_bible.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


main()
