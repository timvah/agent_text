import pytest
from src.text_normalizer import (
    normalize_for_tts,
    normalize_regnal_names,
    normalize_currency,
    normalize_percentages,
    normalize_years_and_eras,
    normalize_ordinals_and_cardinals,
    clean_markdown_and_punctuation,
    split_into_sentences,
)


def test_regnal_names():
    assert "Ptolemy the First ruled Egypt" in normalize_regnal_names("Ptolemy I ruled Egypt")
    assert "Henry the Eighth" in normalize_regnal_names("King Henry VIII")


def test_currency_normalization():
    res_m = normalize_currency("The project raised $1.5M for the expedition.")
    assert "one point five million dollars" in res_m

    res_std = normalize_currency("A scroll cost $500 in gold.")
    assert "five hundred dollars" in res_std

    res_cents = normalize_currency("The tax was $45.50.")
    assert "forty-five dollars and fifty cents" in res_cents


def test_percentages():
    res = normalize_percentages("Over 30% of ships were inspected, while 4.5% were seized.")
    assert "thirty percent" in res
    assert "four point five percent" in res


def test_years_and_eras():
    res_era = normalize_years_and_eras("In 300 BC, the empire flourished.")
    assert "three hundred BC" in res_era

    res_century = normalize_years_and_eras("During the 3rd century, scrolls multiplied.")
    assert "third century" in res_century

    res_year = normalize_years_and_eras("The treaty was signed in 1945.")
    assert "nineteen forty-five" in res_year

    res_decade = normalize_years_and_eras("Throughout the 1800s, excavation began.")
    assert "eighteen hundreds" in res_decade


def test_ordinals_and_cardinals():
    res_ord = normalize_ordinals_and_cardinals("He was the 1st scholar and 2nd director.")
    assert "first" in res_ord
    assert "second" in res_ord

    res_dec = normalize_ordinals_and_cardinals("The value of pi is approximately 3.14.")
    assert "three point one four" in res_dec


def test_markdown_and_abbreviations():
    text = "The **Great Library** — founded ca. 300 BC — contained scrolls vs. codices."
    res = clean_markdown_and_punctuation(text)
    assert "**" not in res
    assert "—" not in res
    assert "circa" in res
    assert "versus" in res


def test_full_tts_pipeline():
    raw = (
        "In 300 BC, Ptolemy I invested $2.5M to acquire 10,000 scrolls. "
        "Over 40% of vessels in the 3rd century were searched — confiscating every work!"
    )
    normalized = normalize_for_tts(raw)
    assert "three hundred BC" in normalized
    assert "Ptolemy the First" in normalized
    assert "two point five million dollars" in normalized
    assert "ten thousand" in normalized
    assert "forty percent" in normalized
    assert "third century" in normalized
    assert "—" not in normalized


def test_split_into_sentences():
    paragraph = (
        "In the third century before Christ, Alexandria was unrivaled. "
        "Every arriving ship was inspected by guards! Did any escape? "
        "Not a single scroll was spared."
    )
    sentences = split_into_sentences(paragraph)
    assert len(sentences) == 4
    assert sentences[0] == "In the third century before Christ, Alexandria was unrivaled."
    assert sentences[1] == "Every arriving ship was inspected by guards!"
    assert sentences[2] == "Did any escape?"
    assert sentences[3] == "Not a single scroll was spared."
