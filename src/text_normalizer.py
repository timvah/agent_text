"""Deterministic Text Normalizer for TTS and Narrative Script Processing.

Handles:
- Numeral expansion (cardinals, ordinals, decimals, years, centuries).
- Currency normalization ($4.5M -> four point five million dollars).
- Percentage expansion (30% -> thirty percent).
- Common historical and grammatical abbreviations (BC/BCE, ca., Dr., etc.).
- Markdown & punctuation stripping with cadence preserving commas/ellipses.
"""

import re
from typing import List
from num2words import num2words


# Historical titles and regnal Roman numerals
REGNAL_NUMERALS = {
    "I": "the First",
    "II": "the Second",
    "III": "the Third",
    "IV": "the Fourth",
    "V": "the Fifth",
    "VI": "the Sixth",
    "VII": "the Seventh",
    "VIII": "the Eighth",
    "IX": "the Ninth",
    "X": "the Tenth",
    "XI": "the Eleventh",
    "XII": "the Twelfth",
    "XIII": "the Thirteenth",
    "XIV": "the Fourteenth",
    "XV": "the Fifteenth",
    "XVI": "the Sixteenth",
}

COMMON_ABBREVIATIONS = {
    r"\bca\.(?=\s|$)": "circa",
    r"\bc\.(?=\s|$)": "circa",
    r"\be\.g\.,?(?=\s|$)": "for example",
    r"\bi\.e\.,?(?=\s|$)": "that is",
    r"\betc\.(?=\s|$)": "et cetera",
    r"\bvs\.(?=\s|$)": "versus",
    r"\bDr\.(?=\s|$)": "Doctor",
    r"\bMr\.(?=\s|$)": "Mister",
    r"\bMrs\.(?=\s|$)": "Missus",
    r"\bProf\.(?=\s|$)": "Professor",
    r"\bft\.(?=\s|$)": "feet",
    r"\bkm\b": "kilometers",
    r"\bmph\b": "miles per hour",
}


def normalize_regnal_names(text: str) -> str:
    """Expands monarch/pope names like 'Ptolemy I' to 'Ptolemy the First'."""
    pattern = r"\b([A-Z][a-z]+)\s+(X{0,3}(?:IX|IV|V?I{1,3}))\b"

    def replace_numeral(match):
        name = match.group(1)
        numeral = match.group(2)
        if numeral in REGNAL_NUMERALS:
            return f"{name} {REGNAL_NUMERALS[numeral]}"
        return match.group(0)

    return re.sub(pattern, replace_numeral, text)


def normalize_currency(text: str) -> str:
    """Normalizes currency symbols to spoken words."""
    # Matches $1.5M, $1.5 million, $1.5B, $500, $45.50
    # Millions / Billions shorthand: $4.5M, $10B
    def replace_multiplier(match):
        sym, num, unit = match.groups()
        currency_name = "dollars" if sym == "$" else ("euros" if sym == "€" else "pounds")
        unit_word = "million" if unit.upper() == "M" else "billion"
        num_word = num2words(float(num) if "." in num else int(num))
        return f"{num_word} {unit_word} {currency_name}"

    text = re.sub(r"([\$€£])(\d+(?:\.\d+)?)\s*([mMbB])\b", replace_multiplier, text)

    # Standard currency: $500, $45.50
    def replace_standard(match):
        sym, num = match.groups()
        currency_name = "dollars" if sym == "$" else ("euros" if sym == "€" else "pounds")
        if "." in num:
            parts = num.split(".")
            dollars = int(parts[0])
            cents = int(parts[1]) if len(parts[1]) == 2 else int(parts[1] + "0")
            dollars_word = num2words(dollars)
            cents_word = num2words(cents)
            if dollars > 0 and cents > 0:
                return f"{dollars_word} {currency_name} and {cents_word} cents"
            elif dollars > 0:
                return f"{dollars_word} {currency_name}"
            else:
                return f"{cents_word} cents"
        else:
            return f"{num2words(int(num))} {currency_name}"

    text = re.sub(r"([\$€£])(\d+(?:\.\d{1,2})?)\b", replace_standard, text)
    return text


def normalize_percentages(text: str) -> str:
    """Normalizes percentages: '30%' -> 'thirty percent'."""
    def replace_percent(match):
        num_str = match.group(1)
        if "." in num_str:
            word = num2words(float(num_str))
        else:
            word = num2words(int(num_str))
        return f"{word} percent"

    return re.sub(r"(\d+(?:\.\d+)?)\s*%", replace_percent, text)


def normalize_years_and_eras(text: str) -> str:
    """Expands historical years, eras and century references."""
    # Centuries: 3rd century, 21st century
    def replace_century(match):
        num = int(match.group(1))
        return f"{num2words(num, to='ordinal')} century"

    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\s+century\b", replace_century, text, flags=re.IGNORECASE)

    # Eras: 300 BC -> three hundred BC, 1200 BCE -> twelve hundred BCE
    def replace_era(match):
        num = int(match.group(1))
        era = match.group(2)
        return f"{num2words(num, to='year')} {era}"

    text = re.sub(r"\b(\d{2,4})\s*(BC|AD|BCE|CE)\b", replace_era, text)

    # Decades: 1980s, 1800s
    def replace_decade(match):
        base_year = int(match.group(1))
        year_word = num2words(base_year, to="year")
        return f"{year_word}s"

    text = re.sub(r"\b(1\d{3}|20\d{2})s\b", replace_decade, text)

    # Standalone 4-digit years (between 1000 and 2099)
    def replace_year(match):
        year = int(match.group(0))
        return num2words(year, to="year")

    # Match 4 digit year when surrounded by word boundaries or punctuation
    text = re.sub(r"\b(1[0-9]{3}|20[0-9]{2})\b", replace_year, text)
    return text


def normalize_ordinals_and_cardinals(text: str) -> str:
    """Expands standard ordinals (1st, 2nd, 3rd) and remaining digits."""
    # Ordinals: 1st, 2nd, 3rd, 4th...
    def replace_ordinal(match):
        num = int(match.group(1))
        return num2words(num, to="ordinal")

    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", replace_ordinal, text, flags=re.IGNORECASE)

    # Decimals: 3.14 -> three point one four
    def replace_decimal(match):
        full_num = match.group(0)
        integer_part, frac_part = full_num.split(".")
        int_word = num2words(int(integer_part))
        frac_words = " ".join([num2words(int(d)) for d in frac_part])
        return f"{int_word} point {frac_words}"

    text = re.sub(r"\b\d+\.\d+\b", replace_decimal, text)

    # Remaining integers (including comma-separated like 10,000)
    def replace_comma_int(match):
        clean_num = match.group(0).replace(",", "")
        return num2words(int(clean_num))

    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", replace_comma_int, text)

    def replace_standalone_int(match):
        return num2words(int(match.group(0)))

    text = re.sub(r"\b\d+\b", replace_standalone_int, text)
    return text


def clean_markdown_and_punctuation(text: str) -> str:
    """Strips markdown and standardizes cadence pauses (ellipses/commas)."""
    # Remove markdown bold/italic/headers/links
    text = re.sub(r"\#{1,6}\s*", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"`[^`]*`", "", text)

    # Replace em-dashes and en-dashes with a comma pause
    text = re.sub(r"\s*[—–]\s*", ", ", text)

    # Standardize quotation marks
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Expand abbreviations
    for pattern, replacement in COMMON_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Clean double punctuation or trailing awkward characters
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_tts(text: str) -> str:
    """Full text normalization pipeline for TTS input."""
    if not text:
        return ""
    text = clean_markdown_and_punctuation(text)
    text = normalize_regnal_names(text)
    text = normalize_currency(text)
    text = normalize_percentages(text)
    text = normalize_years_and_eras(text)
    text = normalize_ordinals_and_cardinals(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """Splits narrative prose into well-formed sentences for video beats."""
    text = text.strip()
    if not text:
        return []

    # Match sentence endings: . ! ? followed by space and uppercase or end of string
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    results = []
    for s in raw_sentences:
        s_clean = s.strip()
        if s_clean:
            results.append(s_clean)
    return results
