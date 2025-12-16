import re
import pandas as pd
from num2words import num2words
from underthesea import text_normalize


emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE
)

def remove_emoji(text):
    if not isinstance(text, str):
        return text
    return emoji_pattern.sub("", text)

def remove_special_chars(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"[^a-zA-Z0-9À-ỹ .,!?'-]", " ", text)
def is_year(num_str):
    try:
        n = int(num_str)
        return 1000 <= n <= 2100
    except:
        return False
def normalize_numbers_en(text):
    if not isinstance(text, str):
        return text

    def convert(match):
        num = match.group()
        if is_year(num):
            return num

        return num2words(num, lang="en")

    return re.sub(r"\d+", convert, text)


def normalize_numbers_vi(text):
    if not isinstance(text, str):
        return text

    def convert(match):
        num = match.group()
        if is_year(num):
            return num

        num = int(num)
        return text_normalize(num2words(num, lang="vi"))

    return re.sub(r"\d+", convert, text)

def clean_en(text):
    if not isinstance(text, str):
        return text
    text = remove_emoji(text)
    text = remove_special_chars(text)
    text = normalize_numbers_en(text)   
    text = text.lower()
    return text.strip()


def clean_vi(text):
    if not isinstance(text, str):
        return text
    text = remove_emoji(text)
    text = remove_special_chars(text)
    text = normalize_numbers_vi(text)   
    text = text.lower()
    return text.strip()