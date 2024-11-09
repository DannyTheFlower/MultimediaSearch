import os


def detect_language(text: str):
    english_letters = 0
    russian_letters = 0
    for char in text:
        if char.isalpha() and char.isascii():
            english_letters += 1
        elif char.isalpha() and not char.isascii():
            russian_letters += 1
    language = 'en' if english_letters > russian_letters else 'ru'
    return language


def get_file_extension(text: str):
    return os.path.splitext(text)[1]