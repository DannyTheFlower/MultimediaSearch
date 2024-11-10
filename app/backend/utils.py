import os


def detect_language(text: str) -> str:
    """
    Detects whether the text is in English or Russian.

    :param text: The text to analyze.
    :return: 'en' for English, 'ru' for Russian.
    """
    english_letters = 0
    russian_letters = 0
    for char in text:
        if char.isalpha() and char.isascii():
            english_letters += 1
        elif char.isalpha() and not char.isascii():
            russian_letters += 1
    language = 'en' if english_letters > russian_letters else 'ru'
    return language


def get_file_extension(text: str) -> str:
    """
    Extracts the file extension from a filename.

    :param text: The filename.
    :return: The file extension.
    """
    return os.path.splitext(text)[1]