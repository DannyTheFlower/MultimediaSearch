from typing import List, Set, Dict, Union, Optional, Any, Tuple
from pydantic import BaseModel, validator
import numpy as np
import pandas as pd
import time
import os
import pickle
import csv
import fitz
from PIL import Image
import cv2
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
import easyocr
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk import RegexpParser
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.tree import Tree
from tqdm import tqdm
import pymorphy2
from spacy.cli import download
from spacy import load
from sentence_transformers import SentenceTransformer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('wordnet2022', quiet=True)
nlp = load('en_core_web_sm')

# !cp -rf /usr/share/nltk_data/corpora/wordnet2022 /usr/share/nltk_data/corpora/wordnet # temp fix for lookup error.


CSV_FILE_PATH = 'global_final.csv'


class CSVRow(BaseModel):
    """
    Data model representing a row in the global CSV file.

    Attributes:
        is_for_whole_file (bool): Indicates if the backup is aggregated for the whole file.
        filename (str): Name of the file being processed.
        slide_n_piece (Union[int, str]): Slide number or text piece identifier.
        extracted_raw_text (str): The raw text extracted from the file.
        language (Optional[str]): Detected language ('en', 'ru', or None).
        keyphrases (List[str]): List of keyphrases extracted from the text.
        embedding (List[float]): Numerical embedding representation of the text.
    """
    is_for_whole_file: bool
    filename: str
    slide_n_piece: Union[int, str]
    extracted_raw_text: str
    language: Optional[str]  # 'en' or 'ru' or None
    keyphrases: List[str]
    embedding: List[float]

    class Config:
        arbitrary_types_allowed = True

    @validator('language')
    def validate_language(cls, v):
        if v not in ('en', 'ru', None):
            raise ValueError('language must be "en", "ru", or None')
        return v


class LanguageDetectionResult(BaseModel):
    """
    Data model representing the result of language detection.

    Attributes:
        language (Optional[str]): Detected language ('en', 'ru', or None).
    """
    language: Optional[str]


class EmbeddingResult(BaseModel):
    """
    Data model representing the embedding result of a text piece.

    Attributes:
        embedding (List[float]): Numerical embedding representation of the text.
    """
    embedding: List[float]


class KeyphrasesResult(BaseModel):
    """
    Data model representing the keyphrases extracted from a text piece.

    Attributes:
        list_of_keyphrases (List[str]): List of keyphrases.
    """
    list_of_keyphrases: List[str]


#############################################################################


class Embedder:
    """
    EMBEDDER class using TF-IDF for generating embeddings.

    Methods:
        predict(text_piece): Generates an embedding for the given text piece.
    """

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def predict(self, text_piece: str) -> EmbeddingResult:
        """
        Generates an embedding for the given text piece.

        Parameters:
            text_piece (str): The text piece to generate an embedding for.

        Returns:
            EmbeddingResult: The embedding result containing a list of floats.
        """
        embedding = self.model.encode(text_piece)
        return EmbeddingResult(embedding=embedding)


class Translator:
    """
    TRANSLATOR class for translating text between English and Russian.

    Methods:
        translate(text, src_lang, tgt_lang): Translates text from source to target language.
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer_ruen = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
        self.tokenizer_enru = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ru')
        self.model_enru = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-ru').to(self.device)
        self.model_ruen = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-ru-en').to(self.device)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translates text from source language to target language.

        Parameters:
            text (str): The text to translate.
            src_lang (str): Source language code ('en' or 'ru').
            tgt_lang (str): Target language code ('en' or 'ru').

        Returns:
            str: The translated text.
        """
        if src_lang == tgt_lang:
            return text
        if (src_lang, tgt_lang) == ('en', 'ru'):
            tokenizer = self.tokenizer_enru
            model = self.model_enru
        elif (src_lang, tgt_lang) == ('ru', 'en'):
            tokenizer = self.tokenizer_ruen
            model = self.model_ruen
        else:
            return ""
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True).to(self.device)
        outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text


class OCRModel:
    """
    OCR class for extracting text from images using EasyOCR.

    Methods:
        extract_text(image): Extracts text from a given image.
    """

    def __init__(self):
        self.reader = easyocr.Reader(['en', 'ru'], gpu=torch.cuda.is_available())

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text from a given image using EasyOCR.

        Parameters:
            image (Image.Image): The image to extract text from.

        Returns:
            str: The extracted text.
        """
        image_np = np.array(image)
        results = self.reader.readtext(image_np)
        text = ' '.join([result[1] for result in results])
        return text


class Chart2Text:
    """
    CHART2TEXT class for extracting text descriptions from chart images.

    Methods:
        extract_text(image): Extracts descriptive text from a chart image.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma",
                                                                       torch_dtype=torch.float16).to(self.device)
        self.processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts descriptive text from a chart image using the Chart2Text model.

        Parameters:
            image (Image.Image): The chart image to process.

        Returns:
            str: The extracted descriptive text.
        """
        input_text = "program of thought: describe the chart in great detail"
        inputs = self.processor(text=input_text, images=image, return_tensors="pt").to(self.device)
        prompt_length = inputs['input_ids'].shape[1]
        generate_ids = self.model.generate(**inputs, num_beams=4, max_new_tokens=512)
        output_text = self.processor.batch_decode(
            generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text


EMBEDDER = Embedder()
TRANSLATOR = Translator()
OCR = OCRModel()
CHART2TEXT = Chart2Text()

STOPWORDS = set(stopwords.words('english') + stopwords.words('russian'))
LEMMATIZER_EN = WordNetLemmatizer()
LEMMATIZER_RU = pymorphy2.MorphAnalyzer()
MORPH_RULES = [
    [{"POS": "NOUN", "case": ["nomn", "accs"]}],
    [{"POS": "ADJF", "case": ["nomn", "accs"]}, {"POS": "NOUN", "case": ["nomn", "accs"]}],
    [{"POS": "NOUN", "case": ["nomn", "accs"]}, {"POS": "NOUN", "case": ["gent"]}],
    [{"POS": "NOUN", "case": ["nomn", "accs"]}, {"POS": "ADJF", "case": ["gent"]}, {"POS": "NOUN", "case": ["gent"]}],
    [{"POS": "ADJF", "case": ["nomn", "accs"]}, {"POS": "ADJF", "case": ["nomn", "accs"]},
     {"POS": "NOUN", "case": ["nomn", "accs"]}],
    [{"POS": "ADJF", "case": ["nomn", "accs"]}, {"POS": "NOUN", "case": ["nomn", "accs"]},
     {"POS": "NOUN", "case": ["gent"]}]
]

VECTORIZER_EN = TfidfVectorizer()
VECTORIZER_RU = TfidfVectorizer()


# DATA = pd.read_csv(CSV_FILE_PATH).dropna(subset=['keyphrases']).reset_index(drop=True)


#########################################################################


def preprocess_text(
        text: Optional[str],
        lang: str = 'en',
        lowercase: bool = True,
        clean: bool = True,
        lemmatize: bool = True,
        stopwords: Optional[Set[str]] = None
) -> str:
    global LEMMATIZER_EN, LEMMATIZER_RU
    """
    Preprocesses the input text by performing operations like lowercasing, cleaning,
    lemmatization, and stopword removal.

    Parameters:
        text (Optional[str]): The input text to preprocess.
        lang (str): The language of the text ('en' or 'ru'). Defaults to 'en'.
        lowercase (bool): Whether to convert the text to lowercase. Defaults to True.
        clean (bool): Whether to remove non-alphanumeric characters. Defaults to True.
        lemmatize (bool): Whether to lemmatize the words. Defaults to True.
        stopwords (Optional[Set[str]]): A set of stopwords to remove. Defaults to STOPWORDS.

    Returns:
        str: The preprocessed text.
    """
    if text is None or str(text).lower() == 'nan':
        return ''
    if lowercase:
        text = text.lower()
    if clean:
        if lang == 'en':
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        elif lang == 'ru':
            text = re.sub(r'[^а-яА-ЯёЁ0-9\s]', '', text)
    words = text.split()
    if lemmatize:
        if lang == 'en':
            words = [LEMMATIZER_EN.lemmatize(word) for word in words]
        elif lang == 'ru':
            words = [LEMMATIZER_RU.parse(word)[0].normal_form for word in words]
    if stopwords is None:
        stopwords_set = STOPWORDS
    else:
        stopwords_set = stopwords
    if stopwords_set:
        words = [word for word in words if word not in stopwords_set]
    return ' '.join(words)


def extract_keywords_en(
        text: str,
        preprocess_afterwards: bool = True,
        recursive: bool = False
) -> List[str]:
    """
    Extracts nouns and noun phrases from English text.

    Parameters:
        text (str): The input text.
        preprocess_afterwards (bool): If True, applies preprocess_text to the extracted words/phrases.
        recursive (bool): If True, also extracts all possible subphrases from the found noun phrases.

    Returns:
        List[str]: A list of unique nouns and noun phrases.
    """
    # Tokenization and POS-tagging
    words = word_tokenize(text)
    pos_tags = pos_tag(words)

    # Define grammar for noun phrases
    grammar = "NP: {<DT>?<JJ>*<NN.*>+}"

    # Create parser and parse POS-tagged text
    chunk_parser = RegexpParser(grammar)
    noun_phrases_tree = chunk_parser.parse(pos_tags)

    # Extract noun phrases and individual nouns
    noun_phrases = []
    for subtree in noun_phrases_tree:
        if isinstance(subtree, Tree) and subtree.label() == "NP":
            # Join the words of the noun phrase into a single string
            item = " ".join(word for word, pos in subtree.leaves())
            noun_phrases.append(preprocess_text(item) if preprocess_afterwards else item)
        elif isinstance(subtree, tuple) and subtree[1].startswith("NN"):
            noun_phrases.append(preprocess_text(subtree[0]) if preprocess_afterwards else subtree[0])

    # Add subphrases if recursive is True
    if recursive:
        final_nouns_and_phrases = set(noun_phrases)  # Use a set to avoid duplicates
        for phrase in noun_phrases:
            # Create subphrases from the noun phrase
            words_in_phrase = phrase.split()
            for i in range(len(words_in_phrase)):
                for j in range(i + 1, len(words_in_phrase) + 1):
                    subphrase = " ".join(words_in_phrase[i:j])
                    final_nouns_and_phrases.add(preprocess_text(subphrase) if preprocess_afterwards else subphrase)
        return list(final_nouns_and_phrases)
    else:
        # If recursive=False, return only the found noun phrases and nouns
        return list(set(noun_phrases))  # Use set for uniqueness


def match_morph_rule(
        text: str,
        rule: List[Dict[str, Any]]
) -> bool:
    global LEMMATIZER_RU
    """
    Checks if the given text matches the specified morphological rule.

    Parameters:
        text (str): The text to check.
        rule (List[Dict[str, Any]]): The morphological rule to match against.

    Returns:
        bool: True if the text matches the rule, False otherwise.
    """
    words = text.split()
    if len(words) != len(rule):
        return False
    for word, rule_part in zip(words, rule):
        parsed = LEMMATIZER_RU.parse(word)[0]
        if parsed.tag.POS != rule_part["POS"]:
            return False
        if parsed.tag.case not in rule_part["case"]:
            return False
    return True


def extract_keywords_ru(
        text: str,
        remove_stopwords: bool = True,
        recursive: bool = False
) -> List[str]:
    global MORPH_RULES
    """
    Extracts nouns and noun phrases from Russian text based on morphological rules.

    Parameters:
        text (str): The input text.
        remove_stopwords (bool): If True, removes stopwords during preprocessing.
        recursive (bool): If True, continues extraction recursively.

    Returns:
        List[str]: A list of extracted terms (nouns and noun phrases).
    """
    terms = []
    tokens = preprocess_text(text, lang='ru') if remove_stopwords else preprocess_text(text, lang='ru', stopwords=None)
    tokens = tokens.split()
    i = 0
    while i < len(tokens):
        matched = False
        for rule in MORPH_RULES:
            phrase_tokens = tokens[i:i + len(rule)]
            if len(phrase_tokens) < len(rule):
                continue
            phrase = ' '.join(phrase_tokens)
            if match_morph_rule(phrase, rule):
                terms.append(phrase)
                matched = True
                if not recursive:
                    i += len(rule) - 1
                break
        i += 1
    return list(set(terms))


def get_keyphrases_for_text_piece(
        text_piece: str,
        language: str,
        recursive: bool = True
) -> KeyphrasesResult:
    """
    Extracts keyphrases from a text piece.

    Parameters:
        text_piece (str): The text piece to extract keyphrases from.
        language (str): The language of the text piece ('en' or 'ru').
        recursive (bool): Flag indicating recursive extraction.

    Returns:
        KeyphrasesResult: An object containing the list of keyphrases.
    """
    if language == 'en':
        keyphrases = extract_keywords_en(text_piece, preprocess_afterwards=True, recursive=recursive)
    elif language == 'ru':
        keyphrases = extract_keywords_ru(text_piece, recursive=recursive)
    else:
        keyphrases = []
    keyphrases.extend(re.findall(r'\b\d{4}\b', text_piece))
    return KeyphrasesResult(list_of_keyphrases=keyphrases)


def detect_language(text: str) -> LanguageDetectionResult:
    """
    Detects the language of the given text.

    Parameters:
        text (str): The text whose language needs to be detected.

    Returns:
        LanguageDetectionResult: The result containing the detected language.
    """
    english_letters = 0
    russian_letters = 0
    for char in text:
        if char.isalpha() and char.isascii():
            english_letters += 1
        elif char.isalpha() and not char.isascii():
            russian_letters += 1
    language = 'en' if english_letters > russian_letters else 'ru'
    return LanguageDetectionResult(language=language)


def get_embedding_for_text_piece(text_piece: str, embedder: Embedder) -> EmbeddingResult:
    """
    Generates an embedding for the given text piece using the provided embedder.

    Parameters:
        text_piece (str): The text piece to generate an embedding for.
        embedder (Embedder): The embedder instance to use.

    Returns:
        EmbeddingResult: The embedding result containing a list of floats.
    """
    embedding_result = embedder.predict(text_piece)
    return embedding_result


def get_file_extension(filename: str) -> str:
    """
    Retrieves and validates the file extension.

    Parameters:
        filename (str): The name of the file.

    Returns:
        str: The file extension ('.txt' or '.pdf').

    Raises:
        ValueError: If the file extension is not supported.
    """
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ['.txt', '.pdf']:
        raise ValueError('Unsupported file extension')
    return extension


def aggregate_keyphrases(keyphrases_list: List[List[str]]) -> List[str]:
    """
    Aggregates multiple lists of keyphrases into a single list.

    Parameters:
        keyphrases_list (List[List[str]]): A list of keyphrase lists.

    Returns:
        List[str]: The aggregated list of keyphrases.
    """
    aggregated_keyphrases = []
    for keyphrases in keyphrases_list:
        aggregated_keyphrases.extend(keyphrases)
    return list(set(aggregated_keyphrases))


def aggregate_embeddings(embeddings_list: List[List[float]]) -> List[float]:
    """
    Aggregates multiple embeddings using mean pooling.

    Parameters:
        embeddings_array (List[List[float]]): A list of embeddings.

    Returns:
        List[float]: The aggregated embedding.
    """
    embeddings_array = np.array(embeddings_list)
    aggregated_embedding = embeddings_array.mean(axis=0)
    return aggregated_embedding.tolist()


def translate(text: str, text_language: str, desired_language: str, translator: Translator) -> str:
    """
    Translates text from the source language to the desired language using the provided translator.

    Parameters:
        text (str): The text to translate.
        text_language (str): The language of the text.
        desired_language (str): The language to translate the text into.
        translator (Translator): The translator instance to use.

    Returns:
        str: The translated text.
    """
    if text_language == desired_language:
        return text
    else:
        return translator.translate(text, text_language, desired_language)


def split_text_into_chunks(text: str, chunk_size: int = 250, overlap: int = 10) -> List[str]:
    """
    Splits the text into chunks of specified size with overlap.

    Parameters:
        text (str): The text to split.
        chunk_size (int): The maximum number of tokens in each chunk.
        overlap (int): The number of tokens to overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    tokens = word_tokenize(text)
    total_tokens = len(tokens)
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def get_text_pieces_from_txt_file(
        filepath: str,
        chunk_size: int = 250,
        overlap: int = 15
) -> List[str]:
    """
    Extracts text pieces from a TXT file, splitting it into chunks.

    Parameters:
        filepath (str): The path to the TXT file.
        chunk_size (int): The maximum number of tokens in each chunk.
        overlap (int): The number of tokens to overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    return chunks


def get_text_pieces_from_pdf_file(
        filepath: str,
        index_in_two_languages: bool,
        too_few_chars: int = 50
) -> Dict[str, List[str]]:
    """
    Extracts text pieces from a PDF file, handling pages with too few characters by using OCR or CHART2TEXT.

    Parameters:
        filepath (str): The path to the PDF file.
        index_in_two_languages (bool): Whether to index in two languages.
        too_few_chars (int): Threshold for minimal acceptable characters in extracted text.

    Returns:
        Dict[str, List[str]]: Dictionary containing text pieces in 'en' and/or 'ru'.
    """
    global EMBEDDER, TRANSLATOR, OCR, CHART2TEXT

    doc = fitz.open(filepath)
    text_pieces_en = []
    text_pieces_ru = []
    embedder = EMBEDDER
    translator = TRANSLATOR
    ocr = OCR
    chart2text = CHART2TEXT
    language = None
    first_three_pages_text = ''
    language_detected = False

    for page_number, page in enumerate(doc):
        chart2text_was_used = False
        page_text = page.get_text()
        if len(page_text.strip()) < too_few_chars and page_number not in [0, len(doc) - 1]:
            # Use OCR for text extraction if too few characters
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            page_text = ocr.extract_text(img)
            if len(page_text.strip()) < too_few_chars:
                # Use CHART2TEXT if still too few characters
                page_text = chart2text.extract_text(img)
                chart2text_was_used = True
        if page_number < 3:
            first_three_pages_text += ' ' + page_text
            if not language_detected and len(first_three_pages_text.strip()) > too_few_chars:
                # Detect language from the first three pages
                language_result = detect_language(first_three_pages_text)
                language = language_result.language or 'en'
                language_detected = True
        # Accumulate text pieces for indexing
        if language == 'ru' and not chart2text_was_used:
            if index_in_two_languages:
                translated_text = translate(page_text, 'ru', 'en', translator)
                text_pieces_en.append(translated_text)
            text_pieces_ru.append(page_text)
        else:
            if index_in_two_languages:
                translated_text = translate(page_text, 'en', 'ru', translator)
                text_pieces_ru.append(translated_text)
            text_pieces_en.append(page_text)

    result = {}
    if text_pieces_en:
        result['en'] = text_pieces_en
    if text_pieces_ru:
        result['ru'] = text_pieces_ru
    return result


def write_data_to_csv(data: CSVRow):
    global CSV_FILE_PATH
    """
    Writes backup to the global CSV file. Handles concurrent access by using a lock.

    Parameters:
        backup (CSVRow): The backup to write to the CSV file.
    """
    while True:
        try:
            file_exists = os.path.isfile(CSV_FILE_PATH)
            with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['is_for_whole_file', 'filename', 'slide_n_piece', 'extracted_raw_text',
                              'language', 'keyphrases', 'embedding']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'is_for_whole_file': data.is_for_whole_file,
                    'filename': data.filename,
                    'slide_n_piece': data.slide_n_piece,
                    'extracted_raw_text': data.extracted_raw_text,
                    'language': data.language,
                    'keyphrases': ','.join(data.keyphrases),
                    'embedding': ','.join(map(str, data.embedding))
                })
            break
        except IOError:
            # Wait and retry if the file is being accessed by another process
            time.sleep(1)
            continue


def index_file(
        filepath: str,
        index_in_two_languages: bool = True,
        too_few_chars: int = 50,
        chunk_size: int = 250,
        overlap: int = 15,
        recursive_keyphrases: bool = True
):
    """
    Main function to index a file (TXT or PDF). Processes text pieces, extracts keyphrases, embeddings, and writes backup to CSV.

    Parameters:
        filepath (str): The path to the file to index.
        index_in_two_languages (bool): Whether to index the file in two languages.
        too_few_chars (int): Threshold for minimal acceptable characters in extracted text.
        chunk_size (int): The maximum number of tokens in each chunk for txt-file processing.
        overlap (int): The number of tokens to overlap between chunks for txt-file processing.
        recursive_keyphrases (bool): Flag to recursive keyphrases extraction.
    """
    global EMBEDDER, TRANSLATOR

    filename = os.path.basename(filepath)
    extension = get_file_extension(filename)
    embedder = EMBEDDER
    translator = TRANSLATOR
    embeddings_en = []
    keyphrases_en = []
    embeddings_ru = []
    keyphrases_ru = []

    if extension == '.txt':
        # Processing TXT files
        text_pieces = get_text_pieces_from_txt_file(filepath, chunk_size, overlap)
        language_result = detect_language(text_pieces[0])
        language = language_result.language or 'en'
        if index_in_two_languages:
            for idx, text_piece in enumerate(text_pieces):
                # Extract keyphrases and embeddings for original language
                keyphrases_result = get_keyphrases_for_text_piece(text_piece, language, recursive_keyphrases)
                embedding_result = get_embedding_for_text_piece(text_piece, embedder)
                csv_row = CSVRow(
                    is_for_whole_file=False,
                    filename=filename,
                    slide_n_piece=idx + 1,
                    extracted_raw_text=text_piece,
                    language=language,
                    keyphrases=keyphrases_result.list_of_keyphrases,
                    embedding=embedding_result.embedding
                )
                write_data_to_csv(csv_row)
                if language == 'en':
                    embeddings_en.append(embedding_result.embedding)
                    keyphrases_en.append(keyphrases_result.list_of_keyphrases)
                else:
                    embeddings_ru.append(embedding_result.embedding)
                    keyphrases_ru.append(keyphrases_result.list_of_keyphrases)
                # Process translation
                desired_language = 'ru' if language == 'en' else 'en'
                translated_text = translate(text_piece, language, desired_language, translator)
                keyphrases_result_trans = get_keyphrases_for_text_piece(translated_text, desired_language,
                                                                        recursive_keyphrases)
                embedding_result_trans = get_embedding_for_text_piece(translated_text, embedder)
                csv_row_trans = CSVRow(
                    is_for_whole_file=False,
                    filename=filename,
                    slide_n_piece=idx + 1,
                    extracted_raw_text=translated_text,
                    language=desired_language,
                    keyphrases=keyphrases_result_trans.list_of_keyphrases,
                    embedding=embedding_result_trans.embedding
                )
                write_data_to_csv(csv_row_trans)
                if desired_language == 'en':
                    embeddings_en.append(embedding_result_trans.embedding)
                    keyphrases_en.append(keyphrases_result_trans.list_of_keyphrases)
                else:
                    embeddings_ru.append(embedding_result_trans.embedding)
                    keyphrases_ru.append(keyphrases_result_trans.list_of_keyphrases)
        else:
            # Indexing in one language
            for idx, text_piece in enumerate(text_pieces):
                keyphrases_result = get_keyphrases_for_text_piece(text_piece, language, recursive_keyphrases)
                embedding_result = get_embedding_for_text_piece(text_piece, embedder)
                csv_row = CSVRow(
                    is_for_whole_file=False,
                    filename=filename,
                    slide_n_piece=idx + 1,
                    extracted_raw_text=text_piece,
                    language=language,
                    keyphrases=keyphrases_result.list_of_keyphrases,
                    embedding=embedding_result.embedding
                )
                write_data_to_csv(csv_row)
                if language == 'en':
                    embeddings_en.append(embedding_result.embedding)
                    keyphrases_en.append(keyphrases_result.list_of_keyphrases)
                else:
                    embeddings_ru.append(embedding_result.embedding)
                    keyphrases_ru.append(keyphrases_result.list_of_keyphrases)
        # Aggregate backup for the whole file
        if embeddings_en:
            aggregated_embedding_en = aggregate_embeddings(embeddings_en)
            aggregated_keyphrases_en = aggregate_keyphrases(keyphrases_en)
            csv_row = CSVRow(
                is_for_whole_file=True,
                filename=filename,
                slide_n_piece='whole',
                extracted_raw_text='',
                language='en',
                keyphrases=aggregated_keyphrases_en,
                embedding=aggregated_embedding_en
            )
            write_data_to_csv(csv_row)
        if embeddings_ru:
            aggregated_embedding_ru = aggregate_embeddings(embeddings_ru)
            aggregated_keyphrases_ru = aggregate_keyphrases(keyphrases_ru)
            csv_row = CSVRow(
                is_for_whole_file=True,
                filename=filename,
                slide_n_piece='whole',
                extracted_raw_text='',
                language='ru',
                keyphrases=aggregated_keyphrases_ru,
                embedding=aggregated_embedding_ru
            )
            write_data_to_csv(csv_row)
    elif extension == '.pdf':
        # Processing PDF files
        text_pieces_dict = get_text_pieces_from_pdf_file(filepath, index_in_two_languages, too_few_chars)
        if 'en' in text_pieces_dict:
            text_pieces_en = text_pieces_dict['en']
            for idx, text_piece in enumerate(text_pieces_en):
                keyphrases_result = get_keyphrases_for_text_piece(text_piece, 'en', recursive_keyphrases)
                embedding_result = get_embedding_for_text_piece(text_piece, embedder)
                csv_row = CSVRow(
                    is_for_whole_file=False,
                    filename=filename,
                    slide_n_piece=idx + 1,
                    extracted_raw_text=text_piece,
                    language='en',
                    keyphrases=keyphrases_result.list_of_keyphrases,
                    embedding=embedding_result.embedding
                )
                write_data_to_csv(csv_row)
                embeddings_en.append(embedding_result.embedding)
                keyphrases_en.append(keyphrases_result.list_of_keyphrases)
            aggregated_embedding_en = aggregate_embeddings(embeddings_en)
            aggregated_keyphrases_en = aggregate_keyphrases(keyphrases_en)
            csv_row = CSVRow(
                is_for_whole_file=True,
                filename=filename,
                slide_n_piece='whole',
                extracted_raw_text='',
                language='en',
                keyphrases=aggregated_keyphrases_en,
                embedding=aggregated_embedding_en
            )
            write_data_to_csv(csv_row)
        if 'ru' in text_pieces_dict:
            text_pieces_ru = text_pieces_dict['ru']
            for idx, text_piece in enumerate(text_pieces_ru):
                keyphrases_result = get_keyphrases_for_text_piece(text_piece, 'ru', recursive_keyphrases)
                embedding_result = get_embedding_for_text_piece(text_piece, embedder)
                csv_row = CSVRow(
                    is_for_whole_file=False,
                    filename=filename,
                    slide_n_piece=idx + 1,
                    extracted_raw_text=text_piece,
                    language='ru',
                    keyphrases=keyphrases_result.list_of_keyphrases,
                    embedding=embedding_result.embedding
                )
                write_data_to_csv(csv_row)
                embeddings_ru.append(embedding_result.embedding)
                keyphrases_ru.append(keyphrases_result.list_of_keyphrases)
            aggregated_embedding_ru = aggregate_embeddings(embeddings_ru)
            aggregated_keyphrases_ru = aggregate_keyphrases(keyphrases_ru)
            csv_row = CSVRow(
                is_for_whole_file=True,
                filename=filename,
                slide_n_piece='whole',
                extracted_raw_text='',
                language='ru',
                keyphrases=aggregated_keyphrases_ru,
                embedding=aggregated_embedding_ru
            )
            write_data_to_csv(csv_row)
    else:
        raise ValueError('Unsupported file extension')


def build_tfidf(index_storage_path: str):
    """
    Creates TF-IDF vectorizers for English and Russian keyphrases.

    Parameters:
        index_storage_path (str): Path to store the inverted indexes and TF-IDF vectorizers.
    """
    global VECTORIZER_EN, VECTORIZER_RU, DATA

    if not os.path.exists(index_storage_path):
        os.mkdir(index_storage_path)

    data_en = []
    data_ru = []

    # Read the CSV file and separate backup by language
    for _, row in DATA.iterrows():
        if row['is_for_whole_file'] == False:
            language = row['language']
            keyphrases = row['keyphrases']

            if language == 'en':
                data_en.append(keyphrases)
            elif language == 'ru':
                data_ru.append(keyphrases)

    # Create TF-IDF vectorizer and fit on keyphrases for English
    if data_en:
        tfidf_matrix_en = VECTORIZER_EN.fit_transform(data_en)
        # Save the vectorizer and backup
        with open(os.path.join(index_storage_path, 'tfidf_vectorizer_en.pkl'), 'wb') as f:
            pickle.dump(VECTORIZER_EN, f)
        np.save(os.path.join(index_storage_path, 'tfidf_matrix_en.npy'), tfidf_matrix_en.toarray())

    # Create TF-IDF vectorizer and fit on keyphrases for Russian
    if data_ru:
        tfidf_matrix_ru = VECTORIZER_RU.fit_transform(data_ru)
        # Save the vectorizer and backup
        with open(os.path.join(index_storage_path, 'tfidf_vectorizer_ru.pkl'), 'wb') as f:
            pickle.dump(VECTORIZER_RU, f)
        np.save(os.path.join(index_storage_path, 'tfidf_matrix_ru.npy'), tfidf_matrix_ru.toarray())


def retrieval_stage(keywords: List[str], language: str, topk: int = 100,
                    index_storage_path: str = None) -> pd.DataFrame:
    """
    Retrieves the top-k documents based on TF-IDF cosine similarity of keyphrases.

    Parameters:
        keywords (List[str]): List of keyphrases from the user's query.
        language (str): Language of the query ('en' or 'ru').
        topk (int): Number of top documents to retrieve.
        index_storage_path (str): Path where the inverted indexes and TF-IDF vectorizers are stored.

    Returns:
        pd.DataFrame: DataFrame containing document backup and similarity scores.
    """
    global DATA

    if language == 'en':
        vectorizer_path = os.path.join(index_storage_path, 'tfidf_vectorizer_en.pkl')
        tfidf_matrix_path = os.path.join(index_storage_path, 'tfidf_matrix_en.npy')
    elif language == 'ru':
        vectorizer_path = os.path.join(index_storage_path, 'tfidf_vectorizer_ru.pkl')
        tfidf_matrix_path = os.path.join(index_storage_path, 'tfidf_matrix_ru.npy')
    else:
        return pd.DataFrame()

    # Load the TF-IDF vectorizer and matrix
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = np.load(tfidf_matrix_path)

    # Transform the query keyphrases
    query_tfidf = vectorizer.transform([' '.join(keywords)])
    # Compute cosine similarity
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)[0]
    # Get top-k indices
    topk_indices = similarities.argsort()[-topk:][::-1]
    topk_similarities = similarities[topk_indices]
    # Get the candidate documents
    candidates = DATA.iloc[topk_indices].copy()
    # Add the similarities to the candidates DataFrame
    candidates['tfidf_similarity'] = topk_similarities
    # Sort candidates by tfidf_similarity
    candidates = candidates.sort_values(by='tfidf_similarity', ascending=False)
    return candidates


def ranking_stage(query_embedding: List[float], candidates: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
    """
    Ranks the candidate documents based on embedding cosine similarity.

    Parameters:
        query_embedding (List[float]): Embedding of the user's query.
        candidates (pd.DataFrame): Candidate documents from the retrieval stage.
        topk (int): Number of top documents to return.

    Returns:
        pd.DataFrame: DataFrame of top-k documents with highest embedding similarity.
    """
    # Convert embeddings from strings to numpy arrays
    candidate_embeddings = np.stack(
        candidates['embedding'].apply(lambda x: np.array(eval('[' + x + ']'))).values
    )
    query_embedding = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
    # Add similarities to candidates DataFrame
    candidates = candidates.copy()  # To avoid modifying the original DataFrame
    candidates['embedding_similarity'] = similarities
    # Sort candidates by embedding similarity
    candidates = candidates.sort_values(by='embedding_similarity', ascending=False)
    # Return top-k candidates
    return candidates.head(topk)


def search(
        query: str,
        topk_retrieval: int = 100,
        topk_ranking: int = 5,
        use_tfidf: bool = False,
        use_embedder: bool = True,
        index_storage_path: str = None
):
    """
    Performs the search operation without using embeddings, ranking only by TF-IDF similarities.

    Parameters:
        query (str): The user's search query.
        topk_retrieval (int): Number of top documents to retrieve in the retrieval stage.
        topk_ranking (int): Number of top documents to return.
        use_tfidf (bool): Whether to use TF-IDF vectorizer or not.
        use_embedder (bool): Whether to use embedder or not.
        index_storage_path (str): Path where the inverted indexes and TF-IDF vectorizers are stored.

    Returns:
        pd.DataFrame: DataFrame of top-k search results.
    """
    global TRANSLATOR, STOPWORDS, EMBEDDER

    # Detect language of the query
    language_result = detect_language(query)
    query_language = language_result.language

    # Translate the query to the other language
    if query_language == 'en':
        translated_query = TRANSLATOR.translate(query, 'en', 'ru')
        languages = ['en', 'ru']
        queries = [query, translated_query]
    elif query_language == 'ru':
        translated_query = TRANSLATOR.translate(query, 'ru', 'en')
        languages = ['ru', 'en']
        queries = [query, translated_query]
    else:
        # Default to English if language detection fails
        query_language = 'en'
        languages = ['en']
        queries = [query]

    all_results = []
    for lang, q in zip(languages, queries):
        # Extract keyphrases from the query
        preprocessed_query = preprocess_text(q, lang=lang, stopwords=STOPWORDS)
        keyphrases_result = get_keyphrases_for_text_piece(preprocessed_query, lang)
        keyphrases = keyphrases_result.list_of_keyphrases

        # Retrieve top-k documents based on TF-IDF cosine similarity
        if use_tfidf:
            candidates = retrieval_stage(keyphrases, lang, topk_retrieval, index_storage_path)
        else:
            candidates = DATA

        if candidates.empty:
            continue  # Skip if no candidates found in this language

        if use_embedder:
            embedding_result = EMBEDDER.predict(q)
            query_embedding = embedding_result.embedding
            topk_candidates = ranking_stage(query_embedding, candidates, topk_ranking)
        else:
            topk_candidates = candidates.head(topk_ranking)

        all_results.append(topk_candidates)

    # If results from both languages, combine and get top-k
    if all_results:
        all_results_df = pd.concat(all_results)
        sort_by = 'embedding_similarity' if use_embedder else 'tfidf_similarity'
        all_results_df = all_results_df.sort_values(by=sort_by, ascending=False)
        return all_results_df.head(topk_ranking)
    else:
        return pd.DataFrame()
