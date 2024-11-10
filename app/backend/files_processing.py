from backend.utils import get_file_extension
from backend.config import config
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from easyocr import Reader
import os
import time
import torch
import csv
import numpy as np
from PIL import Image
import fitz
import streamlit as st
from typing import List


class OCRModel:
    """
    Optical Character Recognition (OCR) model using EasyOCR.
    """
    def __init__(self):
        self.reader = Reader(['en', 'ru'], gpu=torch.cuda.is_available())

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text from an image using OCR.

        :param image: PIL Image to extract text from.
        :return: Extracted text.
        """
        image_np = np.array(image)
        results = self.reader.readtext(image_np)
        text = ' '.join([result[1] for result in results])
        return text


class Chart2Text:
    """
    Model to extract textual descriptions from charts.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma",
                                                                       torch_dtype=torch.float16).to(self.device)
        self.processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text from a chart image using Chart2Text model.

        :param image: PIL Image of the chart.
        :return: Extracted text description.
        """
        input_text = "program of thought: describe the chart in great detail"
        inputs = self.processor(text=input_text, images=image, return_tensors="pt").to(self.device)
        prompt_length = inputs['input_ids'].shape[1]
        generate_ids = self.model.generate(**inputs, num_beams=4, max_new_tokens=512)
        output_text = self.processor.batch_decode(
            generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text


@st.cache_resource(show_spinner=False)
def load_ocr():
    """
    Loads the OCR model as a cached resource.

    :return: OCRModel instance.
    """
    return OCRModel


@st.cache_resource(show_spinner=False)
def load_chart2text():
    """
    Loads the Chart2Text model as a cached resource.

    :return: Chart2Text instance.
    """
    return Chart2Text()


@st.cache_resource(show_spinner=False)
def load_fp_resources(
    include_ocr: bool = config.INCLUDE_OCR,
    include_chart2text: bool = config.INCLUDE_CHART2TEXT
):
    """
    Loads file processing resources based on the configuration.

    :param include_ocr: Flag to include OCR model.
    :param include_chart2text: Flag to include Chart2Text model.
    :return: Loaded resources.
    """
    if include_ocr and include_chart2text:
        return load_ocr(), load_chart2text()
    if include_ocr:
        return load_ocr(), None
    if include_chart2text:
        return None, load_chart2text()
    return None, None


def get_text_pieces_from_txt_file(
        filepath: str,
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.OVERLAP
) -> List[str]:
    """
    Splits text from a TXT file into pieces.

    :param filepath: Path to the TXT file.
    :param chunk_size: Number of tokens in each chunk.
    :param overlap: Number of overlapping tokens between chunks.
    :return: List of text chunks.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = text.split()
    total_tokens = len(tokens)
    chunks = []
    start = 0
    while start < total_tokens:
        chunk_tokens = tokens[start:min(start + chunk_size, total_tokens)]
        chunks.append(" ".join(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def get_text_pieces_from_pdf_file(
        filepath: str,
        too_few_chars: int = config.TOO_FEW_CHARS
) -> List[str]:
    """
    Extracts text pieces from a PDF file, using OCR if necessary.

    :param filepath: Path to the PDF file.
    :param too_few_chars: Threshold for using OCR or Chart2Text.
    :return: List of text pieces.
    """
    ocr, chart2text = load_fp_resources()

    doc = fitz.open(filepath)
    text_pieces = []
    for page_number, page in enumerate(doc):
        page_text = page.get_text()
        if len(page_text.strip()) < too_few_chars and page_number not in [0, len(doc) - 1]:
            # Use OCR for text extraction if too few characters
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            page_text = ocr.extract_text(img)
            if len(page_text.strip()) < too_few_chars and chart2text is not None:
                # Use CHART2TEXT if still too few characters
                page_text = chart2text.extract_text(img)
        text_pieces.append(page_text)
    return text_pieces


def write_data_to_csv(data: dict, filepath: str = None):
    """
    Writes a single row of data to a CSV file.

    :param data: Dictionary containing 'filename', 'n_slide', and 'text'.
    :param filepath: Path to the CSV file.
    """
    filepath = config.CSV_FILE_PATH if filepath is None else filepath
    while True:
        try:
            file_exists = os.path.isfile(filepath)
            with open(filepath, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['filename', 'n_slide', 'text'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'filename': data['filename'],
                    'n_slide': data['n_slide'],
                    'text': data['text'],
                })
            break
        except IOError:
            time.sleep(1)
            continue


def upload_filedata_to_csv_file(
        filepath_from: str,
        filepath_to: str = config.TEMP_DATA_CSV,
        too_few_chars: int = config.TOO_FEW_CHARS,
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.OVERLAP,
):
    """
    Processes a file and uploads its data to a CSV file.

    :param filepath_from: Source file path.
    :param filepath_to: Destination CSV file path.
    :param too_few_chars: Threshold for text extraction methods.
    :param chunk_size: Chunk size for TXT files.
    :param overlap: Overlap size for TXT files.
    """
    filename = os.path.basename(filepath_from)
    extension = get_file_extension(filename)
    if extension == '.txt':
        text_pieces = get_text_pieces_from_txt_file(filepath_from, chunk_size, overlap)
        for idx, text_piece in enumerate(text_pieces):
            csv_row = {'filename': filename, 'n_slide': None, 'text': text_piece}
            write_data_to_csv(csv_row, filepath_to)
    elif extension == '.pdf':
        text_pieces = get_text_pieces_from_pdf_file(filepath_from, too_few_chars)
        for idx, text_piece in enumerate(text_pieces):
            csv_row = {'filename': filename, 'n_slide': idx + 1, 'text': text_piece}
            write_data_to_csv(csv_row, filepath_to)
    else:
        raise ValueError('Unsupported file extension')
