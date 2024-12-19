from .utils import get_file_extension
from .config import config
from transformers import AutoProcessor
from qdrant_client import QdrantClient
import torch
import numpy as np
from PIL import Image
import fitz
from typing import List
from functools import lru_cache
import easyocr
from transformers import PaliGemmaForConditionalGeneration


class OCRModel:
    """
    Optical Character Recognition (OCR) model using EasyOCR.
    """
    def __init__(self):
        self.reader = easyocr.Reader(['en', 'ru'], gpu=torch.cuda.is_available())

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


@lru_cache(maxsize=1)
def load_ocr() -> OCRModel:
    """
    Loads the OCR model as a cached resource.

    :return: OCRModel instance.
    """
    return OCRModel()


@lru_cache(maxsize=1)
def load_chart2text() -> Chart2Text:
    """
    Loads the Chart2Text model as a cached resource.

    :return: Chart2Text instance.
    """
    return Chart2Text()


@lru_cache(maxsize=1)
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
    ocr = load_ocr() if include_ocr else None
    c2t = load_chart2text() if include_chart2text else None
    return ocr, c2t


@lru_cache(maxsize=1)
def get_qdrant_client():
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


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
            if ocr is not None:
                page_text = ocr.extract_text(img)
            if len(page_text.strip()) < too_few_chars and chart2text is not None:
                # Use CHART2TEXT if still too few characters
                page_text = chart2text.extract_text(img)
        text_pieces.append(page_text)
    return text_pieces


def prepare_filedata_for_qdrant(
        filepath_from: str,
        too_few_chars: int = config.TOO_FEW_CHARS,
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.OVERLAP
):
    filename = filepath_from.split('/')[-1]
    extension = get_file_extension(filename)
    if extension == '.txt':
        text_pieces = get_text_pieces_from_txt_file(filepath_from, chunk_size, overlap)
        data_rows = [{"filename": filename, "n_slide": -1, "text": t} for t in text_pieces]
    elif extension == '.pdf':
        text_pieces = get_text_pieces_from_pdf_file(filepath_from, too_few_chars)
        data_rows = [{"filename": filename, "n_slide": i + 1, "text": txt} for i, txt in enumerate(text_pieces)]
    else:
        raise ValueError('Unsupported file extension')

    return data_rows
