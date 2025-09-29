import PyPDF2
import pandas as pd
from io import BytesIO
from typing import Dict, List, Optional, Union
import re

class PDFProcessor:
    """Handles PDF text extraction and basic preprocessing."""

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract all text from uploaded PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"

            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_excel_data(self, excel_file) -> Dict[str, pd.DataFrame]:
        """Extract data from Excel file, returning all sheets."""
        try:
            # Read all sheets from the Excel file
            excel_data = pd.read_excel(excel_file, sheet_name=None)
            return excel_data
        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")

    def clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\-\.,\$\(\)\/\%]', '', text)
        return text.strip()