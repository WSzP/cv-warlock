"""PDF text extraction utility."""

from typing import BinaryIO


def extract_text_from_pdf(file: BinaryIO) -> tuple[str | None, str | None]:
    """Extract text content from a PDF file.

    Args:
        file: File-like object containing PDF data.

    Returns:
        Tuple of (extracted_text, error_message). One will be None.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, "PDF parsing library not installed. Run: uv add pymupdf"

    try:
        # Read PDF content
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Extract text from all pages
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text)

        doc.close()

        if not text_parts:
            return None, "No text content found in PDF"

        # Join pages with double newlines
        full_text = "\n\n".join(text_parts)

        # Clean up the text
        full_text = clean_extracted_text(full_text)

        if len(full_text) < 50:
            return None, "Extracted text is too short. The PDF may be image-based."

        return full_text, None

    except Exception as e:
        return None, f"Error reading PDF: {str(e)}"


def clean_extracted_text(text: str) -> str:
    """Clean up extracted PDF text.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    import re

    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove page headers/footers that are just page numbers
    text = re.sub(r'\n\d+\n', '\n', text)

    # Clean up lines
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Skip very short lines that are likely artifacts
        if len(line) < 2 and not line.isalnum():
            continue
        lines.append(line)

    return '\n'.join(lines).strip()
