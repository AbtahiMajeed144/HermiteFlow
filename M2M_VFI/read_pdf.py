import sys
import io

try:
    import fitz  # PyMuPDF
    doc = fitz.open(sys.argv[1])
    text = ""
    for page in doc:
        text += page.get_text()
except ImportError:
    from PyPDF2 import PdfReader
    reader = PdfReader(sys.argv[1])
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

with io.open('paper_text_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(text)
