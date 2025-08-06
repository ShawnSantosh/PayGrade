import pypdf

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    if pdf_file:
        try:
            pdf_reader = pypdf.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"
    return None