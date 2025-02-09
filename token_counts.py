import os
import PyPDF2
import tiktoken

def count_tokens(text):
    # Use the cl100k_base encoding (used by GPT-4)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def get_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def process_pdfs_in_folder(folder_path):
    total_tokens = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = get_pdf_text(pdf_path)
            tokens = count_tokens(text)
            total_tokens += tokens
            print(f"File: {filename}, Tokens: {tokens}")
    
    print(f"Total tokens in all PDFs: {total_tokens}")

if __name__ == "__main__":
    folder_path = "data/"  # Replace with your folder path
    process_pdfs_in_folder(folder_path)