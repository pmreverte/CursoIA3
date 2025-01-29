import fitz  # PyMuPDF
import tiktoken
"""
Formats the given text by replacing newline characters with spaces and stripping leading/trailing whitespace.
Args:
    text (str): The text to be formatted.
Returns:
    str: The formatted text.
"""
pass
"""
Opens a PDF file, reads its content, and extracts text from each page. The text is then tokenized and word count is calculated.
Args:
    pdf_path (str): The path to the PDF file.
Returns:
    list: A list of dictionaries, each containing information about a page in the PDF, including:
        - file_name (str): The name of the PDF file.
        - page_number (int): The page number.
        - page_word_count (int): The word count of the page.
        - page_token_cont (int): The token count of the page.
        - text (str): The extracted text from the page.
"""
pass
"""
Filters out pages with zero word count from the given list of pages and texts.
Args:
    pages_and_texts_sublist (list): A list of dictionaries, each containing information about a page.
Returns:
    list: A filtered list of dictionaries, each containing information about a page with a word count greater than zero.
"""
pass

def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()

    return cleaned_text

def open_and_read_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages_and_texts = []

    tokenizer = tiktoken.encoding_for_model("gpt-4o")

    for page_number, page in enumerate(doc):
        text = page.get_text()
        text = text_formatter(text)

        tokens = tokenizer.encode(text)
        word_count = len(text.split())

        pages_and_texts.append({
            "file_name": pdf_path,
            "page_number": page_number,
            "page_word_count": word_count,
            "page_token_cont": len(tokens),
            "text": text
        })

    return pages_and_texts

def get_pages_and_texts(pages_and_texts_sublist):
    pages_and_texts = [page for page in pages_and_texts_sublist if page["page_word_count"] > 0]

    return pages_and_texts

"""
Concatenates text documents into groups based on a maximum token size.
Args:
    docs (list of dict): A list of dictionaries where each dictionary represents a document with keys:
        - "file_name" (str): The name of the file the document belongs to.
        - "text" (str): The text content of the document.
        - "token_size" (int): The token size of the document.
    max_tokens (int): The maximum number of tokens allowed in a concatenated group.
Returns:
    list of dict: A list of dictionaries where each dictionary represents a concatenated group of documents with keys:
        - "file_name" (str): The name of the file the group belongs to.
        - "text" (str): The concatenated text content of the group.
        - "token_size" (int): The total token size of the concatenated group.
"""
def concatenate_documents(docs, max_tokens):
    concatenated_docs = []
    current_group = {"file_name": "", "text": "", "token_size": 0}

    for doc in docs:
        if current_group["file_name"] != doc["file_name"]:
            if current_group["token_size"] > 0:
                concatenated_docs.append(current_group)
            current_group = {"file_name": doc["file_name"], "text": doc["text"], "token_size": doc["token_size"]}
        elif current_group["token_size"] + doc["token_size"] <= max_tokens:
            current_group["text"] += (" " + doc["text"]).strip()
            current_group["token_size"] += doc["token_size"]
        else:
            concatenated_docs.append(current_group)
            current_group = {"file_name": doc["file_name"], "text": doc["text"], "token_size": doc["token_size"]}

    if current_group["token_size"] > 0:
        concatenated_docs.append(current_group)
    
    return concatenated_docs
