# src/preprocess.py
# A preprocessing script to extract text from PDFs and chunk it into overlapping, memory-efficient passages,
# with robust handling of Unicode characters during JSON serialization.

import os
import yaml
import json
from PyPDF2 import PdfReader


def load_config():
    """
    Load configuration from config/config.yaml
    """
    with open(os.path.join('config', 'config.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def chunk_pdf(pdf_path: str, chunk_size: int, overlap: int):
    """
    Memory-efficient generator that reads a PDF page-by-page,
    splits into words, and yields text chunks of size `chunk_size`
    with `overlap` words carried over between chunks.
    """
    reader = PdfReader(pdf_path)
    buffer = []
    for page in reader.pages:
        page_text = page.extract_text() or ''
        words = page_text.split()
        for word in words:
            buffer.append(word)
            if len(buffer) >= chunk_size:
                # yield a chunk and keep overlap words for context
                yield ' '.join(buffer)
                buffer = buffer[-overlap:]
    # yield any remaining words as the final chunk
    if buffer:
        yield ' '.join(buffer)


def sanitize_text(text: str) -> str:
    """
    Replace or remove characters that might not serialize cleanly.
    """
    # Replace unencodable characters with the Unicode replacement char
    return text.encode('utf-8', errors='replace').decode('utf-8')


def main():
    cfg = load_config()
    raw_dir = cfg['paths']['data_raw']
    proc_dir = cfg['paths']['data_processed']
    chunk_size = cfg.get('preprocessing', {}).get('chunk_size', 500)
    overlap = cfg.get('preprocessing', {}).get('overlap', 50)

    os.makedirs(proc_dir, exist_ok=True)

    total_chunks = 0
    for filename in os.listdir(raw_dir):
        if not filename.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(raw_dir, filename)
        base_name = os.path.splitext(filename)[0]

        for idx, chunk in enumerate(chunk_pdf(pdf_path, chunk_size, overlap), start=1):
            safe_chunk = sanitize_text(chunk)
            output_path = os.path.join(proc_dir, f"{base_name}_chunk{idx:03d}.json")
            record = {
                "text": safe_chunk,
                "metadata": {
                    "source": filename,
                    "chunk_index": idx
                }
            }
            # Write with error replacement to avoid UnicodeEncodeError
            with open(output_path, 'w', encoding='utf-8', errors='replace') as out_f:
                json.dump(record, out_f, ensure_ascii=False, indent=2)
            total_chunks += 1

    print(f"Created {total_chunks} chunks in '{proc_dir}'")


if __name__ == '__main__':
    main()
