import tiktoken

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into chunks."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks

def split_pages_into_chunks(pages: list, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split pages into chunks while keeping page number info.
    Returns list of dicts with text, page number, heading.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        tokens = enc.encode(text)

        # Extract first line as heading
        first_line = text.strip().split("\n")[0][:60]

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)

            chunks.append({
                "text": chunk_text,
                "page": page_num,
                "heading": first_line
            })
            start += chunk_size - overlap

    return chunks