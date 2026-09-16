import re
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean text extracted from PDF pages.
    """

    # Remove null characters
    text = text.replace("\x00", " ")

    # Join words broken across lines by hyphenation
    # Example:
    # "transfor-\nmer" -> "transformer"
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # Replace repeated whitespace/newlines with one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_text(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks = []

    for page in pages:

        cleaned_text = clean_text(
            page["text"]
        )

        source_name = Path(
            page["source"]
        ).stem

        start_position = 0
        page_chunk_number = 0

        while start_position < len(cleaned_text):

            end_position = min(
                start_position + chunk_size,
                len(cleaned_text)
            )

            chunk_text = cleaned_text[
                start_position:end_position
            ].strip()

            if chunk_text:

                chunk_id = (
                    f"{source_name}_"
                    f"P{page['page']:03d}_"
                    f"C{page_chunk_number:03d}"
                )

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": page["source"],
                        "page": page["page"],
                        "text": chunk_text,
                    }
                )

                page_chunk_number += 1

            # Stop when end of page is reached
            if end_position == len(cleaned_text):
                break

            # Create overlap with previous chunk
            start_position = (
                end_position - chunk_overlap
            )

    return chunks