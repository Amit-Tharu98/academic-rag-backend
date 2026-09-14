from pathlib import Path

import fitz

from src.text_splitter import split_text


def extract_pdf_text(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF one page at a time.

    Returns:
        A list of dictionaries containing:
        - source file name
        - page number
        - extracted text
    """

    file_path = Path(pdf_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            "The selected file is not a PDF."
        )

    extracted_pages: list[dict] = []

    try:
        document = fitz.open(file_path)

        for page_index, page in enumerate(document):

            page_text = page.get_text(
                "text"
            ).strip()

            if page_text:
                extracted_pages.append(
                    {
                        "source": file_path.name,
                        "page": page_index + 1,
                        "text": page_text,
                    }
                )

        document.close()

    except Exception as error:
        raise RuntimeError(
            f"Failed to extract text from "
            f"{file_path.name}: {error}"
        ) from error

    return extracted_pages


def process_all_pdfs(
    pdf_folder="data/pdfs",
    chunk_size=800,
    chunk_overlap=150,
):
    """
    Process all PDF files inside a folder
    and return all generated chunks.
    """

    folder = Path(pdf_folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {folder}"
        )

    pdf_files = sorted(
        folder.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {folder}"
        )

    print("=" * 80)
    print("PROCESSING PDF DOCUMENTS")
    print("=" * 80)

    print(
        f"\nTotal PDFs found: {len(pdf_files)}"
    )

    all_chunks = []
    total_pages = 0
    successful_documents = 0

    for number, pdf_file in enumerate(
        pdf_files,
        start=1,
    ):

        print(
            f"\n[{number}/{len(pdf_files)}] "
            f"Processing: {pdf_file.name}"
        )

        try:

            # Extract pages
            pages = extract_pdf_text(
                str(pdf_file)
            )

            total_pages += len(pages)

            print(
                f"  Pages extracted: {len(pages)}"
            )

            # Create chunks
            chunks = split_text(
                pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            print(
                f"  Chunks created: {len(chunks)}"
            )

            all_chunks.extend(chunks)

            successful_documents += 1

        except Exception as error:

            print(
                f"  ERROR: {error}"
            )

    print("\n" + "=" * 80)
    print("PDF PROCESSING COMPLETE")
    print("=" * 80)

    print(
        f"Documents processed: "
        f"{successful_documents}/{len(pdf_files)}"
    )

    print(
        f"Total pages extracted: {total_pages}"
    )

    print(
        f"Total chunks created: {len(all_chunks)}"
    )

    return all_chunks


if __name__ == "__main__":

    try:

        chunks = process_all_pdfs(
            pdf_folder="data/pdfs",
            chunk_size=800,
            chunk_overlap=150,
        )

        if chunks:

            print("\nExample first chunk:")
            print("-" * 60)

            print(
                f"Chunk ID: "
                f"{chunks[0]['chunk_id']}"
            )

            print(
                f"Source: "
                f"{chunks[0]['source']}"
            )

            print(
                f"Page: "
                f"{chunks[0]['page']}"
            )

            print(
                f"Characters: "
                f"{len(chunks[0]['text'])}"
            )

            print()

            print(
                chunks[0]["text"][:500]
            )

            print("-" * 60)

    except Exception as error:

        print(
            f"Error: {error}"
        )