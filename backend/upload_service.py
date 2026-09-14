import json
import shutil
import uuid
from pathlib import Path

from src.pdf_processor import extract_pdf_text
from src.text_splitter import split_text


TEMP_UPLOAD_ROOT = Path("temp_uploads")


def create_document_id() -> str:
    """
    Generate a unique ID for one uploaded document.
    """
    return uuid.uuid4().hex


def get_document_folder(
    document_id: str,
) -> Path:
    """
    Return the temporary folder for a document.
    """
    return TEMP_UPLOAD_ROOT / document_id


def save_uploaded_pdf(
    file_object,
    filename: str,
    document_id: str,
) -> Path:
    """
    Save uploaded PDF into its temporary document folder.
    """

    document_folder = get_document_folder(
        document_id
    )

    document_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        document_folder
        / "original.pdf"
    )

    with open(
        pdf_path,
        "wb",
    ) as output_file:

        shutil.copyfileobj(
            file_object,
            output_file,
        )

    return pdf_path


def process_uploaded_pdf(
    pdf_path: Path,
    original_filename: str,
    document_id: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> dict:
    """
    Extract PDF pages, create chunks,
    and save chunk metadata.
    """

    # ----------------------------------------------
    # Extract pages
    # ----------------------------------------------

    pages = extract_pdf_text(
        pdf_path
    )

    if not pages:
        raise ValueError(
            "No readable text was found in the PDF."
        )

    # ----------------------------------------------
    # Ensure uploaded PDF uses original filename
    # ----------------------------------------------

    for page in pages:
        page["source"] = (
            original_filename
        )

    # ----------------------------------------------
    # Chunk document
    # ----------------------------------------------

    chunks = split_text(
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(
            "No chunks could be created from the PDF."
        )

    # ----------------------------------------------
    # Make uploaded chunk IDs session-specific
    # ----------------------------------------------

    updated_chunks = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        updated_chunk = {
            "chunk_id": (
                f"{document_id}_C{index:04d}"
            ),
            "source":
                original_filename,
            "page":
                chunk["page"],
            "text":
                chunk["text"],
        }

        updated_chunks.append(
            updated_chunk
        )

    # ----------------------------------------------
    # Save chunks
    # ----------------------------------------------

    document_folder = (
        get_document_folder(
            document_id
        )
    )

    chunks_path = (
        document_folder
        / "chunks.json"
    )

    with open(
        chunks_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            updated_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ----------------------------------------------
    # Save document metadata
    # ----------------------------------------------

    metadata = {
        "document_id":
            document_id,

        "filename":
            original_filename,

        "pages":
            len(pages),

        "chunks":
            len(updated_chunks),

        "chunk_size":
            chunk_size,

        "chunk_overlap":
            chunk_overlap,

        "status":
            "processed",
    }

    metadata_path = (
        document_folder
        / "document.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    return metadata