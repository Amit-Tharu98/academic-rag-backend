import json
import shutil
import uuid
from pathlib import Path

from src.pdf_processor import extract_pdf_text
from src.text_splitter import split_text


TEMP_UPLOAD_ROOT = Path("temp_uploads")


def create_document_id() -> str:
    """
    Generate a unique ID for one uploaded document collection.
    """
    return uuid.uuid4().hex


def get_document_folder(
    document_id: str,
) -> Path:
    """
    Return the temporary folder for an uploaded collection.
    """
    return TEMP_UPLOAD_ROOT / document_id


def _safe_filename(filename: str) -> str:
    """
    Keep only a safe basename for local temporary storage.
    """
    clean_name = Path(filename).name.strip()

    if not clean_name:
        return "document.pdf"

    return clean_name


def _unique_pdf_path(
    folder: Path,
    filename: str,
) -> Path:
    """
    Return a non-conflicting path when two uploaded PDFs have
    the same filename.
    """
    safe_name = _safe_filename(filename)
    candidate = folder / safe_name

    if not candidate.exists():
        return candidate

    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix or ".pdf"
    counter = 2

    while True:
        candidate = folder / f"{stem}_{counter}{suffix}"

        if not candidate.exists():
            return candidate

        counter += 1


def save_uploaded_pdf(
    file_object,
    filename: str,
    document_id: str,
) -> Path:
    """
    Save one uploaded PDF inside the collection folder.

    Each original PDF is preserved instead of every upload being
    written to the same ``original.pdf`` path.
    """
    document_folder = get_document_folder(document_id)

    files_folder = document_folder / "files"
    files_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = _unique_pdf_path(
        files_folder,
        filename,
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
    Extract text from one PDF and create chunks.

    The function keeps the existing chunks.json/document.json output
    for compatibility with the current upload endpoint. The endpoint
    combines the per-file chunks into one collection after each PDF
    has been processed.
    """
    pages = extract_pdf_text(pdf_path)

    if not pages:
        raise ValueError(
            f"No readable text was found in {original_filename}."
        )

    for page in pages:
        page["source"] = original_filename

    chunks = split_text(
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(
            f"No chunks could be created from {original_filename}."
        )

    updated_chunks = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        updated_chunks.append(
            {
                "chunk_id": (
                    f"{document_id}_"
                    f"{Path(original_filename).stem}_"
                    f"C{index:04d}"
                ),
                "source": original_filename,
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )

    document_folder = get_document_folder(document_id)
    document_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # These files are intentionally overwritten for each individual
    # PDF. main.py reads the per-file chunks immediately and then writes
    # the final combined collection versions after processing all PDFs.
    chunks_path = document_folder / "chunks.json"

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

    metadata = {
        "document_id": document_id,
        "filename": original_filename,
        "pages": len(pages),
        "chunks": len(updated_chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "status": "processed",
    }

    metadata_path = document_folder / "document.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return metadata
