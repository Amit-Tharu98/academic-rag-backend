from pathlib import Path

from src.embedding_models import get_embedding_model
from src.pdf_processor import extract_pdf_text
from src.text_splitter import split_text
from src.vector_store import FaissVectorStore


def load_all_chunks(
    pdf_folder: str = "data/pdfs",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """
    Process all PDFs and return one combined list of chunks.
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
    print("LOADING AND CHUNKING PDF DOCUMENTS")
    print("=" * 80)

    print(
        f"\nTotal PDFs found: "
        f"{len(pdf_files)}"
    )

    all_chunks = []

    for number, pdf_file in enumerate(
        pdf_files,
        start=1,
    ):

        print(
            f"\n[{number}/{len(pdf_files)}] "
            f"Processing: {pdf_file.name}"
        )

        try:
            pages = extract_pdf_text(
                str(pdf_file)
            )

            chunks = split_text(
                pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            print(
                f"  Pages extracted: "
                f"{len(pages)}"
            )

            print(
                f"  Chunks created: "
                f"{len(chunks)}"
            )

            all_chunks.extend(
                chunks
            )

        except Exception as error:
            raise RuntimeError(
                f"Failed while processing "
                f"{pdf_file.name}: {error}"
            ) from error

    print("\n" + "=" * 80)
    print("CHUNKING COMPLETE")
    print("=" * 80)

    print(
        f"Total chunks: "
        f"{len(all_chunks)}"
    )

    return all_chunks


def validate_chunks(
    chunks: list[dict],
) -> None:
    """
    Check that chunk IDs are unique.
    """

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    unique_chunk_ids = set(
        chunk_ids
    )

    if len(chunk_ids) != len(
        unique_chunk_ids
    ):
        raise ValueError(
            "Duplicate chunk IDs were found."
        )

    print(
        f"Chunk validation successful. "
        f"Unique chunk IDs: "
        f"{len(unique_chunk_ids)}"
    )


def build_model_index(
    model_name: str,
    chunks: list[dict],
) -> None:
    """
    Generate embeddings for all chunks,
    build a FAISS index and save it.
    """

    print("\n" + "=" * 80)
    print(
        f"BUILDING INDEX: "
        f"{model_name.upper()}"
    )
    print("=" * 80)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"\nLoading embedding model: "
        f"{model_name}"
    )

    embedding_model = (
        get_embedding_model(
            model_name
        )
    )

    print(
        f"\nGenerating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = (
        embedding_model.encode_documents(
            texts
        )
    )

    print(
        f"\nEmbeddings generated."
    )

    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    if len(embeddings) != len(
        chunks
    ):
        raise ValueError(
            f"Embedding count mismatch for "
            f"{model_name}. "
            f"Embeddings: {len(embeddings)}, "
            f"Chunks: {len(chunks)}"
        )

    vector_store = (
        FaissVectorStore()
    )

    print(
        "\nBuilding FAISS index..."
    )

    vector_store.build(
        embeddings=embeddings,
        metadata=chunks,
    )

    save_folder = (
        f"indexes/{model_name}"
    )

    vector_store.save(
        save_folder
    )

    print(
        f"\nIndex completed for: "
        f"{model_name}"
    )

    print(
        f"Stored vectors: "
        f"{vector_store.index.ntotal}"
    )


def build_all_indexes() -> None:
    """
    Build FAISS indexes for all
    embedding models.
    """

    models = [
        "sentence_transformer",
        "bge",
        "openai",
    ]

    chunks = load_all_chunks(
        pdf_folder="data/pdfs",
        chunk_size=800,
        chunk_overlap=150,
    )

    validate_chunks(
        chunks
    )

    print("\n" + "=" * 80)
    print("FINAL DATASET INFORMATION")
    print("=" * 80)

    print(
        f"Total chunks to index: "
        f"{len(chunks)}"
    )

    for model_name in models:

        build_model_index(
            model_name=model_name,
            chunks=chunks,
        )

    print("\n" + "=" * 80)
    print("ALL INDEXES BUILT SUCCESSFULLY")
    print("=" * 80)

    for model_name in models:
        print(
            f"indexes/{model_name}"
        )


if __name__ == "__main__":
    build_all_indexes()