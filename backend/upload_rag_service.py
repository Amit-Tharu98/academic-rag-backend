import json
import time
from pathlib import Path

from src.embedding_models import (
    get_embedding_model,
)
from src.vector_store import (
    FaissVectorStore,
)
from src.generator import (
    ResponseGenerator,
)


TEMP_UPLOAD_ROOT = Path(
    "temp_uploads"
)


def get_document_folder(
    document_id: str,
) -> Path:
    """
    Return folder for an uploaded document.
    """

    return (
        TEMP_UPLOAD_ROOT
        / document_id
    )


def get_chunks_path(
    document_id: str,
) -> Path:
    """
    Return chunks.json path for an uploaded document.
    """

    return (
        get_document_folder(
            document_id
        )
        / "chunks.json"
    )


def get_index_folder(
    document_id: str,
    model_name: str,
) -> Path:
    """
    Return temporary FAISS index folder
    for one document and embedding model.
    """

    return (
        get_document_folder(
            document_id
        )
        / "indexes"
        / model_name
    )


def load_uploaded_chunks(
    document_id: str,
) -> list[dict]:
    """
    Load chunks belonging to an uploaded PDF.
    """

    chunks_path = (
        get_chunks_path(
            document_id
        )
    )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Uploaded document "
            f"was not found: "
            f"{document_id}"
        )

    with open(
        chunks_path,
        "r",
        encoding="utf-8",
    ) as file:

        chunks = json.load(
            file
        )

    if not chunks:
        raise ValueError(
            "The uploaded document "
            "contains no chunks."
        )

    return chunks


def build_uploaded_index(
    document_id: str,
    model_name: str,
    embedding_model=None,
) -> FaissVectorStore:
    """
    Build and save a temporary FAISS index
    for one uploaded PDF and one embedding model.
    """

    chunks = (
        load_uploaded_chunks(
            document_id
        )
    )

    texts = [
        chunk["text"]
        for chunk
        in chunks
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "BUILDING TEMPORARY INDEX"
    )

    print(
        "=" * 60
    )

    print(
        f"Document ID: "
        f"{document_id}"
    )

    print(
        f"Embedding model: "
        f"{model_name}"
    )

    print(
        f"Chunks: "
        f"{len(chunks)}"
    )

    
    # Load embedding model
    
    if embedding_model is None:
        embedding_model = (
            get_embedding_model(
                model_name
            )
        )

    
    # Generate document embeddings
    

    embeddings = (
        embedding_model
        .encode_documents(
            texts
        )
    )

    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    
    # Build FAISS vector store
    

    vector_store = (
        FaissVectorStore()
    )

    vector_store.build(
        embeddings=embeddings,
        metadata=chunks,
    )

    
    # Save temporary index
    

    index_folder = (
        get_index_folder(
            document_id,
            model_name,
        )
    )

    index_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save(
        str(
            index_folder
        )
    )

    print(
        f"Temporary index saved: "
        f"{index_folder}"
    )

    return vector_store


def load_or_build_uploaded_index(
    document_id: str,
    model_name: str,
    embedding_model=None,
) -> FaissVectorStore:
    """
    Load cached temporary index if it already exists.
    Otherwise build it.
    """

    index_folder = (
        get_index_folder(
            document_id,
            model_name,
        )
    )

    index_file = (
        index_folder
        / "index.faiss"
    )

    metadata_file = (
        index_folder
        / "metadata.json"
    )

    
    # Reuse existing index
    

    if (
        index_file.exists()
        and metadata_file.exists()
    ):

        print(
            f"\nReusing cached "
            f"{model_name} index "
            f"for document "
            f"{document_id}"
        )

        vector_store = (
            FaissVectorStore()
        )

        vector_store.load(
            str(
                index_folder
            )
        )

        return vector_store

    
    # Build new index
    

    return build_uploaded_index(
        document_id=
            document_id,

        model_name=
            model_name,
        embedding_model=
            embedding_model,
    )


def ask_uploaded_document(
    document_id: str,
    question: str,
    model_name: str,
    top_k: int = 5,
) -> dict:
    """
    Ask a question against one uploaded PDF.

    The temporary index is built only once
    for the selected embedding model and reused
    for future questions.
    """

    question = (
        question.strip()
    )

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    
    # Confirm document exists
    

    load_uploaded_chunks(
        document_id
    )

    
    # Load selected embedding model
    

    embedding_model = (
        get_embedding_model(
            model_name
        )
    )

    
    # Load/build temporary index
    

    vector_store = (
        load_or_build_uploaded_index(
            document_id=
                document_id,

            model_name=
                model_name,
            embedding_model=
                embedding_model,
        )
    )

    
    # Start retrieval timer
    

    start_time = (
        time.perf_counter()
    )

    
    # Generate query embedding
    

    query_embedding = (
        embedding_model
        .encode_query(
            question
        )
    )

    
    # Search FAISS
    

    results = (
        vector_store.search(
            query_embedding=
                query_embedding,

            top_k=
                top_k,
        )
    )

    retrieval_time = (
        time.perf_counter()
        - start_time
    )

    if not results:
        raise ValueError(
            "No relevant chunks "
            "were found."
        )

    
    # Prepare results
    

    prepared_results = []

    for result in results:

        prepared_results.append(
            {
                "rank":
                    int(
                        result["rank"]
                    ),

                "chunk_id":
                    result[
                        "chunk_id"
                    ],

                "source":
                    result[
                        "source"
                    ],

                "page":
                    int(
                        result[
                            "page"
                        ]
                    ),

                "text":
                    result[
                        "text"
                    ],

                "score":
                    float(
                        result[
                            "score"
                        ]
                    ),
            }
        )

    
    # Generate answer
    

    generator = (
        ResponseGenerator()
    )

    answer = (
        generator
        .generate_answer(
            question=
                question,

            retrieved_chunks=
                prepared_results,
        )
    )

    
    # Calculate score summary
    

    similarity_scores = [
        item["score"]
        for item
        in prepared_results
    ]

    top_similarity_score = (
        max(
            similarity_scores
        )
        if similarity_scores
        else None
    )

    average_similarity_score = (
        sum(
            similarity_scores
        )
        / len(
            similarity_scores
        )
        if similarity_scores
        else None
    )

    
    # Return result
    

    return {
        "document_id":
            document_id,

        "embedding_model":
            model_name,

        "question":
            question,

        "answer":
            answer,

        "retrieval_time":
            retrieval_time,

        "top_similarity_score":
            top_similarity_score,

        "average_similarity_score":
            average_similarity_score,

        "results":
            prepared_results,
    }