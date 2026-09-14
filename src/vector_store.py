from pathlib import Path
import json

import faiss
import numpy as np


class FaissVectorStore:
    """
    Store document embeddings and perform similarity search using FAISS.

    All document and query embeddings are L2-normalized here so that
    inner-product search behaves like cosine similarity.
    """

    def __init__(self) -> None:
        self.index = None
        self.metadata: list[dict] = []

    def build(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """
        Build a FAISS index from document embeddings.

        Embeddings are centrally L2-normalized before being added
        to an IndexFlatIP index.
        """

        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Document embeddings must be a 2D array."
            )

        if len(vectors) != len(metadata):
            raise ValueError(
                "The number of embeddings must match "
                "the number of metadata records."
            )

        if len(vectors) == 0:
            raise ValueError(
                "No embeddings were provided."
            )

        # Copy before normalization so the original
        # embedding array is not modified.
        vectors = vectors.copy()

        # Centralized L2 normalization.
        # Inner product then behaves like cosine similarity.
        faiss.normalize_L2(vectors)

        embedding_dimension = (
            vectors.shape[1]
        )

        self.index = faiss.IndexFlatIP(
            embedding_dimension
        )

        self.index.add(
            vectors
        )

        self.metadata = (
            metadata.copy()
        )

        print(
            f"FAISS index built successfully. "
            f"Stored vectors: "
            f"{self.index.ntotal}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search for the most semantically similar chunks.

        Query embeddings are L2-normalized here before search.
        """

        if self.index is None:
            raise RuntimeError(
                "The FAISS index has not been built."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_vector = np.asarray(
            query_embedding,
            dtype="float32",
        )

        if query_vector.ndim == 1:
            query_vector = (
                query_vector.reshape(
                    1,
                    -1,
                )
            )

        if query_vector.ndim != 2:
            raise ValueError(
                "Query embedding must be "
                "a 1D or 2D array."
            )

        if (
            query_vector.shape[1]
            != self.index.d
        ):
            raise ValueError(
                "Query embedding dimension "
                "does not match the "
                "FAISS index dimension."
            )

        # Copy before normalization so original
        # query embedding is not modified.
        query_vector = (
            query_vector.copy()
        )

        # Centralized query normalization.
        faiss.normalize_L2(
            query_vector
        )

        number_of_results = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = (
            self.index.search(
                query_vector,
                number_of_results,
            )
        )

        results: list[dict] = []

        for rank, (
            index_position,
            score,
        ) in enumerate(
            zip(
                indices[0],
                scores[0],
            ),
            start=1,
        ):

            if index_position == -1:
                continue

            result = (
                self.metadata[
                    index_position
                ].copy()
            )

            result["rank"] = rank

            # Standard field name used everywhere
            result["score"] = float(
                score
            )

            results.append(
                result
            )

        return results

    def save(
        self,
        folder_path: str,
    ) -> None:
        """
        Save the FAISS index and metadata to disk.
        """

        if self.index is None:
            raise RuntimeError(
                "There is no FAISS index to save."
            )

        folder = Path(
            folder_path
        )

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(
                folder
                / "index.faiss"
            ),
        )

        with open(
            folder
            / "metadata.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"FAISS index saved to: "
            f"{folder}"
        )

    def load(
        self,
        folder_path: str,
    ) -> None:
        """
        Load a saved FAISS index and metadata.
        """

        folder = Path(
            folder_path
        )

        index_path = (
            folder
            / "index.faiss"
        )

        metadata_path = (
            folder
            / "metadata.json"
        )

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{metadata_path}"
            )

        self.index = (
            faiss.read_index(
                str(index_path)
            )
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.metadata = (
                json.load(
                    file
                )
            )

        if (
            self.index.ntotal
            != len(
                self.metadata
            )
        ):
            raise ValueError(
                "Loaded FAISS index "
                "and metadata contain "
                "different numbers "
                "of records."
            )

        print(
            f"FAISS index loaded successfully. "
            f"Stored vectors: "
            f"{self.index.ntotal}"
        )