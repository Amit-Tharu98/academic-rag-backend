import os
from abc import ABC, abstractmethod

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


load_dotenv()


class BaseEmbeddingModel(ABC):
    """
    Common interface for all embedding models.
    """

    @abstractmethod
    def encode_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        pass

    @abstractmethod
    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        pass


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Sentence Transformer embedding model.
    """

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    ) -> None:

        print(f"Loading model: {model_name}")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        print("Sentence Transformer loaded successfully.")

    def encode_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:

        if not texts:
            raise ValueError(
                "No document texts were provided."
            )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )

        return embeddings.astype("float32")

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return embedding.astype("float32")


class BGEEmbedding(BaseEmbeddingModel):
    """
    BGE embedding model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
    ) -> None:

        print(f"Loading BGE model: {model_name}")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        print("BGE model loaded successfully.")

    def encode_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:

        if not texts:
            raise ValueError(
                "No document texts were provided."
            )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )

        return embeddings.astype("float32")

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        instructed_query = (
            "Represent this sentence for searching "
            "relevant passages: "
            + query
        )

        embedding = self.model.encode(
            [instructed_query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return embedding.astype("float32")


class OpenAIEmbedding(BaseEmbeddingModel):
    """
    OpenAI text embedding model.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 100,
    ) -> None:

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing from .env"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.model_name = model_name
        self.batch_size = batch_size

        self.client = OpenAI(
            api_key=api_key
        )

        print(
            f"OpenAI embedding model ready: "
            f"{model_name}"
        )

    def encode_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:

        if not texts:
            raise ValueError(
                "No document texts were provided."
            )

        all_embeddings = []

        total_texts = len(texts)

        for start in range(
            0,
            total_texts,
            self.batch_size,
        ):

            end = min(
                start + self.batch_size,
                total_texts,
            )

            batch = texts[start:end]

            print(
                f"Embedding OpenAI batch "
                f"{start + 1}-{end} "
                f"of {total_texts}"
            )

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )

            batch_embeddings = [
                item.embedding
                for item in response.data
            ]

            all_embeddings.extend(
                batch_embeddings
            )

        return np.asarray(
            all_embeddings,
            dtype="float32",
        )

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        response = self.client.embeddings.create(
            model=self.model_name,
            input=query,
        )

        embedding = response.data[0].embedding

        return np.asarray(
            [embedding],
            dtype="float32",
        )


def get_embedding_model(
    model_name: str,
) -> BaseEmbeddingModel:
    """
    Return the requested embedding model.
    """

    if model_name == "sentence_transformer":
        return SentenceTransformerEmbedding()

    elif model_name == "bge":
        return BGEEmbedding()

    elif model_name == "openai":
        return OpenAIEmbedding()

    else:
        raise ValueError(
            f"Unknown embedding model: {model_name}"
        )