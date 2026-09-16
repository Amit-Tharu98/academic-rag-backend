import time

from src.embedding_models import get_embedding_model
from src.vector_store import FaissVectorStore


class AcademicRetriever:
    """
    Retrieve relevant academic document chunks
    using a selected embedding model.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:

        self.model_name = model_name

        print(
            f"\nLoading embedding model: "
            f"{model_name}"
        )

        # Load embedding model
        self.embedding_model = (
            get_embedding_model(
                model_name
            )
        )

        print(
            f"Loading FAISS index for: "
            f"{model_name}"
        )

        # Load corresponding FAISS index
        self.vector_store = (
            FaissVectorStore()
        )

        self.vector_store.load(
            f"indexes/{model_name}"
        )

        print(
            f"Retriever ready. "
            f"Stored vectors: "
            f"{self.vector_store.index.ntotal}"
        )

    
    # Normalise individual search result
    

    def _prepare_result(
        self,
        result: dict,
        rank: int,
    ) -> dict:
        """
        Convert a vector-store result into a
        consistent format expected by the API.
        """

        
        # Find similarity score
        

        score = None

        possible_score_keys = [
            "score",
            "similarity_score",
            "similarity",
            "distance",
        ]

        for key in possible_score_keys:

            if key in result:

                score = float(
                    result[key]
                )

                break

        if score is None:

            raise KeyError(
                "The vector store search result "
                "does not contain a similarity score. "
                "Expected one of: "
                "'score', 'similarity_score', "
                "'similarity', or 'distance'."
            )

        
        # Validate required metadata
        

        required_fields = [
            "chunk_id",
            "source",
            "page",
            "text",
        ]

        for field in required_fields:

            if field not in result:

                raise KeyError(
                    f"Search result is missing "
                    f"required field: {field}"
                )

        
        # Standard result format
        

        return {
            "rank": rank,
            "chunk_id":
                result["chunk_id"],

            "source":
                result["source"],

            "page":
                result["page"],

            "text":
                result["text"],

            "score":
                score,
        }

    
    # Retrieve
    

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict:
        """
        Retrieve top-k relevant chunks.

        Returns:
            {
                "model": ...,
                "retrieval_time": ...,
                "results": [...]
            }
        """

        
        # Validation
        

        if not query.strip():

            raise ValueError(
                "The query cannot be empty."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )

        
        # Start timing
        

        start_time = (
            time.perf_counter()
        )

        
        # Generate query embedding
        

        query_embedding = (
            self.embedding_model
            .encode_query(
                query
            )
        )

        
        # Search FAISS
        

        raw_results = (
            self.vector_store.search(
                query_embedding=
                    query_embedding,

                top_k=top_k,
            )
        )

        
        # Standardise search results
        

        results = []

        for rank, result in enumerate(
            raw_results,
            start=1,
        ):

            prepared_result = (
                self._prepare_result(
                    result=result,
                    rank=rank,
                )
            )

            results.append(
                prepared_result
            )

        
        # End timing
        

        end_time = (
            time.perf_counter()
        )

        retrieval_time = (
            end_time - start_time
        )

        
        # Return retrieval information
        

        return {
            "model":
                self.model_name,

            "retrieval_time":
                retrieval_time,

            "results":
                results,
        }
        