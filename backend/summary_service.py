import json
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
import os

from src.embedding_models import get_embedding_model
from backend.upload_rag_service import (
    load_uploaded_chunks,
    load_or_build_uploaded_index,
)
import time

load_dotenv()


TEMP_UPLOAD_ROOT = Path("temp_uploads")


class DocumentSummaryService:
    """
    Generate full-paper or topic-focused summaries
    for user-uploaded academic PDFs.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
    ) -> None:

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY was not found."
            )

        self.model_name = model_name

        self.client = OpenAI(
            api_key=api_key
        )

    # --------------------------------------------------
    # LLM helper
    # --------------------------------------------------

    def _call_llm(
        self,
        prompt: str,
    ) -> str:

        response = (
            self.client.responses.create(
                model=self.model_name,
                input=prompt,
            )
        )

        text = (
            response.output_text.strip()
        )

        if not text:
            raise ValueError(
                "The language model returned "
                "an empty summary."
            )

        return text

    # --------------------------------------------------
    # Full paper summary
    # --------------------------------------------------

    def generate_full_paper_summary(
        self,
        document_id: str,
        summary_length: str = "medium",
    ) -> dict:
        """
        Generate a hierarchical summary of
        the complete uploaded document.
        """

        chunks = load_uploaded_chunks(
            document_id
        )

        length_instruction = (
            self._get_length_instruction(
                summary_length
            )
        )

        # ------------------------------------------
        # Group chunks
        # ------------------------------------------

        group_size = 8

        chunk_groups = [
            chunks[index:index + group_size]
            for index in range(
                0,
                len(chunks),
                group_size,
            )
        ]

        intermediate_summaries = []

        # ------------------------------------------
        # Summarise each group
        # ------------------------------------------

        for group_number, group in enumerate(
            chunk_groups,
            start=1,
        ):

            print(
                f"Summarising section "
                f"{group_number}/"
                f"{len(chunk_groups)}"
            )

            context_parts = []

            for chunk in group:

                context_parts.append(
                    (
                        f"[Page {chunk['page']}]\n"
                        f"{chunk['text']}"
                    )
                )

            context = "\n\n".join(
                context_parts
            )

            prompt = f"""
You are an academic document summarisation assistant.

Summarise the following section of an academic paper.

Instructions:
1. Use only information contained in the supplied text.
2. Preserve important arguments, methods, findings and conclusions.
3. Do not invent information.
4. Do not include citations unless they are necessary to understand the content.
5. Produce a concise intermediate summary.

Document section:

{context}
"""

            section_summary = (
                self._call_llm(
                    prompt
                )
            )

            intermediate_summaries.append(
                section_summary
            )

        # ------------------------------------------
        # Combine intermediate summaries
        # ------------------------------------------

        combined_sections = "\n\n".join(
            [
                (
                    f"Section {index}:\n"
                    f"{summary}"
                )
                for index, summary
                in enumerate(
                    intermediate_summaries,
                    start=1,
                )
            ]
        )

        final_prompt = f"""
You are an academic document summarisation assistant.

Create a coherent overall summary of the academic paper
using the section summaries below.

Instructions:
1. Use only information contained in the section summaries.
2. Do not invent information.
3. Combine repeated ideas rather than repeating them.
4. Clearly explain the paper's main purpose, approach,
   key findings and conclusions where these are available.
5. Maintain an academic but readable style.
6. {length_instruction}

Section summaries:

{combined_sections}
"""

        final_summary = (
            self._call_llm(
                final_prompt
            )
        )

        # ------------------------------------------
        # Load document metadata
        # ------------------------------------------

        metadata_path = (
            TEMP_UPLOAD_ROOT
            / document_id
            / "document.json"
        )

        filename = ""

        if metadata_path.exists():

            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

            filename = metadata.get(
                "filename",
                "",
            )

        return {
            "document_id":
                document_id,

            "filename":
                filename,

            "summary_type":
                "full_paper",

            "summary_length":
                summary_length,

            "summary":
                final_summary,

            "chunks_processed":
                len(chunks),

            "sections_summarised":
                len(
                    intermediate_summaries
                ),
        }

    # --------------------------------------------------
    # Topic-focused summary
    # --------------------------------------------------

    def generate_topic_focused_summary(
        self,
        document_id: str,
        topic: str,
        embedding_model_name: str,
        top_k: int = 8,
        summary_length: str = "medium",
    ) -> dict:
        """
        Generate a summary focused on a particular
        topic using retrieved chunks.
        """

        topic = topic.strip()

        if not topic:
            raise ValueError(
                "Topic cannot be empty."
            )

        # ------------------------------------------
        # Load embedding model
        # ------------------------------------------

        embedding_model = (
            get_embedding_model(
                embedding_model_name
            )
        )

        # ------------------------------------------
        # Load/build document index
        # ------------------------------------------

        vector_store = (
            load_or_build_uploaded_index(
                document_id=
                    document_id,

                model_name=
                    embedding_model_name,
            )
        )

        # ------------------------------------------
        # Embed topic
        # ------------------------------------------
        start_time = time.perf_counter()
        query_embedding = (
            embedding_model.encode_query(
                topic
            )
        )
      
        # ------------------------------------------
        # Retrieve relevant chunks
        # ------------------------------------------

        results = (
            vector_store.search(
                query_embedding=
                    query_embedding,

                top_k=
                    top_k,
            )
        )
        # Stop timer AFTER FAISS search
        retrieval_time = (
            time.perf_counter()
            - start_time
        )

        if not results:
            raise ValueError(
                "No relevant document sections "
                "were found for this topic."
            )

        # ------------------------------------------
        # Build context
        # ------------------------------------------

        context_parts = []

        sources = []

        for result in results:

            context_parts.append(
                (
                    f"[Page {result['page']}]\n"
                    f"{result['text']}"
                )
            )

            sources.append(
                {
                    "rank":
                        result["rank"],

                    "chunk_id":
                        result["chunk_id"],

                    "source":
                        result["source"],

                    "page":
                        result["page"],

                    "score":
                        float(
                            result["score"]
                        ),
                }
            )

        context = "\n\n".join(
            context_parts
        )

        length_instruction = (
            self._get_length_instruction(
                summary_length
            )
        )

        # ------------------------------------------
        # Generate focused summary
        # ------------------------------------------

        prompt = f"""
You are an academic document analysis assistant.

Generate a topic-focused summary using only the
retrieved sections of the uploaded academic paper.

Topic:
{topic}

Instructions:
1. Focus only on information relevant to the requested topic.
2. Use only the supplied context.
3. Do not add unsupported information.
4. Combine related ideas into a coherent academic summary.
5. If the retrieved context does not adequately discuss the
   requested topic, clearly state that.
6. {length_instruction}

Retrieved document context:

{context}
"""

        summary = (
            self._call_llm(
                prompt
            )
        )

        scores = [
            source["score"]
            for source in sources
        ]

        return {
            "document_id":
                document_id,

            "summary_type":
                "topic_focused",

            "embedding_model":
                embedding_model_name,

            "topic":
                topic,

            "summary_length":
                summary_length,

            "summary":
                summary,
            
            "retrieval_time":
                retrieval_time,

            "top_similarity_score":
                (
                    max(scores)
                    if scores
                    else None
                ),

            "average_similarity_score":
                (
                    sum(scores) / len(scores)
                    if scores
                    else None
                ),

            "sources":
                sources,
        }

    # --------------------------------------------------
    # Length instructions
    # --------------------------------------------------

    def _get_length_instruction(
        self,
        summary_length: str,
    ) -> str:

        options = {
            "short": (
                "Keep the final summary concise, "
                "approximately 100-150 words."
            ),

            "medium": (
                "Produce a moderately detailed summary, "
                "approximately 250-350 words."
            ),

            "detailed": (
                "Produce a detailed summary, "
                "approximately 500-700 words."
            ),
        }

        if summary_length not in options:

            raise ValueError(
                "summary_length must be one of: "
                "short, medium, detailed."
            )

        return options[
            summary_length
        ]