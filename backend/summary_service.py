import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.embedding_models import get_embedding_model
from backend.upload_rag_service import (
    load_uploaded_chunks,
    load_or_build_uploaded_index,
)


load_dotenv()

TEMP_UPLOAD_ROOT = Path("temp_uploads")


class DocumentSummaryService:
    """Generate summaries for uploaded academic document collections."""

    def __init__(self, model_name: str = "gpt-4.1-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY was not found.")

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    
    # LLM helper
    

    def _call_llm(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )

        text = response.output_text.strip()

        if not text:
            raise ValueError(
                "The language model returned an empty summary."
            )

        return text

    
    # Collection metadata
    

    def _load_filenames(self, document_id: str) -> list[str]:
        """Return filenames belonging to an uploaded collection."""

        metadata_path = (
            TEMP_UPLOAD_ROOT
            / document_id
            / "document.json"
        )

        if not metadata_path.exists():
            return []

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        filenames = metadata.get("filenames", [])

        # Support collections created before multi-PDF upload
        # was introduced.
        if not filenames and metadata.get("filename"):
            filenames = [metadata["filename"]]

        return filenames

    
    # Full collection summary
    

    def generate_full_paper_summary(
        self,
        document_id: str,
        summary_length: str = "medium",
    ) -> dict:
        """
        Generate a hierarchical summary of the complete
        uploaded document collection.
        """

        chunks = load_uploaded_chunks(document_id)

        if not chunks:
            raise ValueError(
                "No document chunks were found for this collection."
            )

        filenames = self._load_filenames(document_id)
        document_count = len(filenames)

        length_instruction = self._get_length_instruction(
            summary_length
        )

        # Process a manageable number of chunks at a time.
        group_size = 8

        chunk_groups = [
            chunks[index:index + group_size]
            for index in range(0, len(chunks), group_size)
        ]

        intermediate_summaries = []

        
        # Summarise each group
        

        for group_number, group in enumerate(
            chunk_groups,
            start=1,
        ):
            print(
                f"Summarising section "
                f"{group_number}/{len(chunk_groups)}"
            )

            context_parts = []

            for chunk in group:
                context_parts.append(
                    (
                        f"[Source: {chunk['source']} | "
                        f"Page {chunk['page']}]\n"
                        f"{chunk['text']}"
                    )
                )

            context = "\n\n".join(context_parts)

            prompt = f"""
You are an academic document summarisation assistant.

The supplied text comes from an uploaded collection of academic
documents. A collection may contain one or multiple PDF files.

Create a concise intermediate summary of the supplied sections.

Instructions:
1. Use only information contained in the supplied text.
2. Preserve important aims, arguments, methods, findings,
   limitations and conclusions where available.
3. Do not invent or infer unsupported information.
4. Pay attention to the source filename attached to each section.
5. Do not merge findings from different documents in a way that
   makes them appear to come from the same study.
6. Where sections from different documents discuss related ideas,
   identify their relationship without overstating agreement.
7. Avoid unnecessary citations or references.
8. Do not refer to the entire collection as "this paper" or
   "the paper".
9. Produce a concise intermediate summary.

Document sections:

{context}
"""

            section_summary = self._call_llm(prompt)

            intermediate_summaries.append(
                section_summary
            )

        
        # Combine intermediate summaries
        

        combined_sections = "\n\n".join(
            (
                f"Section {index}:\n{summary}"
            )
            for index, summary in enumerate(
                intermediate_summaries,
                start=1,
            )
        )

        if document_count > 1:
            collection_instruction = f"""
The collection contains {document_count} academic documents.

The uploaded filenames are:
{chr(10).join(f"- {name}" for name in filenames)}

This is a multi-document synthesis.

Important:
- Never describe the collection as "this paper" or "the paper".
- Use wording such as "the uploaded documents",
  "the document collection", "the studies", or the individual
  document name where appropriate.
- Synthesise related findings across documents.
- Preserve meaningful differences between documents.
- Do not imply that findings from different documents belong
  to one study.
- Where the documents agree, explain the shared finding.
- Where they differ, explain the difference when the supplied
  summaries provide enough evidence.
"""
        else:
            collection_instruction = """
The collection contains one academic document.

Summarise the document as a single academic paper while remaining
faithful to the supplied section summaries.
"""

        final_prompt = f"""
You are an academic document summarisation assistant.

Create a coherent overall summary from the section summaries below.

{collection_instruction}

Instructions:
1. Use only information contained in the section summaries.
2. Do not invent information.
3. Combine repeated ideas instead of repeating them.
4. Explain the main aims, approaches, findings, limitations and
   conclusions where these are available.
5. Maintain an academic but clear and readable style.
6. Give appropriate attention to the different documents rather
   than allowing one document to dominate without reason.
7. {length_instruction}

Section summaries:

{combined_sections}
"""

        final_summary = self._call_llm(
            final_prompt
        )

        return {
            "document_id": document_id,
            "filenames": filenames,
            "document_count": document_count,
            "summary_type": "full_paper",
            "summary_length": summary_length,
            "summary": final_summary,
            "chunks_processed": len(chunks),
            "sections_summarised": len(
                intermediate_summaries
            ),
        }

    
    # Topic-focused summary
    

    def generate_topic_focused_summary(
        self,
        document_id: str,
        topic: str,
        embedding_model_name: str,
        top_k: int = 8,
        summary_length: str = "medium",
    ) -> dict:
        """
        Generate a topic-focused summary using relevant chunks
        retrieved from the uploaded document collection.
        """

        topic = topic.strip()

        if not topic:
            raise ValueError("Topic cannot be empty.")

        filenames = self._load_filenames(document_id)
        document_count = len(filenames)

        # Load the selected embedding model.
        embedding_model = get_embedding_model(
            embedding_model_name
        )

        # Load or build the FAISS index for this collection.
        vector_store = load_or_build_uploaded_index(
            document_id=document_id,
            model_name=embedding_model_name,
        )

        # Measure embedding + retrieval time.
        start_time = time.perf_counter()

        query_embedding = embedding_model.encode_query(
            topic
        )

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        retrieval_time = (
            time.perf_counter() - start_time
        )

        if not results:
            raise ValueError(
                "No relevant document sections were found "
                "for this topic."
            )

        
        # Build retrieved context
        

        context_parts = []
        sources = []

        for result in results:
            context_parts.append(
                (
                    f"[Source: {result['source']} | "
                    f"Page {result['page']}]\n"
                    f"{result['text']}"
                )
            )

            sources.append(
                {
                    "rank": result["rank"],
                    "chunk_id": result["chunk_id"],
                    "source": result["source"],
                    "page": result["page"],
                    "score": float(result["score"]),
                }
            )

        context = "\n\n".join(context_parts)

        length_instruction = self._get_length_instruction(
            summary_length
        )

        if document_count > 1:
            collection_instruction = """
The retrieved context may contain evidence from multiple uploaded
academic documents.

When evidence comes from different documents:
- synthesise related information;
- preserve important differences;
- do not make findings from separate studies appear to belong
  to one paper;
- refer to the documents, studies, or source filenames where
  useful;
- never describe the entire collection as "this paper".
"""
        else:
            collection_instruction = """
The retrieved context comes from a single uploaded academic
document.
"""

        
        # Generate topic-focused summary
        

        prompt = f"""
You are an academic document analysis assistant.

Generate a topic-focused academic summary using only the retrieved
sections from the uploaded document collection.

Topic:
{topic}

{collection_instruction}

Instructions:
1. Focus only on information relevant to the requested topic.
2. Use only the supplied retrieved context.
3. Do not add unsupported information.
4. Combine related ideas into a coherent academic summary.
5. Preserve differences between sources when they are relevant.
6. If multiple documents contribute evidence, synthesise them
   rather than presenting them as one paper.
7. If the retrieved context does not adequately discuss the
   requested topic, clearly state this limitation.
8. {length_instruction}

Retrieved document context:

{context}
"""

        summary = self._call_llm(prompt)

        scores = [
            source["score"]
            for source in sources
        ]

        return {
            "document_id": document_id,
            "filenames": filenames,
            "document_count": document_count,
            "summary_type": "topic_focused",
            "embedding_model": embedding_model_name,
            "topic": topic,
            "summary_length": summary_length,
            "summary": summary,
            "retrieval_time": retrieval_time,
            "top_similarity_score": (
                max(scores) if scores else None
            ),
            "average_similarity_score": (
                sum(scores) / len(scores)
                if scores
                else None
            ),
            "sources": sources,
        }

    
    # Summary length
    

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

        return options[summary_length]