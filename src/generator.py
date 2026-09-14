import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class ResponseGenerator:
    """
    Generate answers using retrieved academic document chunks.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY was not found in the .env file."
            )

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> str:
        """
        Generate a grounded answer from retrieved chunks.
        """

        if not question.strip():
            raise ValueError(
                "The question cannot be empty."
            )

        if not retrieved_chunks:
            raise ValueError(
                "No retrieved document chunks were provided."
            )

        context_parts = []

        for chunk in retrieved_chunks:
            source_label = (
                f"Source: {chunk['source']}, "
                f"page {chunk['page']}, "
                f"chunk {chunk['chunk_id']}"
            )

            context_parts.append(
                f"[{source_label}]\n{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are an academic document analysis assistant.

Answer the question using only the supplied document context.

Instructions:
1. Use only information contained in the supplied context.
2. Do not use unsupported outside information.
3. If the context does not contain enough information, clearly state that.
4. Give a clear, concise and factual answer.
5. Do not include source filenames, page numbers or citations in the answer.
6. Do not add unnecessary explanation.

Document context:
{context}

Question:
{question}
"""

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )

        answer = response.output_text.strip()

        if not answer:
            raise ValueError(
                "The language model returned an empty answer."
            )

        return answer