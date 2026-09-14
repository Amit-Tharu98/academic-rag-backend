from dotenv import load_dotenv
from openai import OpenAI
import os


load_dotenv()


class ComparisonService:
    """
    Generate qualitative comparisons between outputs
    produced using different embedding models.
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

        result = (
            response.output_text.strip()
        )

        if not result:
            raise ValueError(
                "The comparison model returned "
                "an empty response."
            )

        return result

    def generate_qa_comparison(
        self,
        question: str,
        results: list,
    ) -> str:
        """
        Compare QA outputs from different embedding models.
        """

        model_sections = []

        for result in results:

            model_name = (
                result.embedding_model
                if hasattr(
                    result,
                    "embedding_model"
                )
                else result[
                    "embedding_model"
                ]
            )

            answer = (
                result.answer
                if hasattr(
                    result,
                    "answer"
                )
                else result["answer"]
            )

            retrieval_time = (
                result.retrieval_time
                if hasattr(
                    result,
                    "retrieval_time"
                )
                else result[
                    "retrieval_time"
                ]
            )

            sources = (
                result.sources
                if hasattr(
                    result,
                    "sources"
                )
                else result["sources"]
            )

            source_lines = []

            for source in sources:

                page = (
                    source.page
                    if hasattr(
                        source,
                        "page"
                    )
                    else source["page"]
                )

                source_name = (
                    source.source
                    if hasattr(
                        source,
                        "source"
                    )
                    else source["source"]
                )

                source_lines.append(
                    f"{source_name}, page {page}"
                )

            source_text = (
                "; ".join(source_lines)
            )

            model_sections.append(
                f"""
Embedding model: {model_name}

Answer:
{answer}

Retrieval time:
{retrieval_time:.4f} seconds

Retrieved sources:
{source_text}
"""
            )

        combined = "\n".join(
            model_sections
        )

        prompt = f"""
You are comparing outputs from three retrieval-augmented
generation systems that use different embedding models.

User question:
{question}

Results:

{combined}

Write a concise comparison of the three systems.

Instructions:
1. Compare the content and coverage of the three answers.
2. Identify important similarities and differences.
3. Comment on differences in retrieved sources or pages.
4. Compare retrieval speed where useful.
5. Do not claim that one answer is factually correct unless
   the supplied evidence clearly supports that conclusion.
6. Do not rank models using raw similarity scores because
   similarity values are not directly calibrated across
   different embedding spaces.
7. Do not invent evaluation metrics.
8. Do not use the dissertation's experimental results here.
9. Keep the comparison approximately 120-180 words.
10. Use clear academic language.
"""

        return self._call_llm(
            prompt
        )

    def generate_summary_comparison(
        self,
        topic: str,
        results: list,
    ) -> str:
        """
        Compare topic-focused summaries produced using
        different embedding models.
        """

        model_sections = []

        for result in results:

            model_name = (
                result.embedding_model
                if hasattr(
                    result,
                    "embedding_model"
                )
                else result[
                    "embedding_model"
                ]
            )

            summary = (
                result.summary
                if hasattr(
                    result,
                    "summary"
                )
                else result["summary"]
            )

            retrieval_time = (
                result.retrieval_time
                if hasattr(
                    result,
                    "retrieval_time"
                )
                else result[
                    "retrieval_time"
                ]
            )

            sources = (
                result.sources
                if hasattr(
                    result,
                    "sources"
                )
                else result["sources"]
            )

            page_numbers = []

            for source in sources:

                page = (
                    source.page
                    if hasattr(
                        source,
                        "page"
                    )
                    else source["page"]
                )

                page_numbers.append(
                    str(page)
                )

            pages = ", ".join(
                page_numbers
            )

            model_sections.append(
                f"""
Embedding model: {model_name}

Topic-focused summary:
{summary}

Retrieval time:
{retrieval_time:.4f} seconds

Retrieved pages:
{pages}
"""
            )

        combined = "\n".join(
            model_sections
        )

        prompt = f"""
You are comparing topic-focused summaries generated from
the same academic PDF using three different embedding models.

Requested topic:
{topic}

Results:

{combined}

Write a concise comparison.

Instructions:
1. Compare the information covered by each summary.
2. Identify important points shared across the summaries.
3. Identify meaningful differences or omissions.
4. Comment on differences in retrieved pages where relevant.
5. Compare retrieval speed where useful.
6. Do not rank models using raw similarity scores.
7. Do not claim factual superiority without appropriate evidence.
8. Do not invent evaluation scores.
9. Keep the comparison approximately 120-180 words.
10. Use clear academic language.
"""

        return self._call_llm(
            prompt
        )