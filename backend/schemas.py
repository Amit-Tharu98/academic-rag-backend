from typing import Literal

from pydantic import BaseModel, Field


EmbeddingModelName = Literal[
    "sentence_transformer",
    "bge",
    "openai",
]


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

class AskQuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
    )

    embedding_model: EmbeddingModelName

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class SourceResult(BaseModel):
    rank: int
    chunk_id: str
    source: str
    page: int
    score: float


class AskQuestionResponse(BaseModel):
    embedding_model: str
    question: str
    answer: str
    retrieval_time: float
    sources: list[SourceResult]
    top_similarity_score: float | None = None
    average_similarity_score: float | None = None


# --------------------------------------------------
# Compare Models
# --------------------------------------------------

class CompareRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class ModelComparisonResult(BaseModel):
    embedding_model: str
    answer: str
    retrieval_time: float
    top_similarity_score: float | None
    average_similarity_score: float | None
    sources: list[SourceResult]


class CompareResponse(BaseModel):
    question: str

    results: list[
        ModelComparisonResult
    ]

    comparison_summary: str

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    chunk_size: int
    chunk_overlap: int
    status: str

class UploadAskRequest(BaseModel):
    document_id: str = Field(
        ...,
        min_length=1,
    )

    question: str = Field(
        ...,
        min_length=1,
    )

    embedding_model: EmbeddingModelName

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class UploadAskResponse(BaseModel):
    document_id: str
    embedding_model: str
    question: str
    answer: str

    retrieval_time: float

    top_similarity_score: float | None
    average_similarity_score: float | None

    sources: list[SourceResult]

class UploadCompareRequest(BaseModel):
    document_id: str = Field(
        ...,
        min_length=1,
    )

    question: str = Field(
        ...,
        min_length=1,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class UploadModelComparisonResult(BaseModel):
    embedding_model: str
    answer: str

    retrieval_time: float

    top_similarity_score: float | None
    average_similarity_score: float | None

    sources: list[SourceResult]


class UploadCompareResponse(BaseModel):
    document_id: str
    question: str

    results: list[
        UploadModelComparisonResult
    ]

    comparison_summary: str

SummaryType = Literal[
    "full_paper",
    "topic_focused",
]

SummaryLength = Literal[
    "short",
    "medium",
    "detailed",
]


class UploadSummaryRequest(BaseModel):
    document_id: str = Field(
        ...,
        min_length=1,
    )

    summary_type: SummaryType

    summary_length: SummaryLength = (
        "medium"
    )

    # Used only for topic-focused summary
    topic: str | None = None

    # Used only for topic-focused summary
    embedding_model: (
        EmbeddingModelName | None
    ) = None

    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
    )


class SummarySourceResult(BaseModel):
    rank: int
    chunk_id: str
    source: str
    page: int
    score: float


class UploadSummaryResponse(BaseModel):
    document_id: str

    summary_type: str
    summary_length: str

    summary: str

    filename: str | None = None

    embedding_model: str | None = None
    topic: str | None = None

    chunks_processed: int | None = None
    sections_summarised: int | None = None

    top_similarity_score: (
        float | None
    ) = None

    average_similarity_score: (
        float | None
    ) = None

    sources: list[
        SummarySourceResult
    ] = []

class UploadSummaryCompareRequest(BaseModel):
    document_id: str = Field(
        ...,
        min_length=1,
    )

    topic: str = Field(
        ...,
        min_length=1,
    )

    summary_length: SummaryLength = "medium"

    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
    )


class UploadSummaryComparisonResult(BaseModel):
    embedding_model: str

    summary: str

    retrieval_time: float

    top_similarity_score: float | None

    average_similarity_score: float | None

    sources: list[SummarySourceResult]


class UploadSummaryCompareResponse(BaseModel):
    document_id: str
    topic: str
    summary_length: str

    results: list[
        UploadSummaryComparisonResult
    ]

    comparison_summary: str

class ResearchCorpusInfo(BaseModel):
    documents: int
    pages: int
    chunks: int
    evaluation_questions: int
    top_k: int


class BestMetricResult(BaseModel):
    model: str
    value: float


class FastestModelResult(BaseModel):
    model: str
    retrieval_time: float


class RetrievalFindings(BaseModel):
    best_top5_accuracy: BestMetricResult
    best_mrr: BestMetricResult
    fastest_model: FastestModelResult


class GenerationFindings(BaseModel):
    best_rouge1: BestMetricResult
    best_rouge2: BestMetricResult
    best_rougeL: BestMetricResult


class ResearchOverviewResponse(BaseModel):
    corpus: ResearchCorpusInfo

    embedding_models: list[str]

    retrieval_findings: RetrievalFindings

    generation_findings: GenerationFindings

    interpretation: str