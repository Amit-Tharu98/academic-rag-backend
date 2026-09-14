from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    CompareRequest,
    CompareResponse,
    ModelComparisonResult,
    SourceResult,
)

from src.generator import ResponseGenerator
from src.retriever import AcademicRetriever

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
)

from backend.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    CompareRequest,
    CompareResponse,
    ModelComparisonResult,
    SourceResult,
    UploadResponse,
)

from backend.upload_service import (
    create_document_id,
    save_uploaded_pdf,
    process_uploaded_pdf,
)

from backend.schemas import (
    UploadAskRequest,
    UploadAskResponse,
)

from backend.upload_rag_service import (
    ask_uploaded_document,
)
from backend.schemas import (
    UploadCompareRequest,
    UploadCompareResponse,
    UploadModelComparisonResult,
    UploadSummaryRequest,
    UploadSummaryResponse,
    SummarySourceResult,
    UploadSummaryCompareRequest,
    UploadSummaryCompareResponse,
    UploadSummaryComparisonResult,
    ResearchOverviewResponse,
)

from backend.summary_service import (
    DocumentSummaryService,
)

from backend.comparison_service import (
    ComparisonService,
)
from backend.research_results_service import (
    ResearchResultsService,
)
from fastapi.middleware.cors import CORSMiddleware
# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Academic RAG API",
    description=(
        "Retrieval-Augmented Academic Document "
        "Analysis API"
    ),
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "https://academic-rag-frontend.netlify.app",
    ],
    allow_origin_regex=r"https://.*--academic-rag-frontend\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Shared generator
# --------------------------------------------------

generator = ResponseGenerator()

# --------------------------------------------------
# Shared summary service
# --------------------------------------------------

summary_service = DocumentSummaryService()

comparison_service = ComparisonService()
research_results_service = (
    ResearchResultsService()
)

# --------------------------------------------------
# Retriever cache
# --------------------------------------------------

retrievers = {}


def get_retriever(
    model_name: str,
) -> AcademicRetriever:

    if model_name not in retrievers:

        print(
            f"Loading retriever: "
            f"{model_name}"
        )

        retrievers[model_name] = (
            AcademicRetriever(
                model_name=model_name
            )
        )

    return retrievers[model_name]


# --------------------------------------------------
# Helper: convert retrieved chunks into sources
# --------------------------------------------------

def prepare_sources(
    retrieved_chunks: list[dict],
) -> list[SourceResult]:

    sources = []

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        sources.append(
            SourceResult(
                rank=rank,
                chunk_id=
                    chunk["chunk_id"],
                source=
                    chunk["source"],
                page=
                    chunk["page"],
                score=float(
                    chunk["score"]
                ),
            )
        )

    return sources


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message":
            "Academic RAG API is running."
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/api/health")
def health_check():

    return {
        "status": "ok"
    }


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

@app.post(
    "/api/ask",
    response_model=AskQuestionResponse,
)
def ask_question(
    request: AskQuestionRequest,
):

    try:

        question = (
            request.question.strip()
        )

        # Get selected retriever
        retriever = get_retriever(
            request.embedding_model
        )

        # Retrieve chunks
        retrieval_output = (
            retriever.retrieve(
                query=question,
                top_k=request.top_k,
            )
        )

        retrieved_chunks = (
            retrieval_output["results"]
        )

        if not retrieved_chunks:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No relevant document "
                    "chunks were found."
                ),
            )

        # Generate answer
        answer = (
            generator.generate_answer(
                question=question,
                retrieved_chunks=
                    retrieved_chunks,
            )
        )

        # Prepare source data
        sources = prepare_sources(
            retrieved_chunks
        )

        # -----------------------------------------
        # Calculate similarity statistics
        # -----------------------------------------

        scores = [
            float(source.score)
            for source in sources
            if source.score is not None
        ]

        top_similarity_score = (
            max(scores)
            if scores
            else None
        )

        average_similarity_score = (
            sum(scores) / len(scores)
            if scores
            else None
        )

        # -----------------------------------------
        # Return response
        # -----------------------------------------

        return AskQuestionResponse(

            embedding_model=
                request.embedding_model,

            question=
                question,

            answer=
                answer,

            retrieval_time=float(
                retrieval_output[
                    "retrieval_time"
                ]
            ),

            top_similarity_score=
                top_similarity_score,

            average_similarity_score=
                average_similarity_score,

            sources=
                sources,
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# --------------------------------------------------
# Compare all embedding models
# --------------------------------------------------

@app.post(
    "/api/compare",
    response_model=CompareResponse,
)
def compare_models(
    request: CompareRequest,
):

    try:

        question = (
            request.question.strip()
        )

        models = [
            "sentence_transformer",
            "bge",
            "openai",
        ]

        comparison_results = []

        for model_name in models:

            print(
                f"\nComparing model: "
                f"{model_name}"
            )

            # --------------------------------------
            # Load / get cached retriever
            # --------------------------------------

            retriever = get_retriever(
                model_name
            )

            # --------------------------------------
            # Retrieve chunks
            # --------------------------------------

            retrieval_output = (
                retriever.retrieve(
                    query=question,
                    top_k=request.top_k,
                )
            )

            retrieved_chunks = (
                retrieval_output["results"]
            )

            if not retrieved_chunks:
                continue

            # --------------------------------------
            # Generate answer
            # --------------------------------------

            answer = (
                generator.generate_answer(
                    question=question,
                    retrieved_chunks=
                        retrieved_chunks,
                )
            )

            # --------------------------------------
            # Prepare sources
            # --------------------------------------

            sources = prepare_sources(
                retrieved_chunks
            )

            # --------------------------------------
            # Similarity scores
            # --------------------------------------

            scores = [
                float(
                    chunk["score"]
                )
                for chunk
                in retrieved_chunks
            ]

            if scores:

                top_similarity_score = (
                    max(scores)
                )

                average_similarity_score = (
                    sum(scores)
                    / len(scores)
                )

            else:

                top_similarity_score = None
                average_similarity_score = None

            # --------------------------------------
            # Store model result
            # --------------------------------------

            comparison_results.append(
                ModelComparisonResult(
                    embedding_model=
                        model_name,

                    answer=answer,

                    retrieval_time=float(
                        retrieval_output[
                            "retrieval_time"
                        ]
                    ),

                    top_similarity_score=
                        top_similarity_score,

                    average_similarity_score=
                        average_similarity_score,

                    sources=sources,
                )
            )

        if not comparison_results:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No models returned "
                    "retrieval results."
                ),
            )

        comparison_summary = (
            comparison_service
            .generate_qa_comparison(
            question=question,
            results=comparison_results,
            )
        )

        return CompareResponse(
            question=question,
            results=comparison_results,
            comparison_summary=
                comparison_summary,
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

@app.post(
    "/api/upload",
    response_model=UploadResponse,
)
def upload_pdf(
    file: UploadFile = File(...),
):

    try:

        # ------------------------------------------
        # Validate filename
        # ------------------------------------------

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail=(
                    "A PDF file is required."
                ),
            )

        filename = (
            file.filename.strip()
        )

        # ------------------------------------------
        # Validate extension
        # ------------------------------------------

        if not filename.lower().endswith(
            ".pdf"
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Only PDF files are supported."
                ),
            )

        # ------------------------------------------
        # Validate MIME type
        # ------------------------------------------

        if (
            file.content_type
            and file.content_type
            != "application/pdf"
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded file must be "
                    "a PDF document."
                ),
            )

        # ------------------------------------------
        # Generate document ID
        # ------------------------------------------

        document_id = (
            create_document_id()
        )

        # ------------------------------------------
        # Save uploaded PDF
        # ------------------------------------------

        pdf_path = (
            save_uploaded_pdf(
                file_object=file.file,
                filename=filename,
                document_id=
                    document_id,
            )
        )

        # ------------------------------------------
        # Extract + chunk
        # ------------------------------------------

        metadata = (
            process_uploaded_pdf(
                pdf_path=pdf_path,
                original_filename=
                    filename,
                document_id=
                    document_id,

                chunk_size=800,
                chunk_overlap=150,
            )
        )

        # ------------------------------------------
        # Return response
        # ------------------------------------------

        return UploadResponse(
            **metadata
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

# --------------------------------------------------
# Ask Uploaded PDF
# --------------------------------------------------

@app.post(
    "/api/upload/ask",
    response_model=UploadAskResponse,
)
def ask_uploaded_pdf(
    request: UploadAskRequest,
):

    try:

        output = (
            ask_uploaded_document(
                document_id=
                    request.document_id,

                question=
                    request.question,

                model_name=
                    request.embedding_model,

                top_k=
                    request.top_k,
            )
        )

        sources = []

        for item in output["results"]:

            sources.append(
                SourceResult(
                    rank=
                        item["rank"],

                    chunk_id=
                        item["chunk_id"],

                    source=
                        item["source"],

                    page=
                        item["page"],

                    score=
                        float(
                            item["score"]
                        ),
                )
            )

        return UploadAskResponse(
            document_id=output["document_id"],
            embedding_model=output["embedding_model"],
            question=output["question"],
            answer=output["answer"],

            retrieval_time=output[
                "retrieval_time"
            ],

            top_similarity_score=output[
                "top_similarity_score"
            ],

            average_similarity_score=output[
                "average_similarity_score"
            ],

            sources=sources,
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

# --------------------------------------------------
# Compare Models on Uploaded PDF
# --------------------------------------------------

@app.post(
    "/api/upload/compare",
    response_model=UploadCompareResponse,
)
def compare_uploaded_pdf(
    request: UploadCompareRequest,
):

    try:

        question = (
            request.question.strip()
        )

        models = [
            "sentence_transformer",
            "bge",
            "openai",
        ]

        comparison_results = []

        # ------------------------------------------
        # Run same question through all models
        # ------------------------------------------

        for model_name in models:

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"UPLOADED PDF COMPARISON: "
                f"{model_name.upper()}"
            )

            print(
                "=" * 70
            )

            output = (
                ask_uploaded_document(
                    document_id=
                        request.document_id,

                    question=
                        question,

                    model_name=
                        model_name,

                    top_k=
                        request.top_k,
                )
            )

            # --------------------------------------
            # Convert retrieved results to sources
            # --------------------------------------

            sources = []

            for item in output["results"]:

                sources.append(
                    SourceResult(
                        rank=
                            item["rank"],

                        chunk_id=
                            item["chunk_id"],

                        source=
                            item["source"],

                        page=
                            item["page"],

                        score=float(
                            item["score"]
                        ),
                    )
                )

            # --------------------------------------
            # Store model comparison result
            # --------------------------------------

            comparison_results.append(
                UploadModelComparisonResult(
                    embedding_model=
                        output[
                            "embedding_model"
                        ],

                    answer=
                        output[
                            "answer"
                        ],

                    retrieval_time=
                        float(
                            output[
                                "retrieval_time"
                            ]
                        ),

                    top_similarity_score=
                        output[
                            "top_similarity_score"
                        ],

                    average_similarity_score=
                        output[
                            "average_similarity_score"
                        ],

                    sources=
                        sources,
                )
            )

        # ------------------------------------------
        # Return comparison
        # ------------------------------------------

        comparison_summary = (
            comparison_service
            .generate_qa_comparison(
            question=question,
            results=comparison_results,
            )
        )

        return UploadCompareResponse(
            document_id=
                request.document_id,

            question=
                question,

            results=
                comparison_results,

            comparison_summary=
                comparison_summary,
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

# --------------------------------------------------
# Summarise Uploaded PDF
# --------------------------------------------------

@app.post(
    "/api/upload/summary",
    response_model=UploadSummaryResponse,
)
def summarise_uploaded_pdf(
    request: UploadSummaryRequest,
):

    try:

        # ------------------------------------------
        # Full paper
        # ------------------------------------------

        if (
            request.summary_type
            == "full_paper"
        ):

            output = (
                summary_service
                .generate_full_paper_summary(
                    document_id=
                        request.document_id,

                    summary_length=
                        request.summary_length,
                )
            )

            return UploadSummaryResponse(
                document_id=
                    output[
                        "document_id"
                    ],

                filename=
                    output[
                        "filename"
                    ],

                summary_type=
                    output[
                        "summary_type"
                    ],

                summary_length=
                    output[
                        "summary_length"
                    ],

                summary=
                    output[
                        "summary"
                    ],

                chunks_processed=
                    output[
                        "chunks_processed"
                    ],

                sections_summarised=
                    output[
                        "sections_summarised"
                    ],

                sources=[],
            )

        # ------------------------------------------
        # Topic focused
        # ------------------------------------------

        if (
            request.summary_type
            == "topic_focused"
        ):

            if not request.topic:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "topic is required "
                        "for topic_focused "
                        "summaries."
                    ),
                )

            if not request.embedding_model:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "embedding_model is "
                        "required for "
                        "topic_focused summaries."
                    ),
                )

            output = (
                summary_service
                .generate_topic_focused_summary(
                    document_id=
                        request.document_id,

                    topic=
                        request.topic,

                    embedding_model_name=
                        request.embedding_model,

                    top_k=
                        request.top_k,

                    summary_length=
                        request.summary_length,
                )
            )

            sources = [
                SummarySourceResult(
                    **source
                )
                for source
                in output["sources"]
            ]

            return UploadSummaryResponse(
                document_id=
                    output[
                        "document_id"
                    ],

                summary_type=
                    output[
                        "summary_type"
                    ],

                summary_length=
                    output[
                        "summary_length"
                    ],

                embedding_model=
                    output[
                        "embedding_model"
                    ],

                topic=
                    output[
                        "topic"
                    ],

                summary=
                    output[
                        "summary"
                    ],

                top_similarity_score=
                    output[
                        "top_similarity_score"
                    ],

                average_similarity_score=
                    output[
                        "average_similarity_score"
                    ],

                sources=
                    sources,
            )

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported summary type."
            ),
        )

    except HTTPException:
        raise

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

# --------------------------------------------------
# Compare Topic-Focused Summaries
# --------------------------------------------------

@app.post(
    "/api/upload/summary/compare",
    response_model=UploadSummaryCompareResponse,
)
def compare_uploaded_pdf_summaries(
    request: UploadSummaryCompareRequest,
):

    try:

        topic = request.topic.strip()

        models = [
            "sentence_transformer",
            "bge",
            "openai",
        ]

        comparison_results = []

        for model_name in models:

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"TOPIC SUMMARY COMPARISON: "
                f"{model_name.upper()}"
            )

            print(
                "=" * 70
            )

            output = (
                summary_service
                .generate_topic_focused_summary(
                    document_id=
                        request.document_id,

                    topic=
                        topic,

                    embedding_model_name=
                        model_name,

                    top_k=
                        request.top_k,

                    summary_length=
                        request.summary_length,
                )
            )

            sources = [
                SummarySourceResult(
                    rank=
                        source["rank"],

                    chunk_id=
                        source["chunk_id"],

                    source=
                        source["source"],

                    page=
                        source["page"],

                    score=
                        float(
                            source["score"]
                        ),
                )
                for source
                in output["sources"]
            ]

            comparison_results.append(
                UploadSummaryComparisonResult(
                    embedding_model=
                        output[
                            "embedding_model"
                        ],

                    summary=
                        output[
                            "summary"
                        ],

                    retrieval_time=
                        float(
                            output[
                                "retrieval_time"
                            ]
                        ),

                    top_similarity_score=
                        output[
                            "top_similarity_score"
                        ],

                    average_similarity_score=
                        output[
                            "average_similarity_score"
                        ],

                    sources=
                        sources,
                )
            )

        comparison_summary = (
            comparison_service
            .generate_summary_comparison(
            topic=topic,
            results=comparison_results,
            )
        )

        return UploadSummaryCompareResponse(
            document_id=
                request.document_id,

            topic=
                topic,

            summary_length=
                request.summary_length,

            results=
                comparison_results,

            comparison_summary=
                comparison_summary,
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

# --------------------------------------------------
# Research Overview
# --------------------------------------------------

@app.get(
    "/api/research/overview",
    response_model=
        ResearchOverviewResponse,
)
def research_overview():

    try:

        return (
            research_results_service
            .get_overview()
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

# --------------------------------------------------
# Research Retrieval Results
# --------------------------------------------------

@app.get(
    "/api/research/retrieval"
)
def research_retrieval():

    try:

        summary = (
            research_results_service
            .get_retrieval_summary()
        )

        return {
            "metric_type":
                "retrieval",

            "results":
                summary,
        }

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

# --------------------------------------------------
# Research Generation Results
# --------------------------------------------------

@app.get(
    "/api/research/generation"
)
def research_generation():

    try:

        summary = (
            research_results_service
            .get_generation_summary()
        )

        return {
            "metric_type":
                "generation",

            "results":
                summary,
        }

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

@app.get(
    "/api/research/retrieval/details"
)
def research_retrieval_details():

    try:

        return {
            "results":
                research_results_service
                .get_retrieval_results()
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get(
    "/api/research/generation/details"
)
def research_generation_details():

    try:

        return {
            "results":
                research_results_service
                .get_generation_results()
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }