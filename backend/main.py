import json
from pathlib import Path
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

# FastAPI application


app = FastAPI(
    title="Academic RAG API",
    description=(
        "Retrieval-Augmented Academic Document "
        "Analysis API"
    ),
    version="1.0.0",
)



# CORS


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "https://academicrag.netlify.app",
    ],
    allow_origin_regex=r"https://.*--academicrag\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Shared generator


generator = ResponseGenerator()


# Shared summary service


summary_service = DocumentSummaryService()

comparison_service = ComparisonService()
research_results_service = (
    ResearchResultsService()
)


# Retriever cache


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



# Helper: convert retrieved chunks into sources


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



# Root


@app.get("/")
def root():

    return {
        "message":
            "Academic RAG API is running."
    }



# Health check


@app.get("/api/health")
def health_check():

    return {
        "status": "ok"
    }



# Ask Question


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

       
        # Calculate similarity statistics
       

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

       
        # Return response
       

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



# Compare all embedding models


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

            
            # Load / get cached retriever
            

            retriever = get_retriever(
                model_name
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
                continue

            
            # Generate answer
            

            answer = (
                generator.generate_answer(
                    question=question,
                    retrieved_chunks=
                        retrieved_chunks,
                )
            )

            
            # Prepare sources
            

            sources = prepare_sources(
                retrieved_chunks
            )

            
            # Similarity scores
            

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

            
            # Store model result
            

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


# Upload PDF Collection


MAX_UPLOAD_FILES = 10
MAX_UPLOAD_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_UPLOAD_TOTAL_SIZE = 50 * 1024 * 1024  # 50 MB

@app.post(
    "/api/upload",
    response_model=UploadResponse,
)
def upload_pdfs(
    files: list[UploadFile] = File(...),
):
    """
    Process one or more PDFs as a single temporary collection.

    Every uploaded PDF is chunked with the same settings. The chunks are
    combined under one document_id so the existing retrieval endpoints can
    search the whole collection without changing their request format.
    """

    try:
        if not files:
            raise HTTPException(
                status_code=400,
                detail="At least one PDF document is required.",
            )

        if len(files) > MAX_UPLOAD_FILES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"A maximum of {MAX_UPLOAD_FILES} PDF documents "
                    "can be uploaded per collection."
                ),
            )

        filenames = []
        total_upload_size = 0

        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="Every uploaded document must have a filename.",
                )

            filename = file.filename.strip()

            if not filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF files are supported: {filename}",
                )

            if (
                file.content_type
                and file.content_type != "application/pdf"
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded file must be a PDF document: {filename}",
                )

            # Validate the actual uploaded file size on the server.
            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(0)

            if file_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded PDF is empty: {filename}",
                )

            if file_size > MAX_UPLOAD_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} exceeds the 10 MB file-size limit.",
                )

            total_upload_size += file_size

            if total_upload_size > MAX_UPLOAD_TOTAL_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The uploaded documents exceed the "
                        "50 MB collection limit."
                    ),
                )

            filenames.append(filename)

        document_id = create_document_id()
        collection_chunks = []
        total_pages = 0
        chunk_size = 800
        chunk_overlap = 150

        for file, filename in zip(files, filenames):
            pdf_path = save_uploaded_pdf(
                file_object=file.file,
                filename=filename,
                document_id=document_id,
            )

            metadata = process_uploaded_pdf(
                pdf_path=pdf_path,
                original_filename=filename,
                document_id=document_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            total_pages += int(metadata.get("pages", 0))

            chunks_path = (
                Path("temp_uploads")
                / document_id
                / "chunks.json"
            )

            with open(
                chunks_path,
                "r",
                encoding="utf-8",
            ) as chunk_file:
                file_chunks = json.load(chunk_file)

            collection_chunks.extend(file_chunks)

        # Re-number chunk IDs where necessary so every chunk in the
        # collection has a unique identifier.
        for index, chunk in enumerate(collection_chunks, start=1):
            chunk["chunk_id"] = (
                f"{document_id}_C{index:05d}"
            )

        collection_folder = (
            Path("temp_uploads")
            / document_id
        )

        with open(
            collection_folder / "chunks.json",
            "w",
            encoding="utf-8",
        ) as chunk_file:
            json.dump(
                collection_chunks,
                chunk_file,
                ensure_ascii=False,
                indent=2,
            )

        collection_metadata = {
            "document_id": document_id,
            "filenames": filenames,
            "document_count": len(filenames),
            "pages": total_pages,
            "chunks": len(collection_chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "status": "processed",
        }

        with open(
            collection_folder / "document.json",
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                collection_metadata,
                metadata_file,
                ensure_ascii=False,
                indent=2,
            )

        return UploadResponse(**collection_metadata)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )



# Ask Uploaded Collection


@app.post(
    "/api/upload/ask",
    response_model=UploadAskResponse,
)
def ask_uploaded_collection(
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


# Compare Models on Uploaded Collection


@app.post(
    "/api/upload/compare",
    response_model=UploadCompareResponse,
)
def compare_uploaded_collection(
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

       
        # Run same question through all models
       

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

            
            # Convert retrieved results to sources
            

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

            
            # Store model comparison result
            

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

       
        # Return comparison
       

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


# Summarise Uploaded Collection


@app.post(
    "/api/upload/summary",
    response_model=UploadSummaryResponse,
)
def summarise_uploaded_collection(
    request: UploadSummaryRequest,
):

    try:

       
        # Full paper
       

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

                filenames=
                    output[
                        "filenames"
                    ],

                document_count=
                    output[
                        "document_count"
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

       
        # Topic focused
       

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


# Compare Topic-Focused Summaries


@app.post(
    "/api/upload/summary/compare",
    response_model=UploadSummaryCompareResponse,
)
def compare_uploaded_collection_summaries(
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


# Research Overview


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


# Research Retrieval Results


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


# Research Generation Results


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