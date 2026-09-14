import os

import pandas as pd

from src.generator import ResponseGenerator
from src.generation_evaluator import calculate_rouge
from src.retriever import AcademicRetriever


def run_generation_evaluation():

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------
    models = [
        "sentence_transformer",
        "bge",
        "openai",
    ]

    evaluation_file = (
        "data/evaluation_questions.csv"
    )

    results_file = (
        "results/generation_results.csv"
    )

    summary_file = (
        "results/generation_summary.csv"
    )

    top_k = 5

    # Ensure results directory exists
    os.makedirs(
        "results",
        exist_ok=True,
    )

    print("=" * 80)
    print("FINAL GENERATION EVALUATION")
    print("=" * 80)

    # -------------------------------------------------
    # Load evaluation questions
    # -------------------------------------------------
    questions = pd.read_csv(
        evaluation_file,
        sep="\t",
    )

    # Clean column names
    questions.columns = (
        questions.columns
        .str.strip()
        .str.lower()
    )

    required_columns = {
        "question_id",
        "question",
        "reference_answer",
    }

    missing_columns = (
        required_columns
        - set(questions.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: "
            f"{missing_columns}. "
            f"Found columns: "
            f"{questions.columns.tolist()}"
        )

    print(
        f"Evaluation questions loaded: "
        f"{len(questions)}"
    )

    print(
        f"Models: {len(models)}"
    )

    print(
        f"Expected generations: "
        f"{len(questions) * len(models)}"
    )

    # -------------------------------------------------
    # Shared generator
    # -------------------------------------------------
    # Same LLM is used for every embedding model.
    generator = ResponseGenerator()

    all_results = []

    # -------------------------------------------------
    # Evaluate each embedding model
    # -------------------------------------------------
    for model_name in models:

        print("\n" + "=" * 80)
        print(
            f"MODEL: {model_name.upper()}"
        )
        print("=" * 80)

        retriever = AcademicRetriever(
            model_name=model_name
        )

        # ---------------------------------------------
        # Evaluate every question
        # ---------------------------------------------
        for question_number, (
            _,
            row,
        ) in enumerate(
            questions.iterrows(),
            start=1,
        ):

            question_id = str(
                row["question_id"]
            ).strip()

            question = str(
                row["question"]
            ).strip()

            reference_answer = str(
                row["reference_answer"]
            ).strip()

            print(
                f"\n[{question_number}/"
                f"{len(questions)}] "
                f"Processing "
                f"{question_id}"
            )

            print(
                f"Question: {question}"
            )

            # -----------------------------------------
            # Retrieve top-k chunks
            # -----------------------------------------
            retrieval_output = (
                retriever.retrieve(
                    query=question,
                    top_k=top_k,
                )
            )

            retrieved_chunks = (
                retrieval_output["results"]
            )

            print(
                f"Retrieved "
                f"{len(retrieved_chunks)} "
                f"chunks."
            )

            # -----------------------------------------
            # Generate answer
            # -----------------------------------------
            generated_answer = (
                generator.generate_answer(
                    question=question,
                    retrieved_chunks=
                    retrieved_chunks,
                )
            )

            print(
                "Generated answer."
            )

            # -----------------------------------------
            # Calculate ROUGE
            # -----------------------------------------
            rouge_scores = (
                calculate_rouge(
                    reference_answer=
                    reference_answer,
                    generated_answer=
                    generated_answer,
                )
            )

            print(
                f"ROUGE-1: "
                f"{rouge_scores['rouge1']:.4f}"
            )

            print(
                f"ROUGE-2: "
                f"{rouge_scores['rouge2']:.4f}"
            )

            print(
                f"ROUGE-L: "
                f"{rouge_scores['rougeL']:.4f}"
            )

            # -----------------------------------------
            # Get retrieved chunk IDs
            # -----------------------------------------
            retrieved_ids = [
                chunk["chunk_id"]
                for chunk
                in retrieved_chunks
            ]

            # -----------------------------------------
            # Store result
            # -----------------------------------------
            result = {
                "model":
                    model_name,

                "question_id":
                    question_id,

                "question":
                    question,

                "reference_answer":
                    reference_answer,

                "generated_answer":
                    generated_answer,

                "rouge1":
                    rouge_scores["rouge1"],

                "rouge2":
                    rouge_scores["rouge2"],

                "rougeL":
                    rouge_scores["rougeL"],

                "retrieval_time":
                    retrieval_output[
                        "retrieval_time"
                    ],

                "retrieved_rank_1":
                    (
                        retrieved_ids[0]
                        if len(
                            retrieved_ids
                        ) > 0
                        else ""
                    ),

                "retrieved_rank_2":
                    (
                        retrieved_ids[1]
                        if len(
                            retrieved_ids
                        ) > 1
                        else ""
                    ),

                "retrieved_rank_3":
                    (
                        retrieved_ids[2]
                        if len(
                            retrieved_ids
                        ) > 2
                        else ""
                    ),

                "retrieved_rank_4":
                    (
                        retrieved_ids[3]
                        if len(
                            retrieved_ids
                        ) > 3
                        else ""
                    ),

                "retrieved_rank_5":
                    (
                        retrieved_ids[4]
                        if len(
                            retrieved_ids
                        ) > 4
                        else ""
                    ),
            }

            all_results.append(
                result
            )

            # -----------------------------------------
            # Save progress after every question
            # -----------------------------------------
            pd.DataFrame(
                all_results
            ).to_csv(
                results_file,
                index=False,
            )

            print(
                "Progress saved."
            )

    # -------------------------------------------------
    # Create final dataframe
    # -------------------------------------------------
    results_df = pd.DataFrame(
        all_results
    )

    # Final detailed save
    results_df.to_csv(
        results_file,
        index=False,
    )

    # -------------------------------------------------
    # Calculate average generation results
    # -------------------------------------------------
    summary_df = (
        results_df
        .groupby("model")[
            [
                "rouge1",
                "rouge2",
                "rougeL",
                "retrieval_time",
            ]
        ]
        .mean()
        .round(4)
    )

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------
    print("\n")
    print("=" * 80)
    print("AVERAGE GENERATION RESULTS")
    print("=" * 80)

    print(
        summary_df
    )

    # -------------------------------------------------
    # Save summary
    # -------------------------------------------------
    summary_df.to_csv(
        summary_file
    )

    print("\nResults saved to:")

    print(
        results_file
    )

    print(
        summary_file
    )


if __name__ == "__main__":
    run_generation_evaluation()