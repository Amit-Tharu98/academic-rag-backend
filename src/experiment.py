import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

from src.evaluator import (
    precision_at_k,
    recall_at_k,
    top_k_accuracy,
    reciprocal_rank,
)
from src.retriever import AcademicRetriever


def run_evaluation():

    models = [
        "sentence_transformer",
        "bge",
        "openai",
    ]

    evaluation_file = "data/evaluation_questions.csv"

    print("=" * 80)
    print("EMBEDDING MODEL EVALUATION")
    print("=" * 80)

    # Your file is tab-separated
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

    # Check required columns
    required_columns = {
        "question_id",
        "question",
        "relevant_chunk_ids",
    }

    missing_columns = (
        required_columns
        - set(questions.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Found columns: {questions.columns.tolist()}"
        )

    all_results = []

    for model_name in models:

        print("\n" + "=" * 80)
        print(f"MODEL: {model_name.upper()}")
        print("=" * 80)

        retriever = AcademicRetriever(
            model_name=model_name
        )

        for _, row in questions.iterrows():

            question_id = row["question_id"]
            question = row["question"]

            relevant_ids = set(
                str(
                    row["relevant_chunk_ids"]
                ).split("|")
            )

            print(
                f"\nTesting {question_id}: "
                f"{question}"
            )

            output = retriever.retrieve(
                query=question,
                top_k=5,
            )

            retrieved_ids = [
                result["chunk_id"]
                for result in output["results"]
            ]

            p1 = precision_at_k(
                retrieved_ids,
                relevant_ids,
                1,
            )

            p3 = precision_at_k(
                retrieved_ids,
                relevant_ids,
                3,
            )

            p5 = precision_at_k(
                retrieved_ids,
                relevant_ids,
                5,
            )

            r1 = recall_at_k(
                retrieved_ids,
                relevant_ids,
                1,
            )

            r3 = recall_at_k(
                retrieved_ids,
                relevant_ids,
                3,
            )

            r5 = recall_at_k(
                retrieved_ids,
                relevant_ids,
                5,
            )

            top1 = top_k_accuracy(
                retrieved_ids,
                relevant_ids,
                1,
            )

            top3 = top_k_accuracy(
                retrieved_ids,
                relevant_ids,
                3,
            )

            top5 = top_k_accuracy(
                retrieved_ids,
                relevant_ids,
                5,
            )

            rr = reciprocal_rank(
                retrieved_ids,
                relevant_ids,
            )

            all_results.append(
            {
                "model": model_name,
                "question_id": question_id,
                "relevant_chunk_ids": "|".join(sorted(relevant_ids)),

                "precision@1": p1,
                "precision@3": p3,
                "precision@5": p5,

                "recall@1": r1,
                "recall@3": r3,
                "recall@5": r5,

                "top1_accuracy": top1,
                "top3_accuracy": top3,
                "top5_accuracy": top5,

                "reciprocal_rank": rr,
                "retrieval_time": output["retrieval_time"],

                "retrieved_rank_1": retrieved_ids[0] if len(retrieved_ids) > 0 else "",
                "retrieved_rank_2": retrieved_ids[1] if len(retrieved_ids) > 1 else "",
                "retrieved_rank_3": retrieved_ids[2] if len(retrieved_ids) > 2 else "",
                "retrieved_rank_4": retrieved_ids[3] if len(retrieved_ids) > 3 else "",
                "retrieved_rank_5": retrieved_ids[4] if len(retrieved_ids) > 4 else "",
            }
        )

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        "results/retrieval_results.csv",
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("AVERAGE RESULTS")
    print("=" * 80)

    averages = (
        results_df
        .groupby("model")
        .mean(numeric_only=True)
        .round(4)
    )

    print(averages)

    averages.to_csv(
        "results/retrieval_summary.csv"
    )

    print("\nResults saved to:")
    print("results/retrieval_results.csv")
    print("results/retrieval_summary.csv")


if __name__ == "__main__":
    run_evaluation()