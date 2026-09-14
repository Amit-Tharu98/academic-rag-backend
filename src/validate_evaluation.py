import json
import pandas as pd


EVALUATION_FILE = "data/evaluation_questions.csv"
METADATA_FILE = "indexes/sentence_transformer/metadata.json"


def validate_evaluation_dataset():
    print("=" * 70)
    print("VALIDATING FINAL EVALUATION DATASET")
    print("=" * 70)

    # -------------------------------------------------
    # 1. Load evaluation questions
    # -------------------------------------------------
    questions = pd.read_csv(
        EVALUATION_FILE,
        sep="\t"
    )

    questions.columns = questions.columns.str.strip().str.lower()

    print(f"\nEvaluation questions loaded: {len(questions)}")

    # -------------------------------------------------
    # 2. Check required columns
    # -------------------------------------------------
    required_columns = {
        "question_id",
        "question",
        "relevant_chunk_ids",
        "reference_answer",
    }

    missing_columns = required_columns - set(questions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("Required columns: OK")

    # -------------------------------------------------
    # 3. Load final chunk metadata
    # -------------------------------------------------
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    print(f"Final chunks loaded: {len(metadata)}")

    # Create set of all valid chunk IDs
    valid_chunk_ids = {
        item["chunk_id"]
        for item in metadata
    }

    # -------------------------------------------------
    # 4. Check duplicate question IDs
    # -------------------------------------------------
    duplicate_questions = questions[
        questions["question_id"].duplicated()
    ]

    if not duplicate_questions.empty:
        print("\nWARNING: Duplicate question IDs found:")
        print(
            duplicate_questions[
                ["question_id", "question"]
            ]
        )
    else:
        print("Duplicate question IDs: None")

    # -------------------------------------------------
    # 5. Check empty values
    # -------------------------------------------------
    required_fields = [
        "question_id",
        "question",
        "relevant_chunk_ids",
        "reference_answer",
    ]

    for column in required_fields:

        empty_count = (
            questions[column]
            .isna()
            .sum()
        )

        if empty_count > 0:
            print(
                f"WARNING: {column} has "
                f"{empty_count} empty values."
            )
        else:
            print(
                f"Empty {column}: None"
            )

    # -------------------------------------------------
    # 6. Validate relevant chunk IDs
    # -------------------------------------------------
    missing_chunk_ids = []

    total_relevance_labels = 0

    for _, row in questions.iterrows():

        relevant_ids = [
            chunk_id.strip()
            for chunk_id
            in str(
                row["relevant_chunk_ids"]
            ).split("|")
            if chunk_id.strip()
        ]

        total_relevance_labels += len(
            relevant_ids
        )

        for chunk_id in relevant_ids:

            if chunk_id not in valid_chunk_ids:

                missing_chunk_ids.append({
                    "question_id":
                        row["question_id"],

                    "missing_chunk_id":
                        chunk_id,
                })

    # -------------------------------------------------
    # 7. Final report
    # -------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Questions: {len(questions)}"
    )

    print(
        f"Final chunks: {len(metadata)}"
    )

    print(
        f"Ground-truth relevance labels: "
        f"{total_relevance_labels}"
    )

    print(
        f"Missing relevant chunk IDs: "
        f"{len(missing_chunk_ids)}"
    )

    if missing_chunk_ids:

        print("\nMissing IDs:")

        for item in missing_chunk_ids:

            print(
                item["question_id"],
                "->",
                item["missing_chunk_id"]
            )

        print(
            "\nVALIDATION FAILED."
        )

    else:

        print(
            "\nAll relevant chunk IDs exist "
            "in the final FAISS metadata."
        )

        print(
            "VALIDATION PASSED."
        )


if __name__ == "__main__":
    validate_evaluation_dataset()