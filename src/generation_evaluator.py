from rouge_score import rouge_scorer


def calculate_rouge(
    reference_answer: str,
    generated_answer: str,
):
    """
    Calculate ROUGE-1, ROUGE-2 and ROUGE-L.
    """

    scorer = rouge_scorer.RougeScorer(
        [
            "rouge1",
            "rouge2",
            "rougeL",
        ],
        use_stemmer=True,
    )

    scores = scorer.score(
        reference_answer,
        generated_answer,
    )

    return {
        "rouge1":
            scores["rouge1"].fmeasure,

        "rouge2":
            scores["rouge2"].fmeasure,

        "rougeL":
            scores["rougeL"].fmeasure,
    }