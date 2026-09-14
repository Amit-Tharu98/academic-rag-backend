def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.
    """

    retrieved_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1
        for chunk_id in retrieved_k
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.
    """

    if not relevant_ids:
        return 0.0

    retrieved_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1
        for chunk_id in retrieved_k
        if chunk_id in relevant_ids
    )

    return (
        relevant_retrieved
        / len(relevant_ids)
    )


def top_k_accuracy(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> int:
    """
    Return 1 if at least one relevant chunk
    appears in the top K results.
    Otherwise return 0.
    """

    retrieved_k = retrieved_ids[:k]

    for chunk_id in retrieved_k:
        if chunk_id in relevant_ids:
            return 1

    return 0


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Calculate Reciprocal Rank.

    Example:
    relevant result at rank 1 -> 1.0
    rank 2 -> 0.5
    rank 3 -> 0.333
    """

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0