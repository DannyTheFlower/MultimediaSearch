import math
from typing import List, Tuple, Set


def precision_at_k(predicted_indices: List[int], real_indices: Set[int], k: int) -> float:
    """
    Computes Precision@k.

    :param predicted_indices: List of predicted indices.
    :param real_indices: Set of real indices.
    :param k: Cut-off rank.
    :return: Precision at k.
    """
    relevant_items = set(predicted_indices[:k]) & set(real_indices)
    return len(relevant_items) / k


def mean_average_precision(predicted_real_pairs: List[Tuple[List[int], Set[int]]]) -> float:
    """
    Computes Mean Average Precision (MAP).

    :param predicted_real_pairs: List of tuples containing predicted and real indices.
    :return: MAP score.
    """
    avg_precisions = []
    for predicted_indices, real_indices in predicted_real_pairs:
        if not real_indices:
            avg_precisions.append(0)
            continue
        precisions = []
        relevant_count = 0
        for i, pred in enumerate(predicted_indices):
            if pred in real_indices:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        avg_precisions.append(sum(precisions) / len(real_indices) if precisions else 0)
    return sum(avg_precisions) / len(predicted_real_pairs)


def dcg(predicted: List[int], real: Set[int], k: int) -> float:
    """
    Computes Discounted Cumulative Gain (DCG).

    :param predicted: List of predicted indices.
    :param real: Set of real indices.
    :param k: Cut-off rank.
    :return: DCG score.
    """
    dcg_score = 0
    for i, pred in enumerate(predicted[:k]):
        if pred in real:
            dcg_score += 1 / math.log2(i + 2)
    return dcg_score


def ndcg(predicted_indices: List[int], real_indices: Set[int], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG).

    :param predicted_indices: List of predicted indices.
    :param real_indices: Set of real indices.
    :param k: Cut-off rank.
    :return: NDCG score.
    """
    ideal_dcg = dcg(real_indices, real_indices, k)
    if ideal_dcg == 0:
        return 0
    return dcg(predicted_indices, real_indices, k) / ideal_dcg


def mean_reciprocal_rank(predicted_real_pairs: List[Tuple[List[int], Set[int]]]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR).

    :param predicted_real_pairs: List of tuples containing predicted and real indices.
    :return: MRR score.
    """
    reciprocal_ranks = []
    for predicted_indices, real_indices in predicted_real_pairs:
        rank = 0
        for i, pred in enumerate(predicted_indices):
            if pred in real_indices:
                rank = 1 / (i + 1)
                break
        reciprocal_ranks.append(rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_metrics(predicted_real_pairs: List[Tuple[List[int], Set[int]]], k: int) -> dict:
    """
    Computes various evaluation metrics.

    :param predicted_real_pairs: List of tuples containing predicted and real indices.
    :param k: Cut-off rank.
    :return: Dictionary of computed metrics.
    """
    precision_scores = [precision_at_k(pred, real, k) for pred, real in predicted_real_pairs]
    map_score = mean_average_precision(predicted_real_pairs)
    ndcg_scores = [ndcg(pred, real, k) for pred, real in predicted_real_pairs]
    mrr_score = mean_reciprocal_rank(predicted_real_pairs)

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

    return {
        f'Precision@{k}': avg_precision,
        'MAP': map_score,
        f'NDCG@{k}': avg_ndcg,
        'MRR': mrr_score
    }
    