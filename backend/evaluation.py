import math


# Precision@k
def precision_at_k(predicted_indices, real_indices, k):
    relevant_items = set(predicted_indices[:k]) & set(real_indices)
    return len(relevant_items) / k


# Mean Average Precision (MAP)
def mean_average_precision(predicted_real_pairs):
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


# Discounted Cumulative Gain (DCG)
def dcg(predicted, real, k):
    dcg_score = 0
    for i, pred in enumerate(predicted[:k]):
        if pred in real:
            dcg_score += 1 / math.log2(i + 2)
    return dcg_score


# Normalized Discounted Cumulative Gain (NDCG)
def ndcg(predicted_indices, real_indices, k):
    ideal_dcg = dcg(real_indices, real_indices, k)
    if ideal_dcg == 0:
        return 0
    return dcg(predicted_indices, real_indices, k) / ideal_dcg


# Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(predicted_real_pairs):
    reciprocal_ranks = []
    for predicted_indices, real_indices in predicted_real_pairs:
        rank = 0
        for i, pred in enumerate(predicted_indices):
            if pred in real_indices:
                rank = 1 / (i + 1)
                break
        reciprocal_ranks.append(rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_metrics(predicted_real_pairs, k):
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
    