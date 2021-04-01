import bottleneck as bn
import numpy as np


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    """
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance

    idx_topk = np.argsort(X_pred, axis=1)[:, -k:][:, ::-1]
    """
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    # idx_topk = np.argsort(X_pred, axis=1)[:, -k:][:, ::-1]
    tp = 1. / np.log2(np.arange(2, k + 2))

    # DCG = (np.take_along_axis(heldout_batch.toarray(), idx_topk, 1) * tp).sum(axis=1)
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    result = DCG / IDCG
    return result


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    numerator = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    denominator = np.minimum(k, X_true_binary.sum(axis=1))
    numerator = numerator[denominator != 0]
    denominator = denominator[denominator != 0]
    recall = numerator / denominator
    # recall = tmp[denominator != 0 ] / np.minimum(k, X_true_binary.sum(axis=1))
    return recall
