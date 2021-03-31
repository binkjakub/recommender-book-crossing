from abc import ABC
from typing import Optional, Any, Callable

import torch
from sklearn.metrics import ndcg_score
from torchmetrics import Metric


class RecommenderMetric(Metric, ABC):
    def __init__(
            self,
            k: int = 10,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None):
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.k = k
        self.add_state("user_metric", default=torch.tensor(0., dtype=torch.float),
                       dist_reduce_fx='sum')
        self.add_state("total_users", default=torch.tensor(0, dtype=torch.int),
                       dist_reduce_fx='sum')

    def compute(self):
        return self.user_metric.float() / self.total_users


class HitAtK(RecommenderMetric):

    def update(self, preds: torch.Tensor, ground_truth_ranks: torch.Tensor):
        items_ranking = torch.argsort(preds.flatten(), descending=True)
        (interacted_item, *_), *_ = torch.where(ground_truth_ranks == 1)
        self.user_metric += int(interacted_item in items_ranking[:self.k])
        self.total_users += 1


class NDCGAtK(RecommenderMetric):

    def update(self, preds: torch.Tensor, ground_truth_ranks: torch.Tensor):
        self.user_metric += ndcg_score(ground_truth_ranks, preds.reshape(ground_truth_ranks.shape),
                                       k=self.k)
        self.total_users += 1
