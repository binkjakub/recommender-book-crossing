from abc import ABC
from typing import Dict

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from book_crossing.ncf.metrics import HitAtK, NDCGAtK


class RecommenderSystem(LightningModule, ABC):
    def __init__(self, config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(config)

        self.k_eval = 10
        self.hit_at_k = HitAtK(k=self.k_eval)
        self.ndcg_at_k = NDCGAtK(k=self.k_eval)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, train_batch, batch_id):
        users, items, ratings = train_batch
        ratings_pred = self.forward(users, items)
        loss = F.binary_cross_entropy(ratings_pred.view(-1), ratings)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_id):
        users, items, ratings = val_batch
        ratings_pred = self.forward(users, items)
        self.log(f'hit@{self.k_eval}', self.hit_at_k(ratings_pred, ratings), on_epoch=True,
                 prog_bar=True)
        self.log(f'ndcg@{self.k_eval}', self.ndcg_at_k(ratings_pred, ratings), on_epoch=True,
                 prog_bar=True)
