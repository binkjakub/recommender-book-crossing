from typing import Dict

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from book_crossing.ncf.neumf import NeuMF


class RecommenderSystem(LightningModule):
    def __init__(self, config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(config)
        self.neu_mf = NeuMF(config)
        self.criterion = torch.nn.BCELoss()

    def forward(self, x):
        return self.neu_mf(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, train_batch, batch_id):
        users, items, ratings = train_batch
        ratings_pred = self.neu_mf(users, items)
        loss = F.binary_cross_entropy(ratings_pred.view(-1), ratings)
        self.log('train_loss', loss)
        return loss
