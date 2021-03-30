from typing import Dict, List

import torch

from book_crossing.ncf.recommender import RecommenderSystem


class GMF(RecommenderSystem):
    """Generalized Matrix Factorization"""

    def __init__(self, config: Dict):
        super(GMF, self).__init__(config)

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.hparams.num_users,
                                                 embedding_dim=self.hparams.latent_dim_mf)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.hparams.num_items,
                                                 embedding_dim=self.hparams.latent_dim_mf)

        self.affine_output = torch.nn.Linear(in_features=self.hparams.latent_dim_mf, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        repr_vector = self.forward_repr(user_indices, item_indices)
        logits = self.affine_output(repr_vector)
        rating = self.logistic(logits)
        return rating

    def forward_repr(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        return element_product


class MLP(RecommenderSystem):
    """Multilayer Perceptron model."""

    def __init__(self, config):
        super(MLP, self).__init__(config)

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.hparams.num_users,
                                                 embedding_dim=self.hparams.latent_dim_mlp)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.hparams.num_items,
                                                 embedding_dim=self.hparams.latent_dim_mlp)

        self.fc_layers = torch.nn.Sequential(*self._build_hidden_layers())
        self.affine_output = torch.nn.Linear(in_features=self.hparams.layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        repr_vector = self.forward_repr(user_indices, item_indices)
        logits = self.affine_output(repr_vector)
        ratings = self.logistic(logits)
        return ratings

    def forward_repr(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        out = torch.cat([user_embedding, item_embedding], dim=-1)
        out = self.fc_layers(out)
        return out

    def _build_hidden_layers(self) -> List[torch.nn.Module]:
        assert self.hparams.layers[0] == self.hparams.latent_dim_mlp * 2
        layers = []
        for in_size, out_size in zip(self.hparams.layers, self.hparams.layers[1:]):
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(torch.nn.ReLU())
        return layers


class NeuMF(RecommenderSystem):
    def __init__(self, config: Dict):
        super(NeuMF, self).__init__(config)
        self._gmf = GMF(config)
        self._mlp = MLP(config)

        predictive_factors = self.hparams.layers[-1] + self.hparams.latent_dim_mf
        self.affine_output = torch.nn.Linear(in_features=predictive_factors, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        mf_vector = self._gmf.forward_repr(user_indices, item_indices)
        mlp_vector = self._mlp.forward_repr(user_indices, item_indices)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating
