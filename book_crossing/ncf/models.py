from typing import Dict

import torch

from book_crossing.ncf.recommender import RecommenderSystem


class GMF(RecommenderSystem):
    """Generalized Matrix Factorization"""

    def __init__(self, config: Dict):
        super(GMF, self).__init__(config)
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim_mf']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users,
                                                 embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items,
                                                 embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating


class MLP(RecommenderSystem):
    """MLP model from NCF."""

    def __init__(self, config):
        super(MLP, self).__init__(config)

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.hparams.num_users,
                                                 embedding_dim=self.hparams.latent_dim_mlp)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.hparams.num_items,
                                                 embedding_dim=self.hparams.latent_dim_mlp)

        assert self.hparams.layers[0] == self.hparams.latent_dim_mlp * 2
        layers = [
            torch.nn.Linear(in_size, out_size)
            for in_size, out_size in zip(self.hparams.layers, self.hparams.layers[1:])
        ]
        self.fc_layers = torch.nn.Sequential(*layers)

        self.affine_output = torch.nn.Linear(in_features=self.hparams.layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        out = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        out = self.fc_layers(out)
        out = self.affine_output(out)
        out = self.logistic(out)
        return out


class NeuMF(RecommenderSystem):
    def __init__(self, config: Dict):
        super(NeuMF, self).__init__(config)
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users,
                                                     embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items,
                                                     embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users,
                                                    embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items,
                                                    embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(
            in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp],
                               dim=-1)  # the concat latent vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating
