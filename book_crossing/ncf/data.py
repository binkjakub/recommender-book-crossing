import random
from typing import Optional, Tuple, Any, List, Union, Dict

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor: Tensor, item_tensor: Tensor, target_tensor: Tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class UserItemEvaluationDataset(Dataset):
    """Wrapper for evaluation data, as recommendations are evaluated with 100 negative samples:
    Data should contain:
        - examples of positive interactions
        - subsample of negative interactions (preferably 100, as in NCF publication)
    """

    def __init__(self, test_users: Tensor, test_items: Tensor, negative_users: Tensor,
                 negative_items: Tensor):
        self._users_idx = torch.unique(test_users)

        self.test_users = test_users
        self.test_items = test_items
        self.negative_users = negative_users
        self.negative_items = negative_items

    def __getitem__(self, user_index):
        positive_items = self.test_items[self.test_users == user_index]
        negative_items = self.negative_items[self.negative_users == user_index]
        total_len = len(positive_items) + len(negative_items)

        items = torch.cat([positive_items, negative_items])
        users = torch.full((total_len,), user_index, dtype=torch.int)
        ratings = torch.tensor(([1] * len(positive_items) + ([0] * len(negative_items))),
                               dtype=torch.int)
        return users, items, ratings

    def __len__(self):
        return self._users_idx.size(0)


class BookCrossingDM(LightningDataModule):
    """Datamodule responsible for preparation of dataset to be used in training of NCF model."""

    def __init__(
            self,
            ratings_path: str,
            config: Dict,
    ):
        super().__init__()
        self._ratings_path = ratings_path

        self._fraction_of_users = config['users_fraction']
        self._num_negatives = config['num_negatives']
        self._book_interactions_cutoff = config['book_interactions_cutoff']
        self._user_interaction_cutoff = config['user_interaction_cutoff']
        self._batch_size = config['batch_size']

        self._num_workers = config['num_workers']
        self._num_users, self._num_items = config['num_users'], config['num_items']

        self.train_ratings = None
        self.test_ratings = None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        ratings = self._load_ratings()
        ratings = self._filter_ratings(ratings)
        print(f"#users: {ratings['User-ID'].nunique()}, #items: {ratings['ISBN'].nunique()}")
        print(f"#interactions: {len(ratings)}")
        print("sparsity: "
              f"{len(ratings) / (ratings['User-ID'].nunique() * ratings['ISBN'].nunique()) :0.3f}")
        ratings = self._to_implicit_feedback(ratings)
        train_ratings, test_ratings = self._split_leave_one_out(ratings)

        negatives = self._sample_negatives(ratings)
        self.train_ratings = self._build_train_dataset(train_ratings, negatives)
        self.test_ratings = self._build_test_dataset(test_ratings, negatives)

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_ratings, batch_size=self._batch_size, shuffle=True,
                          num_workers=self._num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_ratings, batch_size=1, shuffle=False,
                          num_workers=self._num_workers)

    def _split_leave_one_out(self, ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits ratings into train/test set delegating last interaction to the test set."""
        ratings_grouped = ratings.groupby(['User-ID'])
        assert (ratings_grouped.size() >= 2).all()

        test_indices = ratings_grouped['User-ID'].sample(n=1).index
        train_indices = ratings.index.difference(test_indices)

        test = ratings.loc[test_indices]
        train = ratings.loc[train_indices]

        assert (train['User-ID'] < self._num_users).all()
        assert (train['ISBN'] < self._num_items).all()
        assert np.in1d(train['User-ID'].unique(), test['User-ID'].unique()).all()

        return train, test

    def _load_ratings(self) -> pd.DataFrame:
        """Load interaction matrix: Books rated by Users"""
        ratings = pd.read_csv(self._ratings_path, sep=';')

        assert 'User-ID' in ratings.columns
        assert 'ISBN' in ratings.columns
        assert 'Book-Rating' in ratings.columns

        ratings['Book-Rating'] = ratings['Book-Rating'].astype('int8')

        return ratings

    def _filter_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Filters out interaction of user and books having #interactions below the threshold."""
        book_mask = (ratings['ISBN'].map(ratings['ISBN'].value_counts())
                     >= self._book_interactions_cutoff)
        ratings = ratings[book_mask]

        user_mask = (ratings['User-ID'].map(ratings['User-ID'].value_counts())
                     >= self._user_interaction_cutoff)
        ratings = ratings[user_mask]

        if self._fraction_of_users < 1.0:
            unique_users = ratings['User-ID'].unique()
            sample_size = int(self._fraction_of_users * len(unique_users))
            sampled_users = np.random.choice(unique_users,
                                             size=sample_size,
                                             replace=False)
            ratings = ratings[ratings['User-ID'].isin(sampled_users)]

        # project ids to indices - make index-space compact
        ratings['ISBN'] = ratings['ISBN'].astype('category').cat.codes
        ratings['User-ID'] = ratings['User-ID'].astype('category').cat.codes

        return ratings

    def _to_implicit_feedback(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Performs binarization on interaction matrix to cast feedback into implicit."""
        ratings['Book-Rating'] = 1
        return ratings

    def _build_train_dataset(self, ratings: pd.DataFrame,
                             negatives: pd.DataFrame) -> UserItemRatingDataset:
        """Creates training data."""
        train_ratings = pd.merge(ratings, negatives[['User-ID', 'negative_items']], on='User-ID')

        train_ratings['negatives'] = train_ratings['negative_items'].apply(
            lambda x: random.sample(x, self._num_negatives))

        users, items, ratings = [], [], []
        for _, row in train_ratings.iterrows():
            users.append(int(row['User-ID']))
            items.append(int(row['ISBN']))
            ratings.append(float(row['Book-Rating']))
            for i in range(self._num_negatives):
                users.append(int(row['User-ID']))
                items.append(int(row['negatives'][i]))
                ratings.append(float(0))

        return UserItemRatingDataset(
            torch.tensor(users, dtype=torch.int),
            torch.tensor(items, dtype=torch.int),
            torch.tensor(ratings, dtype=torch.float),
        )

    def _build_test_dataset(self,
                            test_ratings: pd.DataFrame,
                            negatives: pd.DataFrame) -> UserItemEvaluationDataset:
        """Creates evaluation data."""
        test_ratings = pd.merge(test_ratings,
                                negatives[['User-ID', 'test_negatives']],
                                on='User-ID')

        test_users, test_items, negative_users, negative_items = [], [], [], []

        for _, row in test_ratings.iterrows():
            test_users.append(int(row['User-ID']))
            test_items.append(int(row['ISBN']))
            for i in range(len(row['test_negatives'])):
                negative_users.append(int(row['User-ID']))
                negative_items.append(int(row['test_negatives'][i]))

        return UserItemEvaluationDataset(
            torch.tensor(test_users, dtype=torch.int),
            torch.tensor(test_items, dtype=torch.int),
            torch.tensor(negative_users, dtype=torch.int),
            torch.tensor(negative_items, dtype=torch.int),
        )

    def _sample_negatives(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Returns all negative items & 100 sampled negative items (for evaluation purposes)."""
        item_pool = set(ratings['ISBN'].unique())
        interact_status = ratings.groupby('User-ID')['ISBN'].apply(set).reset_index().rename(
            columns={'ISBN': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(
            lambda x: item_pool - x)
        interact_status['test_negatives'] = interact_status['negative_items'].apply(
            lambda x: random.sample(x, 99))
        return interact_status[['User-ID', 'negative_items', 'test_negatives']]
