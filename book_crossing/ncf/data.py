import random
from typing import Optional, Tuple, Any, List, Union, Dict

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
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

    def setup(self, stage: Optional[str] = None):
        ratings = self._load_ratings()
        ratings = self._filter_ratings(ratings)
        print(f"#users: {ratings['User-ID'].nunique()}, #items: {ratings['ISBN'].nunique()}")
        ratings = self._to_implicit_feedback(ratings)
        train_ratings, test_ratings = self._split_leave_one_out(ratings)
        # self.train_ratings = self._build_dataset(ratings)
        self.train_ratings = self._build_dataset(train_ratings)

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_ratings, batch_size=self._batch_size, shuffle=True,
                          num_workers=self._num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        pass

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
        ratings['Book-Rating'] = ratings['Book-Rating'].clip(0, 1)
        return ratings

    def _build_dataset(self, ratings: pd.DataFrame) -> UserItemRatingDataset:
        negatives = self._sample_negatives(ratings)
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

    def _sample_negatives(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Returns all negative items & 100 sampled negative items"""
        item_pool = set(ratings['ISBN'].unique())
        interact_status = ratings.groupby('User-ID')['ISBN'].apply(set).reset_index().rename(
            columns={'ISBN': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(
            lambda x: item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(
            lambda x: random.sample(x, 99))
        return interact_status[['User-ID', 'negative_items', 'negative_samples']]
