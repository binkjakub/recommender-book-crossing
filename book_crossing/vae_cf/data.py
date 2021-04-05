import os
import pandas as pd
from scipy import sparse
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load given dataset (Movielens-20m and Book-Crossing)."""

    def __init__(self, path):
        self.pro_dir = os.path.join(path, 'pro_sg')
        assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"

        self.n_items = self.load_n_items()

    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items

    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')

        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data

    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                     (rows_tr, cols_tr)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                     (rows_te, cols_te)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te


def get_count_df(triplet_data, col):
    vc = pd.DataFrame(triplet_data[col].value_counts()).reset_index()
    vc.columns = [col, 'count']
    return vc


def filter_triplets(tp, min_uc=5, min_ic=5):
    if min_uc > 0:
        user_count = get_count_df(tp, USER_COL)
        tp = tp[tp[USER_COL].isin(user_count[user_count['count'] > min_uc][USER_COL])]

    if min_ic > 0:
        item_count = get_count_df(tp, ITEM_COL)
        tp = tp[tp[ITEM_COL].isin(item_count[item_count['count'] > min_ic][ITEM_COL])]
    return tp


def filter_ratings(triplet_data: pd.DataFrame, item_cut_off: int = 1, user_cut_off: int = 5) -> pd.DataFrame:
    """Filters out interaction of user and books having #interactions below the threshold."""
    item_mask = (triplet_data[ITEM_COL].map(triplet_data[ITEM_COL].value_counts())
                 >= item_cut_off)
    triplet_data = triplet_data[item_mask]

    user_mask = (triplet_data[USER_COL].map(triplet_data[USER_COL].value_counts())
                 >= user_cut_off)
    triplet_data = triplet_data[user_mask]

    # project ids to indices - make index-space compact
    triplet_data[ITEM_COL] = triplet_data[ITEM_COL].astype('category').cat.codes
    triplet_data[USER_COL] = triplet_data[USER_COL].astype('category').cat.codes

    return triplet_data


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby(USER_COL)
    tr_list, te_list = list(), list()

    np.random.seed(SEED)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype(
                'int64')] = True
            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = tp[USER_COL].apply(lambda x: profile2id[x])
    sid = tp[ITEM_COL].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


def read_preprocess_book_crossing_dataset(data_dir, rating_file):
    """
    Reads and preprocess the data.

    File contains the book rating information. Ratings (`Book-Rating`) are either explicit,
    expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit,
    expressed by 0.
    We assign arbitrary 6 value which is medium high RATING (mode is 8), mean is 7.61.
    """
    data = pd.read_csv(os.path.join(data_dir, rating_file), sep=';')
    data[RATING_COL] = data[RATING_COL].replace(0, 7)
    return data


def read_preprocess_movielens(data_dir, rating_file):
    data = pd.read_csv(os.path.join(data_dir, rating_file), header=0)
    return data


if __name__ == '__main__':
    print("Started generating the dataset..")

    DATASET = 'book-crossing'

    if DATASET == 'ml-latest':
        USER_COL = 'userId'
        ITEM_COL = 'movieId'
        RATING_COL = 'rating'
        DATA_DIR = '../ml-latest/'
        RATINGS_FILE = 'ratings.csv'
        raw_data = read_preprocess_movielens(DATA_DIR, RATINGS_FILE)

    elif DATASET == 'book-crossing':
        USER_COL = 'User-ID'
        ITEM_COL = 'ISBN'
        RATING_COL = 'Book-Rating'
        DATA_DIR = '../book-crossing/'
        RATINGS_FILE = 'BX-Book-Ratings.csv'
        raw_data = read_preprocess_book_crossing_dataset(DATA_DIR, RATINGS_FILE)

    else:
        raise ValueError("wrong dataset")

    TEST_SIZE = 0.2  # no users in test and validation sets
    SEED = 42

    # raw_data = read_preprocess_book_crossing_dataset(DATA_DIR, RATINGS_FILE)

    # # Filter Data
    # raw_data = filter_triplets(raw_data, min_uc=10, min_ic=10)
    raw_data = raw_data[raw_data[RATING_COL] > 5]
    raw_data = filter_ratings(raw_data, user_cut_off=10, item_cut_off=10)

    unique_uid = raw_data[USER_COL].unique()
    n_users = unique_uid.size
    print(f"#users: {n_users}, #items: {raw_data[ITEM_COL].nunique()}")

    tr_users, te_users = train_test_split(unique_uid, test_size=TEST_SIZE, random_state=SEED)
    vd_users, te_users = train_test_split(te_users, test_size=0.5, random_state=SEED)

    train_plays = raw_data.loc[raw_data[USER_COL].isin(tr_users)]
    unique_sid = train_plays[ITEM_COL].unique()

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(DATA_DIR, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    vad_plays = raw_data.loc[raw_data[USER_COL].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays[ITEM_COL].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data[USER_COL].isin(te_users)]
    test_plays = test_plays.loc[test_plays[ITEM_COL].isin(unique_sid)]
    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

    print("Done!")
