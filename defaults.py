import os

from pathlib import Path

DATA_PATH = Path(os.path.dirname(__file__)).joinpath('storage')

BOOK_RATINGS = DATA_PATH.joinpath('BX-Book-Ratings.csv')
BOOKS = DATA_PATH.joinpath('BX-Books.csv')
USERS = DATA_PATH.joinpath('BX-Users.csv')
