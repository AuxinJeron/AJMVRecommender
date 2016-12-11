from scipy import sparse
from os import path
from AJMvModel import Movie, User
import logging
import pandas as pd

logger = logging.getLogger("logger")


class RatingParser:
    def __init__(self):
        return

    def get_ratings_data(self, fname):
        ratings_contents = pd.read_csv(fname)
        highest_user_id = ratings_contents.user.max()
        highest_movie_id = ratings_contents.movie.max()
        ratings_as_mat = sparse.lil_matrix((highest_user_id, highest_movie_id))
        logger.info("==============================")
        logger.info("Loading data from csv file...")
        logger.info("==============================")
        finished = 0
        for _, row in ratings_contents.iterrows():
            # subtract 1 from id's due to match 0 indexing
            ratings_as_mat[row.user - 1, row.movie - 1] = row.rating
            finished += 1
            if finished % 1000 == 0:
                logger.debug("Has loaded {} items.".format(finished))
        logger.info("Total data items: {}".format(finished))
        return ratings_contents, ratings_as_mat

    def get_movies_data(self, fname):
        file = open(fname, encoding="ISO-8859-1")
        all_categories = set()
        movies = {}
        for line in file:
            words = line.split("::")
            id = words[0]
            name = words[1]
            categories = set()
            for cat in words[2].split("|"):
                categories.add(cat.strip("\n"))
                all_categories.add(cat)
            movies[id] = Movie(id, name, categories)
            logger.debug(movies[id])
        return movies, all_categories

    def get_users_data(self, fname):
        file = open(fname, encoding="ISO-8859-1")
        users = {}
        for line in file:
            words = line.split("::")
            id = words[0]
            gender = words[1]
            age = words[2]
            occupation = words[3]
            users[id] = User(id, gender, age, occupation)
            logger.debug(users[id])
        return users

ratingParser = RatingParser()