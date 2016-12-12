from scipy import sparse
from time import time
from AJMvModel import Movie, User
import logging
import numpy as np
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
        movies = []
        for line in file:
            words = line.split("::")
            id = words[0]
            name = words[1]
            categories = set()
            for cat in words[2].split("|"):
                categories.add(cat.strip("\n"))
                all_categories.add(cat)
            movies.append(Movie(id, name, categories))
            #logger.debug(movies[id])
        return movies, all_categories

    def get_users_data(self, fname):
        file = open(fname, encoding="ISO-8859-1")
        users = []
        for line in file:
            words = line.split("::")
            id = words[0]
            gender = words[1]
            age = words[2]
            occupation = words[3]
            users.append(User(id, gender, age, occupation))
            #logger.debug(users[id])
        return users

    def get_submission_data(self, fname, user_num, movie_num):
        file = open(fname, encoding="ISO-8859-1")
        ratings_as_mat = sparse.lil_matrix((user_num, movie_num))

        for line in file:
            words = line.split(",")
            if words[0] == "user":
                continue
            user = int(float(words[0])) - 1
            rating = float(words[1])
            tmp = words[2].split("_")
            movie = int(float(tmp[1].strip("\n"))) - 1
            ratings_as_mat[user, movie] = rating

        return ratings_as_mat.todense()

    def store_similarity_data(self, similarity_mat, fname, report_run_time=False):
        start_time = time()
        file = open(fname, 'w+')
        num, _ = similarity_mat.shape
        for i in range(num):
            for j in range(i+1, num):
                user1_id = int(i+1)
                user2_id = int(j+1)
                score = similarity_mat[i, j]
                line = "{},{},{}\n".format(user1_id, user2_id, score)
                file.write(line)
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))

    def store_similarity_csv(self, similarity_mat, fname, report_run_time=False):
        start_time = time()
        num, _ = similarity_mat.shape
        user1_col = np.zeros(num * num, dtype=int)
        user2_col = np.zeros(num * num, dtype=int)
        for i in range(num):
            for j in range(i+1, num):
                user1_col[num * i + j] = int(i+1)
                user2_col[num * i + j] = int(j+1)
        d = {"user1": user1_col, "user2": user2_col}
        df = pd.DataFrame(data=d)
        df["score"] = df.apply(lambda x: similarity_mat[x["user1"] - 1, x["user2"] - 1], axis=1)
        df.to_csv(fname, index=False)
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))

ratingParser = RatingParser()